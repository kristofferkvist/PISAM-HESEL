/******************************************************************************/
/*                                NEUTRALS.CXX                                */
/******************************************************************************/

/*
 * Neutrals.cxx
 *
 * Module for calculating neutral transport and source. The interactions include
 * elastic collisions, molecule dissociation, atom and molecule ionization, and
 * charge exchange between atoms created by dissociation of molecules and ions.
 *
 * Created on: Oct 22, 2014
 *     Author: yolen
 *
 * Change Log
 *     2018-11-05 by alec:
 *         1. Major clean-up of redundant/deprecated code
 *         2. Preparation for compatibility with 3D HESEL
 *         3. Inclusion of elastic collisions
 */

/******************************************************************************/
/*                                CONSTRUCTOR                                 */
/******************************************************************************/

#include "Neutrals.hxx"

Neutrals::Neutrals(
  Solver *i_solver,
  const HeselParameters &i_hesel_para_,
  const Field3D &i_n,
  const Field3D &i_phi,
  const Field3D &i_Pi,
  const Field3D &i_pe,
  const Field2D &i_B) :
  // The class members below are references and hence must be initialized at object instantiation
  solver(i_solver),
  hesel_para_(i_hesel_para_),
  n(i_n),
  phi(i_phi),
  Pi(i_Pi),
  pe(i_pe),
  B(i_B){
    u_perp0 = 0.;
    u_perp0 = 0.;
  }

/******************************************************************************/
/*                             MODULE INITIALIZER                             */
/******************************************************************************/

int Neutrals::InitNeutrals(){

/*                   Import binary settings from input file                   */

  // Declare that options are to be fetched from the [neutrals] section in the input file
  Options *neutraloptions = Options::getRoot()->getSection("neutrals");

  // neutralDynamics : if false, no neutral transport
  neutraloptions->get("neutralDynamics",neutralDynamics,0);
  dump.add(neutralDynamics, "neutralDynamics", 0);

  // solve_for_helium : if false, passive helium species is not solved for
  neutraloptions->get("solve_for_helium",solve_for_helium,0);
  dump.add(solve_for_helium, "solve_for_helium", 0);

  // atomicIonization : if false, no plasma sources from ionization of atoms
  neutraloptions->get("atomicIonization",atomicIonization,0);
  dump.add(atomicIonization, "atomicIonization", 0);

  // molecularIonization : if false, no plasma sources from ionization of molecules
  neutraloptions->get("molecularIonization",molecularIonization,0);
  dump.add(molecularIonization, "molecularIonization", 0);

  // chargeExchange : if false, no plasma sources from charge exchange collisions
  neutraloptions->get("chargeExchange",chargeExchange,0);
  dump.add(chargeExchange, "chargeExchange", 0);

  // dissociation : if false, no plasma sources from dissociation of molecules
  neutraloptions->get("dissociation",dissociation,0);
  dump.add(dissociation, "dissociation", 0);

  // elasticCollisions : if false, no plasma sources from elastic collisions with neutrals
  neutraloptions->get("elasticCollisions",elasticCollisions,0);
  dump.add(elasticCollisions, "elasticCollisions", 0);

  // inputBCs : if true, use boundary conditions given in input file (bndry_xin, bndry_xout)
  neutraloptions->get("inputBCs",inputBCs,0);
  dump.add(inputBCs, "inputBCs", 0);

  // subtractBackground : Subtract normalized background of 1 from n in neutral module. Useful for seeded blobs
  neutraloptions->get("subtractBackground",subtractBackground,0);
  dump.add(subtractBackground, "subtractBackground", 0);

  // local_diffusion : if false, diffusion coefficients are constant
  neutraloptions->get("local_diffusion",local_diffusion,0);
  dump.add(local_diffusion, "local_diffusion", 0);

  // multiplier on neutral equations for initialization
  neutraloptions->get("neutral_transport_multiplier",neutral_transport_multiplier,1);
  dump.add(neutral_transport_multiplier, "neutral_transport_multiplier", 1);

  // floor neutral density profiles
  neutraloptions->get("floor_neutral_profiles",floor_neutral_profiles,0);
  dump.add(floor_neutral_profiles, "floor_neutral_profiles", 0);

  neutraloptions->get("floor_neutral_density",floor_neutral_density,1e-9);
  dump.add(floor_neutral_density, "floor_neutral_density", 1e-9);

  neutraloptions->get("neutral_floor_time",neutral_floor_time,50);
  dump.add(neutral_floor_time, "neutral_floor_time", 50);

/*                 Import parameter settings from input file                  */

  // nColdFlux : Flux of cold neutrals on outer boundary in [m^-2*s^-1]
  neutraloptions->get("nColdFlux",nColdFlux,0);
  dump.add(nColdFlux, "nColdFlux", 0);

  // wallAbsorbtionFraction : Fraction of incoming neutrals absorbed by the wall
  neutraloptions->get("wallAbsorbtionFraction",gamma,0.);
  dump.add(gamma, "wallAbsorbtionFraction", 0);

  // uNx,uNy,uNz : neutral advection velocity in [m*s^-1]
  neutraloptions->get("uNx",uNx,0.);
  neutraloptions->get("uNy",uNy,0.);
  neutraloptions->get("uNz",uNz,0.);
  dump.add(uNx, "uNx", 0);
  dump.add(uNy, "uNy", 0);
  dump.add(uNz, "uNz", 0);
  uN.x  = uNx;	// Neutral fluid velocity, x-component
  uN.y  = uNy;	//                         y-component
  uN.z  = uNz;	//                         z-component

  // Tcold, Twarm, Thot : Cold, warm, hot neutral temperatures in normalized units
  neutraloptions->get("Tcold", Tcold, 0.025/hesel_para_.Te());
  neutraloptions->get("Twarm", Twarm, 2.   /hesel_para_.Te());
  neutraloptions->get("Thot",  Thot,  20.  /hesel_para_.Te());
  T_cold = Tcold; // Cold neutral temperature
  T_warm = Twarm; // Warm neutral temperature
  T_hot  = Thot;  // Hot  neutral temperature

  // Cold, warm, hot neutral diffusion coefficients
  neutraloptions->get("Dcold", Dcold, 0.1 );
  neutraloptions->get("Dwarm", Dwarm, 10. );
  neutraloptions->get("Dhot" , Dhot,  100.);
  D_cold = Dcold; // Cold neutral diffusion coefficient
  D_warm = Dwarm; // Warm neutral diffusion coefficient
  D_hot  = Dhot;  // Hot neutral diffusion coefficient

  // Normalize input parameters
  nColdFlux /= hesel_para_.n()*hesel_para_.cs(); // Normalize cold neutral flux
  uN        /= hesel_para_.cs();		 // Normalize neutral fluid velocity

  D_cold    /= hesel_para_.rhos()*hesel_para_.cs();  // Normalize cold neutral diffusion coefficient
  D_warm    /= hesel_para_.rhos()*hesel_para_.cs();  // Normalize warm neutral diffusion coefficient
  D_hot     /= hesel_para_.rhos()*hesel_para_.cs();  // Normalize hot neutral diffusion coefficient

/*                      Define fixed parameter settings                       */

  // !!! These should be moved to HeselParameters !!!

  // Dissociation, molecule and atom ionization potentials
  phi_Dis = 4.52/hesel_para_.Te();  // Molecular dissociation potential                [eV] normalized
  phi_Iz  = 18.11/hesel_para_.Te(); // Molecular ionization and dissociation potential [eV] normalized
  phi_iz  = 13.6/hesel_para_.Te();  // Atomic ionization potential                     [eV] normalized

  // Electron-ion(neutral) mass ratio
  mu      = Me/hesel_para_.mi();

/*                            Initialize variables                            */

  // Initialize source terms
  Sn      = 0;	// Electron density source
  Spi     = 0;	// Ion pressure source
  Spe     = 0;	// Electron pressure source
  uSi     = 0;	// Ion momentum source drift contribution

  SN_cold = 0;	// Cold neutral density source
  SN_warm = 0;	// Warm neutral density source
  SN_hot  = 0;	// Hot  neutral density source

  // Initialize leading order drift velicities
  u_perp0.x = 0; // Leading order electron drift, x-component
  u_perp0.y = 0; //                               y-component
  u_perp0.z = 0; //                               z-component
  u_perp0.x = 0; // Leading order ion drift,      x-component
  u_perp0.y = 0; //                               y-component
  u_perp0.z = 0; //                               z-component

  u_perp0.setBoundary("dirichlet_o2(0.)"); // Set boundary conditions on u_perp0
  u_perp0.setBoundary("dirichlet_o2(0.)"); // Set boundary conditions on u_perp0

  // Apply neutral initial fields
  initial_profile("lnNcold",  lnNcold); // Initial cold neutral log-density
  initial_profile("lnNwarm",  lnNwarm); // Initial warm neutral log-density
  initial_profile("lnNhot",   lnNhot);  // Initial hot  neutral log-density

  if(solve_for_helium){
    initial_profile("lnNhelium",   lnNhelium);
    Nhelium = exp(lnNhelium);
    SAVE_REPEAT2(Nhelium, SN_helium);
  }
  // Initialize neutral density fields from log-fields
  Ncold = exp(lnNcold);	// Cold neutral density
  Nwarm = exp(lnNwarm);	// Warm neutral density
  Nhot  = exp(lnNhot);	// Hot  neutral density

/*                Declare fields to be included in output file                */

  SAVE_REPEAT3(Ncold,Nwarm,Nhot);  // Neutral density fields
  SAVE_REPEAT(u_perp0);            // Leading order drift velocity
  SAVE_REPEAT4(Sn,Spe,Spi,uSi);    // Plasma source terms from neutral interactions

  return 0;
}

/******************************************************************************/
/*                               MODULE RUNNER                                */
/******************************************************************************/

int Neutrals::RhsNeutrals(BoutReal UNUSED(t)){

/*      Switch for subtracting 99% of a static density background of 1        */

  if(subtractBackground){
    N =  n - .99; // Subtract 99% of background (100% is unstable)
  }
  else{
    N = n;        // If background is not subtracted, the full plasma density is assigned
  }

/*                         Update plasma temperatures                         */

  te = pe / n;    // Electron temperature
  ti = Pi / n;    // Ion temperature

/*                  Reset source terms for every iteration                    */

  Sn      = 0;	// Electron density source
  Spi     = 0;	// Ion pressure source
  Spe     = 0;	// Electron pressure source
  uSi     = 0;	// Ion source drift velocity

  SN_cold = 0;	// Cold neutral density source
  SN_warm = 0;	// Warm neutral density source
  SN_hot  = 0;	// Hot neutral density source

/*               Calculate leading order plasma drift velocity                */

  u_perp0.x  = -1/B*DDZ(phi);  // ExB drift, x-component
  u_perp0.z  =  1/B*DDX(phi);  //            z-component

  u_perp0.applyBoundary();     // Apply boundary condituon to u_perp0

/*                    Calculate auxillary velocity fields                     */

  uDiff       = uN - u_perp0;  // Difference between neutral leading order drift velocity
  uDiff_x_B.x =  uDiff.z;      // Cross product of uDiff and B, x-component
  uDiff_x_B.y =  0;            //                               y-component
  uDiff_x_B.z = -uDiff.x;      //                               z-component

/*              Calculate source terms from neutral interactions              */

  // Molecular dissociation : H2 + e -> 2H + e
  k_Dis = ReactionRateDissociationMolecular(hesel_para_.Te()*te); // Dissociation reaction rate coefficitent
  if(dissociation){
    Spe   -= phi_Dis*Ncold*N*k_Dis; // Subtract electron pressure sink from dissociation potential
  }
  SN_cold -= Ncold*N*k_Dis;         // Subtract molecule density sink
  SN_warm += 2*Ncold*N*k_Dis;       // Add warm atom density source

  // Molecular assisted ionization : H2 + e -> H + H+ + 2e
  k_Iz  = ReactionRateIonizationMolecular(hesel_para_.Te()*te); // Molecular assisted ionization reaction rate coefficitent
  Sn_cold = Ncold*N*k_Iz;                   // Electron density source placeholder
  if(molecularIonization){
    Sn    += Sn_cold; 	                    // Add electron density source
    uSi   += Sn_cold*uDiff_x_B/N;           // Add source drift contribution from molecular assisted ionization
    Spe   += 3./2.*mu*T_warm*Sn_cold        // Add electron thermal pressure source
           + 1./2.*mu*uDiff*uDiff*Sn_cold   // Add electron kinetic pressure source
           - phi_Iz*Sn_cold;                // Subtract electron pressure sink from ionization and dissociation potential
    Spi   += 3./2.*T_warm*Sn_cold           // Add ion thermal pressure source
           + 1./2.*uDiff*uDiff*Sn_cold;     // Add ion kinetic pressure source
  }
  SN_cold -= Ncold*N*k_Iz;                  // Subtract olecule density sink
  SN_warm += Ncold*N*k_Iz;                  // Add warm atom density source

  // Atom ionization : H + e -> H+ + 2e
  k_iz  = ReactionRateIonizationAtomic(hesel_para_.Te()*te); // Atom ionization reaction rate coefficitent
  Sn_warm = Nwarm*N*k_iz;                               // Electron (warm) density source placeholder
  Sn_hot  = Nhot*N*k_iz;                                // Electron (hot)  density source placeholder
  if(atomicIonization){
    Sn    += Sn_warm + Sn_hot;                          // Add electron density source
    uSi   += (Sn_warm + Sn_hot)*uDiff_x_B/N;            // Add source drift contribution from atom ionization
    Spe   += 3./2.*mu*(T_warm*Sn_warm + T_hot*Sn_hot)   // Add electron thermal pressure source
           + 1./2.*mu*uDiff*uDiff*(Sn_warm + Sn_hot)    // Add electron kinetic pressure source
           - phi_iz*(Sn_warm + Sn_hot);                 // Subtract electron pressure sink from ionization potential
    Spi   += 3./2.*(T_warm*Sn_warm + T_hot*Sn_hot)      // Add ion thermal pressure source
           + 1./2.*uDiff*uDiff*(Sn_warm + Sn_hot);      // Add ion kinetic pressure source
  }
  SN_warm -= Sn_warm;                                   // Subtract warm atom density sink
  SN_hot  -= Sn_hot;                                    // Subtract hot  atom density sink

  // Warm atom - ion charge-exchange : H + H+ -> H+ + H
  k_cx  = ReactionRateCXAtomic(hesel_para_.Te()*ti,hesel_para_.mi());  // Charge exchange reaction rate coefficitent
  S_cx  = N*Nwarm*k_cx;             // Charge exchange reaction rate

  v_in.x  = abs(uDiff.x);           // Absolute plasma drift - neutral velocity difference, x-component
  v_in.y  = abs(uDiff.y);           //                                                      y-component
  v_in.z  = abs(uDiff.z);           //                                                      z-component
  v_Ti  = sqrt(2.*ti);              // Ion thermal velocity
  v_Tn  = sqrt(2.*T_warm);          // Warm neutral thermal velocity
  v_cx  = sqrt(4./PI*v_Ti*v_Ti + 4./PI*v_Tn*v_Tn + v_in*v_in);                                           // Charge exchange velocity
  sigma_cx= 1/(hesel_para_.rhos()*hesel_para_.rhos())*(1.09e-18 - 7.15e-20*log(hesel_para_.cs()*v_cx));  // Charge exchange cross section

  Rcx_in  = -sigma_cx*N*Nwarm*v_in*(v_Tn*v_Tn)*pow(16./PI*v_Ti*v_Ti
      + 4.*v_in*v_in + 9.*PI/4.*v_Tn*v_Tn,-0.5);  //  Ion-neutral frictious force
  Rcx_ni  = sigma_cx*N*Nwarm*v_in*(v_Ti*v_Ti)*pow(16./PI*v_Tn*v_Tn
      + 4.*v_in*v_in + 9.*PI/4.*v_Ti*v_Ti,-0.5);  //  Neutral-ion frictious force

  Qcx_in  = sigma_cx*N*Nwarm*3./4.*(v_Tn*v_Tn)*pow(4./PI*v_Ti*v_Ti
      + 64./(9.*PI)*v_Tn*v_Tn + v_in*v_in,0.5);   //  Ion-neutral heating term
  Qcx_ni  = sigma_cx*N*Nwarm*3./4.*(v_Ti*v_Ti)*pow(4./PI*v_Tn*v_Tn
      + 64./(9.*PI)*v_Ti*v_Ti + v_in*v_in,0.5);   //  Neutral-ion heating term

  Rcx_x_B.x = -(Rcx_in.z - Rcx_ni.z);             //  Cross product of cx friction with B, x-component
  Rcx_x_B.y = 0;                                  //                                       y-component
  Rcx_x_B.z = (Rcx_in.x - Rcx_ni.x);              //                                       z-component

  if(chargeExchange){
    Spi   += 1./2.*uDiff*uDiff*S_cx               // Add ion kinetic pressure source
           + uDiff*Rcx_in + Qcx_in - Qcx_ni;      // Add ion thermal pressure source
    uSi   += (S_cx*uDiff_x_B + Rcx_x_B)/N;        // Add source drift contribution from charge exchange
  }
  SN_warm -= S_cx;                                // Subtract warm atom density sink
  SN_hot  += S_cx;                                // Add hot atom density source

  // Elastic collisions : H + H+/e -> H + H+/e
  N_neutral            = Ncold+Nwarm+Nhot;		                      // Total neutral density
  a0                   = 5.3e-11/hesel_para_.rhos();                          // Bohr radius
  neutral_crossSection = PI*pow(a0,2);  		                      // Hard sphere cross section of neutral atom
  tau_en               = 1/(neutral_crossSection*N_neutral)*pow(mu/te,0.5);   // Electron-neutral collision time
  tau_in               = 1/(4*neutral_crossSection*N_neutral)*pow(1/ti,0.5);  // Ion-neutral collion time
  alpha_en	       = 4/3*neutral_crossSection*pow(8/PI*te/mu,0.5);        // Auxillary electron-neutral parameter for calculating frictious term
  alpha_in	       = 16/3*neutral_crossSection*pow(8/PI*ti,0.5);          // Auxillary      ion-neutral parameter for calculating frictious term

  R_en = N*N_neutral*mu*alpha_en*uDiff;                                       // Electron-neutral frictious force
  R_in = N*N_neutral*alpha_in*uDiff;                                          //      Ion-neutral frictious force
  Q_en = 3*mu/tau_en*(Ncold*T_cold+Nwarm*T_warm+Nhot*T_hot - N_neutral*te);   // Electron-neutral heating term
  Q_in = 3/tau_in*(Ncold*T_cold+Nwarm*T_warm+Nhot*T_hot - N_neutral*ti);      //      Ion-neutral heating term

  R_in_x_B.x = -R_in.z;            //  Cross product of elastic friction with B, x-component
  R_in_x_B.y = 0;                  //                                            y-component
  R_in_x_B.z = R_in.x;             //                                            z-component

  if(elasticCollisions){
    Spe   += Q_en - u_perp0*R_en;  // Add electron pressure heating term
    Spi   += Q_in - u_perp0*R_in;  // Add      ion pressure heating term
    uSi   += R_in_x_B/N;           // Add source drift contribution from elastic collisions
  }

/*                       End of source term calculation                       */
/*                   Calculate and impose neutral dynamics                    */

  if(neutralDynamics){
/* If true, the neutral density fields evolve as prescribed by the transport
 * equations, and sources/sinks from plasma interactions. If false, the neutral
 * density fields are static.
 */

/*   Swich for local diffusion. If false, diffusion coefficient is constant   */

    if(local_diffusion){
      n_total = N + N_neutral;     // Total density of ions and neutrals
      D_cold  = 0.5*pow(T_cold/4,0.5)/(neutral_crossSection*n_total);  //  Cold neutral diffusion coefficient
      D_warm  = 0.5*pow(T_warm/2,0.5)/(neutral_crossSection*n_total);  //  Warm neutral diffusion coefficient
      D_hot   = 0.5*pow(T_hot/2,0.5)/(neutral_crossSection*n_total);   //   Hot neutral diffusion coefficient
    }

/*           Obtain neutral density fields from evolved log fields            */

    Ncold = exp(lnNcold); // Cold neutral density
    Nwarm = exp(lnNwarm); // Warm neutral density
    Nhot  = exp(lnNhot);  // Hot neutral density

/*  Swich for calculating neutral BCs. If true, BCs are given in input file   */

    if(inputBCs == 0){
/* If false, the neutral boundary conditions are defined below. If true, the
 * boundary conditions are presribed in the input file.
 */
      if (mesh->firstX()){
        // Inner boundary conditions
        BoutReal firstLowerXGhost = mesh->xstart-1;
        for(int yInd = mesh->ystart; yInd <= mesh->yend; yInd++){
          for(int zInd = 0; zInd < mesh->LocalNz -1; zInd ++){
            Ncold(firstLowerXGhost, yInd, zInd) = (4*Ncold(firstLowerXGhost+1, yInd, zInd) - Ncold(firstLowerXGhost+2, yInd, zInd))/
		(3 + 2*mesh->getCoordinates()->dx(firstLowerXGhost, yInd, zInd)*sqrt((k_Dis(firstLowerXGhost, yInd, zInd) + k_Iz(firstLowerXGhost, yInd, zInd))*N(firstLowerXGhost, yInd, zInd)/D_cold(firstLowerXGhost, yInd, zInd)));

            Nwarm(firstLowerXGhost, yInd, zInd) = (4*Nwarm(firstLowerXGhost+1, yInd, zInd) - Nwarm(firstLowerXGhost+2, yInd, zInd))/
		(3 + 2*mesh->getCoordinates()->dx(firstLowerXGhost, yInd, zInd)*sqrt((k_cx(firstLowerXGhost, yInd, zInd) + k_iz(firstLowerXGhost, yInd, zInd))*N(firstLowerXGhost, yInd, zInd)/D_warm(firstLowerXGhost, yInd, zInd)));

            Nhot(firstLowerXGhost, yInd, zInd) = (4*Nhot(firstLowerXGhost+1, yInd, zInd) - Nhot(firstLowerXGhost+2, yInd, zInd))/
		(3 + 2*mesh->getCoordinates()->dx(firstLowerXGhost, yInd, zInd)*sqrt(k_iz(firstLowerXGhost, yInd, zInd)*N(firstLowerXGhost, yInd, zInd)/D_hot(firstLowerXGhost, yInd, zInd)));
          }
        }
      }
      else if(mesh->lastX()){
        // Outer boundary conditions
        BoutReal firstUpperXGhost = mesh->xend+1;
        for(int yInd = mesh->ystart; yInd <= mesh->yend; yInd++){
          for(int zInd = 0; zInd < mesh->LocalNz -1; zInd ++){
            Ncold(firstUpperXGhost, yInd, zInd) = (4*Ncold(firstUpperXGhost-1, yInd, zInd) - Ncold(firstUpperXGhost-2, yInd, zInd))/
		(3 + 2*mesh->getCoordinates()->dx(firstUpperXGhost, yInd, zInd)*gamma*sqrt((k_Dis(firstUpperXGhost, yInd, zInd) + k_Iz(firstUpperXGhost, yInd, zInd))*N(firstUpperXGhost, yInd, zInd)/D_cold(firstUpperXGhost, yInd, zInd)))
		+ mesh->getCoordinates()->dx(firstUpperXGhost, yInd, zInd)
                /D_cold(firstUpperXGhost, yInd, zInd)*nColdFlux;

            Nwarm(firstUpperXGhost, yInd, zInd) = (4*Nwarm(firstUpperXGhost-1, yInd, zInd) - Nwarm(firstUpperXGhost-2, yInd, zInd))/
		(3 + 2*mesh->getCoordinates()->dx(firstUpperXGhost, yInd, zInd)*gamma*sqrt((k_cx(firstUpperXGhost, yInd, zInd) + k_iz(firstUpperXGhost, yInd, zInd))*N(firstUpperXGhost, yInd, zInd)/D_warm(firstUpperXGhost, yInd, zInd)));

            Nhot(firstUpperXGhost, yInd, zInd) = (4*Nhot(firstUpperXGhost-1, yInd, zInd) - Nhot(firstUpperXGhost-2, yInd, zInd))/
		(3 + 2*mesh->getCoordinates()->dx(firstUpperXGhost, yInd, zInd)*gamma*sqrt(k_iz(firstUpperXGhost, yInd, zInd)*N(firstUpperXGhost, yInd, zInd)/D_hot(firstUpperXGhost, yInd, zInd)));
          }
        }
      }
    }

/*                                  __
 *                         /\    .-" /
 *                        /  ; .'  .'
 *                       :   :/  .'
 *                        \  ;-.'
 *           .--""""--..__/     `.
 *         .'           .'    `o  \
 *        /                    `   ;
 *       :                  \      :
 *     .-;        -.         `.__.-'
 *    :  ;          \     ,   ;
 *    '._:           ;   :   (
 *        \/  .__    ;    \   `-.
 *         ;     "-,/_..--"`-..__)
 *         '""--.._:
 * The following line is a contribution made by Ninja the rabbit, as she
 * walked across the keyboard:
 * m,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,cx,,,,,,,,,,,,,,dddx5<<<<
 * <<<<<<<<<<<<<<<<<<<<<<<<<
 */

/*                    Communicate fields across processors                    */

    mesh->communicate(lnNcold,lnNwarm,lnNhot);  // Communicate log-densities
    mesh->communicate(Ncold,Nwarm,Nhot);        // Communicate densities
    mesh->communicate(D_cold,D_warm,D_hot);     // Communicate diffusion coefficients

/*          Update neutral fields according to transport and sources          */

    update_cold = neutralDynamics & 1;  // Binary switch for updating cold neutrals
    update_warm = neutralDynamics & 2;  // Binary switch for updating warm neutrals
    update_hot  = neutralDynamics & 4;  // Binary switch for updating hot neutrals

    if(update_cold){
	  ddt_lnNcold = (Grad(D_cold)*Grad(Ncold) + D_cold*Laplace(Ncold) - uN*Grad(Ncold))/Ncold;  // Drive cold neutral density
      ddt_lnNcold += SN_cold/Ncold;  // Add cold neutral density source

	  if(floor_neutral_profiles){
		for (auto i:lnNcold.getRegion(RGN_NOBNDRY)){
          if(Ncold[i]  < floor_neutral_density){
            ddt_lnNcold[i]  += (floor_neutral_density /Ncold[i]  - 1)/neutral_floor_time;
		  }
	    }
      }
	  ddt(lnNcold) = ddt_lnNcold;
	}
    else{
      ddt(lnNcold)  = 0;              // Static cold neutral density
    }
    if(update_warm){
      ddt(lnNwarm)  = (Grad(D_warm)*Grad(Nwarm) + D_warm*Laplace(Nwarm) - uN*Grad(Nwarm))/Nwarm;  // Drive warm neutral density
      ddt(lnNwarm) += SN_warm/Nwarm;  // Add warm neutral density source
    }
    else{
      ddt(lnNwarm)  = 0;              // Static warm neutral density
    }
    if(update_hot){
      ddt(lnNhot)   = (Grad(D_hot) *Grad(Nhot)  + D_hot *Laplace(Nhot)  - uN*Grad(Nhot) )/Nhot;   // Drive hot  neutral density
      ddt(lnNhot)  += SN_hot/Nhot;    // Add hot  neutral density source
    }
    else{
      ddt(lnNhot)   = 0;              // Static hot  neutral density
    }

    if(solve_for_helium){
      Nhelium = exp(lnNhelium);
      SN_helium = -Nhelium*N*k_IzHe(hesel_para_.Te()*te);

      mesh->communicate(Nhelium);

      ddt_lnNhelium = (Grad(D_cold)*Grad(Nhelium) + D_cold*Laplace(Nhelium) - uN*Grad(Nhelium))/Nhelium;  // Drive cold neutral density
      ddt_lnNhelium += SN_helium/Nhelium;  // Add cold neutral density source

	  if(floor_neutral_profiles){
		for (auto i:lnNhelium.getRegion(RGN_NOBNDRY)){
          if(Nhelium[i]  < floor_neutral_density){
            ddt_lnNhelium[i]  += (floor_neutral_density /Nhelium[i]  - 1)/neutral_floor_time;
		  }
	    }
      }
	  ddt(lnNhelium) = ddt_lnNhelium;
      ddt(lnNhelium) *= neutral_transport_multiplier;
    }

    ddt(lnNcold) *= neutral_transport_multiplier;
    ddt(lnNwarm) *= neutral_transport_multiplier;
    ddt(lnNhot)  *= neutral_transport_multiplier;

/* False option for neutralDynamics means no change in neutral density fields */
  }
else{
    ddt(lnNcold)  = 0;  // Static cold neutral density
    ddt(lnNwarm)  = 0;  // Static warm neutral density
    ddt(lnNhot)   = 0;  // Static hot  neutral density
  }

  return 0;
}

/******************************************************************************/
/*                              CALL FUNCTIONS                                */
/******************************************************************************/

/* Reaction rate coefficients : Input electron or ion temperature in eV (and
 * mass in kg), output reaction rate coefficient in m^3*s^-1*n0/oci.
 */

// Molecular dissociation (ratko 2.2.5)
Field3D Neutrals::ReactionRateDissociationMolecular(const Field3D &Te){
  return hesel_para_.n()/hesel_para_.oci()*1e-6*exp(-2.858072836568e+01*(pow(log(Te),0))+1.038543976082e+01*(pow(log(Te),1))
      -5.383825026583e+00*(pow(log(Te),2))+1.950636494405e+00*(pow(log(Te),3))
      -5.393666392407e-01*(pow(log(Te),4))+1.006916814453e-01*(pow(log(Te),5))
      -1.160758573972e-02*(pow(log(Te),6))+7.411623859122e-04*(pow(log(Te),7))
      -2.001369618807e-05*(pow(log(Te),8)));
}

// Molecular ionization succeded by dissociation (ratko 2.2.9)
Field3D Neutrals::ReactionRateIonizationMolecular(const Field3D &Te){
  return hesel_para_.n()/hesel_para_.oci()*1e-6*exp(-3.568640293666e+01*(pow(log(Te),0))+1.733468989961e+01*(pow(log(Te),1))
      -7.767469363538e+00*(pow(log(Te),2))+2.211579405415e+00*(pow(log(Te),3))
      -4.169840174384e-01*(pow(log(Te),4))+5.088289820867e-02*(pow(log(Te),5))
      -3.832737518325e-03*(pow(log(Te),6))+1.612863120371e-04*(pow(log(Te),7))
      -2.893391904431e-06*(pow(log(Te),8)));
}

// Atomic ionization (ratko 2.1.5)
Field3D Neutrals::ReactionRateIonizationAtomic(const Field3D &Te){
  return hesel_para_.n()/hesel_para_.oci()*1e-6*exp(-3.271396786375e+01*(pow(log(Te),0))+1.353655609057e+01*(pow(log(Te),1))
      -5.739328757388e+00*(pow(log(Te),2))+1.563154982022e+00*(pow(log(Te),3))
      -2.877056004391e-01*(pow(log(Te),4))+3.482559773737e-02*(pow(log(Te),5))
      -2.631976175590e-03*(pow(log(Te),6))+1.119543953861e-04*(pow(log(Te),7))
      -2.039149852002e-06*(pow(log(Te),8)));
}

// Ionization of Helium (ratko 2.3.9)
Field3D Neutrals::k_IzHe(const Field3D &Te){
  return hesel_para_.n()/hesel_para_.oci()*1e-6*exp(-4.409864886561e+01*(pow(log(Te),0))+2.391596563469e+01*(pow(log(Te),1))
      -1.075323019821e+01*(pow(log(Te),2))+3.058038757198e+00*(pow(log(Te),3))
      -5.685118909884e-01*(pow(log(Te),4))+6.795391233790e-02*(pow(log(Te),5))
      -5.009056101857e-03*(pow(log(Te),6))+2.067236157507e-04*(pow(log(Te),7))
      -3.649161410833e-06*(pow(log(Te),8)));
}

// Atomic charge-exchange (ratko 3.1.8)
Field3D Neutrals::ReactionRateCXAtomic(const Field3D &Ti, BoutReal mi){
  return hesel_para_.n()/hesel_para_.oci()*3.2e-28*(pow(3./2.*Ti,-0.2))*sqrt(3.*Ti/mi);
}
/*============================================================================*/
