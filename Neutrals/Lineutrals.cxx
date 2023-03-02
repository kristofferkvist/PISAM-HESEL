/*
 * As a first this is to be a part of the neutral working area.
 * The idea is to first implement lithium with first ionization then recombination and finally cx
 * This will be the ground state at first for all ionization states.
 * Further down the road, mutiple species should implemented with proper rates
 */
 
#include "Lieutrals.hxx"

int Lineutrals::InitLineutrals(){
  /* Definition of other parameters. [Move to init] */
  mu      = Me/hesel_para_.mi();            // Electron-ion mass ratio
  
  
  /* Definition of masses. [Move to init] */
  mnli    = 3*(0.5*hesel_para_.mi() + hesel_para_.Z()*Me);    // Neutral lithium mass. Is mi 1 or 2? also it should be more general. I.e. get multiplication factor from input.

  mu_nli  = Me/mnli;        // Electron-neutral mass ratio
    

  /* Import switches from input file. */
  Options *neutraloptions = Options::getRoot()->getSection("neutrals");

  /* neutralDynamics : if false, no neutral transport. */
  neutraloptions->get("neutralDynamics",neutralDynamics,0);
  dump.add((int&) neutralDynamics, "neutralDynamics", 0);

  /* neutralSources : if false, no plasma sources from neutral interactions.
   */
  neutraloptions->get("neutralSources",neutralSources,0);
  dump.add((int&) neutralSources, "neutralSources", 0);

  /* atomicIonization : if false, no plasma sources from ionization of atoms.
   */
  neutraloptions->get("atomicIonization",atomicIonization,0);
  dump.add((int&) atomicIonization, "atomicIonization", 0);

// No need for molecular ionization ##########TAKE NOTE FOR FUTURE REF##########
//  /* molecularIonization : if false, no plasma sources from ionization of
//   * molecules.
//   */
//  neutraloptions->get("molecularIonization",molecularIonization,0);
//  dump.add((int&) molecularIonization, "molecularIonization", 0);

  /* chargeExchange : if false, no plasma sources from charge exchange
   * collisions.
   */
  neutraloptions->get("chargeExchange",chargeExchange,0);
  dump.add((int&) chargeExchange, "chargeExchange", 0);

// No need for dissociation ##########TAKE NOTE FOR FUTURE REF##########
//  /* dissociation : if false, no plasma sources from dissociation of
//  * molecules.
//   */
//  neutraloptions->get("dissociation",dissociation,0);
//  dump.add((int&) dissociation, "dissociation", 0);

  /* inputBCs : if true, use boundary conditions given in input file
   * (bndry_xin, bndry_xout).
   */
  neutraloptions->get("inputBCs",inputBCs,0);
  dump.add((int&) inputBCs, "inputBCs", 0);

// THIS PROBABLY NEEDS TO BE TAKEN INTO ACCOUNT BUT NOT WITH THE COLD THINGY. ASK ALEX
//  /* nColdFlux : Flux of cold neutrals on outer boundary in [m^-2*s^-1]. */
//  neutraloptions->get("nColdFlux",nColdFlux,0);
//  dump.add((int&) nColdFlux, "nColdFlux", 0);

  
  /* no_n_force : if true, no forcing of plasma density profile in edge. */
  neutraloptions->get("no_n_force",no_n_force,0);
  dump.add((int&) no_n_force, "n_n_force", 0);
  
  /* uNx,uNy,uNz : neutral advection velocity in [m*s^-1]. */
  neutraloptions->get("uNx",uNlix,0.);
  neutraloptions->get("uNy",uNliy,0.);
  neutraloptions->get("uNz",uNliz,0.);
  dump.add((int&) uNlix, "uNx", 0);
  dump.add((int&) uNliy, "uNy", 0);
  dump.add((int&) uNliz, "uNz", 0);
  
  /* Assign fluid velocity from input */
  uNli.x  = uNlix;
  uNli.y  = uNliy;
  uNli.z  = uNliz;
  
  
  /* Normalization of input parameters. */
//  nColdFlux /= hesel_para_.n()*hesel_para_.cs();
  uNli        /= hesel_para_.cs();
  
  /* Other neutral relevant parameters. */
  T_li  = 0.025/hesel_para_.Te(); // Lithium neutral temperature [eV] normalized to electron background temp
  
  phi_iz  = 5.4/hesel_para_.Te(); // Ground state first ionization potential lithium. Should be part of file to be read.

  /* Initialization of source terms. */
  Sn      = 0;    // Electron density source
  Spili   = 0;    // Lithium ion energy source
  Spi     = 0;
  Spe     = 0;    // Electron energy source
  uSe     = 0;    // Electron bulk velocity source
  uSi     = 0;
  uSili   = 0;    // Lithium Ion bulk velocity source
  
  SNli    = 0;    // Lithium source no. Should typically correspond to a valve opening...
  
  /* Initialization of diffusion coefficients. */
  D_li  = 0.1;   //I don't necessarily think we will have a diffusion part. In that case, make it 


  /* Initialize leading order drift velicities. */
  ue_perp0.x   = 0;
  ue_perp0.y   = 0;
  ue_perp0.z   = 0;
  ui_perp0.x   = 0;
  ui_perp0.y   = 0;
  ui_perp0.z   = 0;
  ui_perp0.x   = 0;
  uili_perp0.y = 0;
  uili_perp0.z = 0;

  /* Set boundaries on auxilary fields. */
  uSe_0.setBoundary("neumann_o2(0.)");
  uSi_0.setBoundary("neumann_o2(0.)");
  uSili_0.setBoundary("neumann_o2(0.)");
  ue_perp0.setBoundary("neumann_o2(0.)");
  ui_perp0.setBoundary("neumann_o2(0.)");
  uili_perp0.setBoundary("neumann_o2(0.)");

  /* Initialize neutral density fields. */
  Nli = exp(lnNli);


  /* Declare other fields to be saved. */
  SAVE_REPEAT3(Nli);                            // Neutral density fields
  SAVE_REPEAT3(ue_perp0,uili_perp0,ui_perp0);            // Ldng order drift velocities
  SAVE_REPEAT4(Sn,Spe,Spili,Spi);                   // Source terms from neutrals
  /*========================================================================*/
  
    return 0;
}
  
  int Neutrals::RhsNeutrals(BoutReal t){
  te = pe / n;
  tili = Pili / n;

  /* Reset source terms for every iteration. */
  Sn      = 0;    // Electron density source
  Spi     = 0;    // Ion energy source
  Spili   = 0;    // Lithium Ion energy source
  Spe     = 0;    // Electron energy source
  uSe     = 0;    // Electron bulk velocity source
  uSi     = 0;    // Ion bulk velocity source
  uSili   = 0;    // Lithium ion bulk velocity source

  SNli    = 0;    // Source of neutral lithium

  /* Calculate leading order electron and ion drift velocities. */
  ue_perp0.x  = 1/B*(DDZ(phi)-DDZ(pe)/n);
  ue_perp0.z  = -1/B*(DDX(phi)-DDX(pe)/n);
  ui_perp0.x  = 1/B*(DDZ(phi)+DDZ(Pi)/n);
  ui_perp0.z  = -1/B*(DDX(phi)+DDX(Pi)/n);
  uili_perp0.x  = 1/B*(DDZ(phi)+DDZ(Pi)/n);
  uili_perp0.z  = -1/B*(DDX(phi)+DDX(Pi)/n);
  
  ue_perp0.applyBoundary();
  ui_perp0.applyBoundary();
  uili_perp0.applyBoundary();

  /* Calculate leading order source drift contributions. */
  uSe_0   = Grad_perp(phi)/B - Grad_perp(pe)/(n*B);
  uSi_0   = Grad_perp(phi)/B + Grad_perp(Pi)/(n*B);
  uSili_0   = Grad_perp(phi)/B + Grad_perp(Pili)/(n*B);

  uSe_0.applyBoundary();
  uSi_0.applyBoundary();
  uSili_0.applyBoundary();

  mesh->communicate(uSe,uSi,uSili);
  /* Calculation of source terms to plasma and neutral fields, originating
   * from inelastic collisions. The processes of molecular dissociation,
   * molecular ionization succeded by dissociation, atomic ionization and
   * atomic charge-exchange are included.
   */


  /* Atomic ionization. */
  k_liz  = ReactionRateIonizationLithium(hesel_para_.Te()*te);
  Sn_li   = Nli*n*k_liz
  if(atomicIonization){
    Sn      += Sn_li
    uSe   += mu*(Sn_li)/n*(uSe_0 - uNli);
    uSili   += (Sn_li)/n*(uNli - uSili_0);

    Spe   += 3./2.*mu_n*(T_warm*Sn_warm)
                                                + 1./2.*mu*(uN - ue_perp0)*(uN - ue_perp0)*(Sn_warm + Sn_hot)
                                                + 1./2.*mu*ue_perp0*ue_perp0*(Sn_warm + Sn_hot)
                                                - phi_iz*(Sn_warm + Sn_hot);

    Spili   += 3./2.*hesel_para_.mi()/mn*(T_warm*Sn_warm + T_hot*Sn_hot)
                                                + 1./2.*(uN - ui_perp0)*(uN - ui_perp0)*(Sn_warm + Sn_hot)
                                                + 1./2.*ui_perp0*ui_perp0*(Sn_warm + Sn_hot);
  }
  SNli -= Sn_li






  if(neutralDynamics){
    /* If true, the neutral density fields evolve as prescribed by the transport
     * equations, and sources/sinks from plasma interactions. If false, the
     * neutral density fields are static.
     */
    /* Communicate neutral fields for differentiation on multi-processor
     * jobs.
     */
    mesh->communicate(lnNcold,lnNwarm,lnNhot);
    mesh->communicate(Nli);
    mesh->communicate(D_li);

    /* Obtain neutral density fields from evolved log fields. */
    Nli = exp(lnNli); // Cold neutral density


    /* Neutral diffusion coefficients */
    D_li  = 0.1;  // Cold neutral diffusion coefficient


    /* Boundary conditions on neutral fields */
    BoutReal gamma = 0.2; // Fraction of wall absorbtion
    if(inputBCs == 0){
      /* If false, the neutral boundary conditions are defined below. If true,
       * the boundary conditions are presribed in the input file.
       */
      if (mesh->firstX()){
        /* Inner boundary conditions. */
        BoutReal firstLowerXGhost = mesh->xstart-1;
        for(int yInd = mesh->ystart; yInd <= mesh->yend; yInd++){
          for(int zInd = 0; zInd < mesh->LocalNz -1; zInd ++){
            Nli(firstLowerXGhost, yInd, zInd) =
                2*mesh->coordinates()->dx(firstLowerXGhost+1, yInd)
                *(-sqrt(uNli.x(firstLowerXGhost+1, yInd, zInd)
                    *uNli.x(firstLowerXGhost+1, yInd, zInd)
                    + 4*D_li(firstLowerXGhost+1, yInd, zInd)
            *k_liz(firstLowerXGhost+1, yInd, zInd)
            *n(firstLowerXGhost+1, yInd, zInd)))
            /D_li(firstLowerXGhost+1, yInd, zInd)
            *Nli(firstLowerXGhost+1, yInd, zInd)
            + Nli(firstLowerXGhost+2, yInd, zInd);
          }
        }
      }
      else if(mesh->lastX()){
        /* Outer boundary conditions. */
        BoutReal firstUpperXGhost = mesh->xend+1;
        for(int yInd = mesh->ystart; yInd <= mesh->yend; yInd++){
          for(int zInd = 0; zInd < mesh->LocalNz -1; zInd ++){
            Nli(firstUpperXGhost, yInd, zInd) =
                2*mesh->coordinates()->dx(firstUpperXGhost-1, yInd)*((1-gamma)
                    *uNli.x(firstUpperXGhost-1, yInd, zInd)
                    -gamma*sqrt(uNli.x(firstUpperXGhost-1, yInd, zInd)
                        *uNli.x(firstUpperXGhost-1, yInd, zInd)
                        + 4*D_li(firstUpperXGhost-1, yInd, zInd)
            *k_liz(firstUpperXGhost-1, yInd, zInd)
            *n(firstUpperXGhost-1, yInd, zInd)))
            /D_li(firstUpperXGhost-1, yInd, zInd)
            *Nli(firstUpperXGhost-1, yInd, zInd)
            + Nli(firstUpperXGhost-2, yInd, zInd);
          }
        }
      }
    }
    /* The following line is a contribution made by Ninja the rabbit, as she
     * walked across the keyboard:
     * m,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,cx,,,,,,,,,,,,,,dddx5
     * <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
     */

    /* Update neutral fields according to transport. */
    ddt(lnNli)  = (D_li*Delp2(Nli)
    + Grad_perp(D_cold)*Grad_perp(Nli)
    - Div(uN*Nli))/Nli;


    /* Update neutral field according to sources. */
    ddt(lnNcold)  += SN_cold/Ncold;
    ddt(lnNwarm)  += SN_warm/Nwarm;
    ddt(lnNhot)     += SN_hot/Nhot;
  }
  else{
    /* False option for neutralDynamics, i.e., no change in neutral density
     * fields.
     */
    ddt(lnNcold)  = 0;
    ddt(lnNwarm)  = 0;
    ddt(lnNhot)   = 0;
  }
  /*========================================================================*/

  return 0;
}

Neutrals::Neutrals(
    Solver *i_solver,
    const HeselParameters &i_hesel_para_,
    const Field3D &i_n,
    const Field3D &i_phi,
    const Field3D &i_Pi,
    const Field3D &i_pe,
    const Field2D &i_B) :
      ///The class member below are references and hence must be initialized at object instantiation
      solver(i_solver),
      hesel_para_(i_hesel_para_),
      n(i_n),
      phi(i_phi),
      Pi(i_Pi),
      pe(i_pe),
      B(i_B){
  ue_perp0 = 0.;
  ui_perp0 = 0.;
}

/* Reaction rate coefficients : Input electron or ion temperature in eV (and
 * mass in kg), output reaction rate coefficient in m^3*s^-1*n0/oci.
 */

/* Atomic ionization fitted ADAS-data */
Field3D Neutrals::ReactionRateIonizationAtomic(const Field3D &Te){
  return hesel_para_.n()/hesel_para_.oci()*1e-6*exp(-3.271396786375e+01*(pow(log(Te),0))+1.353655609057e+01*(pow(log(Te),1))
      -5.739328757388e+00*(pow(log(Te),2))+1.563154982022e+00*(pow(log(Te),3))
      -2.877056004391e-01*(pow(log(Te),4))+3.482559773737e-02*(pow(log(Te),5))
      -2.631976175590e-03*(pow(log(Te),6))+1.119543953861e-04*(pow(log(Te),7))
      -2.039149852002e-06*(pow(log(Te),8)));
}
/*============================================================================*/
  
  
  
  
  
  
  
  
  
  
  
  /* Calculate leading order electron and ion drift velocities. */
  ue_perp0.x  = 1/B*(DDZ(phi)-DDZ(pe)/n);
  ue_perp0.z  = -1/B*(DDX(phi)-DDX(pe)/n);
  uli_perp0.x  = 1/B*(DDZ(phi)+DDZ(Pil)/n);
  uli_perp0.z  = -1/B*(DDX(phi)+DDX(Pil)/n);
  
  ue_perp0.applyBoundary();
  uli_perp0.applyBoundary();  //Now only included lithium. Should regular ions also be included?
