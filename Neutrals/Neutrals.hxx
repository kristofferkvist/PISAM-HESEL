/******************************************************************************/
/*                                NEUTRALS.HXX                                */
/******************************************************************************/

/*
 * Neutrals.hxx
 *
 *  Created on: April 11 , 2017
 *      Author: yolen, Alec
 *
 * Class file of module for calculating neutral transport and source. The inter-
 * actions include elastic collisions, molecule dissociation, atom and molecule 
 * ionization, and charge exchange between atoms created by dissociation of 
 * molecules and ions. 
 */

#ifndef NEUTRALS_HXX_
#define NEUTRALS_HXX_

#include <bout.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>
#include <bout/solver.hxx>
#include <math.h>
#include <initialprofiles.hxx>
#include "../HeselParameters/HeselParameters.hxx"
#include "../BoutFastOutput/fast_output.hxx"

class Neutrals {
public:

/*    Constructor which forces instantiation to assign references to vital    
 *                              (plasma) fields                               */
                                                                              
  Neutrals(
      Solver *i_solver,
      const HeselParameters &i_hesel_para_,
      const Field3D &i_n,
      const Field3D &i_phi,
      const Field3D &i_Pi,
      const Field3D &i_pe,
      const Field2D &i_B
  );

  Solver *solver;
  const HeselParameters &hesel_para_; //  Object holding HESEL collisional parameters
  const Field3D &n;                   //  Plasma density
  const Field3D &phi;                 //  Electric potential
  const Field3D &Pi;                  //  Ion temperature
  const Field3D &pe;                  //  Electron pressure
  const Field2D &B;                   //  Magnetic field

	// Fast output object
	FastOutput fast_output;
	
/*                                 Constants                                  */

  BoutReal nColdFlux;                 // Cold neutral flux boundary condition
  BoutReal phi_Dis, phi_Iz, phi_iz;   // Dissociation/ionization energies
  BoutReal uNx, uNy, uNz;             // Neutral bulk velocities
  BoutReal mu;                        // Electron-ion mass ratio
  BoutReal a0, neutral_crossSection;  // Bohr radius, neutral cross section
  BoutReal gamma;                     // Wall absorbtion fraction 
  BoutReal Tcold, Twarm, Thot;        // Neutral input temperatures
  BoutReal Dcold, Dwarm, Dhot;        // Neutral input diffusion coefficients
  BoutReal floor_neutral_density;
  BoutReal neutral_floor_time;

/*                                   Fields                                   */

  Field3D te, ti;                     // Electron and ion temperatures
  Field3D lnNcold, lnNwarm, lnNhot;   // Log neutral fields
  Field3D ddt_lnNcold, ddt_lnNhelium;                // Log neutral fields
  Field3D lnNhelium, Nhelium;         // Helium density fields
  Field3D SN_helium, k_IzHe(const Field3D &Te); // Helium source fields 
  Field3D Ncold, Nwarm, Nhot;         // Neutral fields
  Field3D N, n_total;		      // Effective electron density, sum of ion and neutral densities
  Field3D N_neutral;		      // Total neutral density
  Field3D k_iz, k_cx, k_Iz, k_Dis;    // Reaction rate coefficients
  Field3D Sn, Spe, Spi;               // Source terms
  Field3D Sn_cold, Sn_warm, Sn_hot;   // Neutral source terms
  Field3D SN_cold, SN_warm, SN_hot;   // Neutral source terms
  Field3D S_cx;                       // Charge exchange rate
  Field3D Qcx_ni, Qcx_in;             // Charge exchange heat exchange terms
  Field3D v_Ti, v_Tn, v_cx, sigma_cx; // Other charge exchange terms
  Field3D Q_en, Q_in;		      // Elastic collision heating terms
  Field3D tau_en, tau_in;             // Elastic collision times
  Field3D alpha_en, alpha_in;         // Auxilary collision parameters
  Field3D T_cold, T_warm, T_hot;      // Neutral temperatures
  Field3D D_cold, D_warm, D_hot;      // Neutral diffusion coefficients

  Field3D ReactionRateDissociationMolecular(const Field3D &Te); // Molecular dissociation reaction rate
  Field3D ReactionRateIonizationMolecular(const Field3D &Te);   // Molecular ionizationreaction rate
  Field3D ReactionRateIonizationAtomic(const Field3D &Te);      // Atomic ionization reaction rate
  Field3D ReactionRateCXAtomic(const Field3D &Ti, BoutReal mi); // Atomic charge-exchange reaction rate

/*                                  Vectors                                   */

  Vector3D v_in;                      // Absolute plasma drift - neutral velocity difference
  Vector3D u_perp0;                   // Leading order drifts
  Vector3D uSe, uSi, uSi_0;           // Inelastic source drift
  Vector3D Rcx_ni, Rcx_in;            // Charge exchange friction
  Vector3D Rcx_x_B;		      // Charge exchange frictous terms cross b
  Vector3D uN;                        // Neutral fluid velocity
  Vector3D uDiff, uDiff_x_B;	      // Difference in neutral and plasma velocity, and that crossed with b unity vector
  Vector3D R_en, R_in;		      // Elastic collision frictious terms
  Vector3D R_in_x_B;	              // R_in cross b

/*                                   Flags                                    */

  bool inputBCs;
  bool plasmaDynamics = true;
  bool neutralSources;
  bool solve_for_helium;
  bool atomicIonization;
  bool molecularIonization;
  bool chargeExchange;
  bool dissociation;
  bool elasticCollisions;
  bool neutralForce;
  bool no_n_force;
  bool subtractBackground;
  bool local_diffusion;
  bool floor_neutral_profiles;
  int neutralDynamics;
  int update_cold, update_warm, update_hot;
  int neutral_transport_multiplier;

/*                                  Methods                                   */

  int InitNeutrals();
  int RhsNeutrals(BoutReal t);

};

#endif /* NEUTRALS_HXX_ */
