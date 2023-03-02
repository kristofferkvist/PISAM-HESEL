/*
 * Parallel.hxx
 *
 *  Created on: Oct 1, 2018
 *      Author: alec
 */

#ifndef PARALLEL_HXX_
#define PARALLEL_HXX_

//#include <boutmain.hxx>
//#include <bout/physicsmodel.hxx>
#include <bout.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>
#include <bout/solver.hxx>
#include <math.h>
#include <initialprofiles.hxx>
#include "../HeselParameters/HeselParameters.hxx" //Class calculating collisional parameters, parametrization of parallel dynamics etc.

class Parallel {


public:
  ///Constructor which forces instantiation to assign references to vital (plasma) fields
  Parallel(
      Solver *i_solver,
      const HeselParameters &i_hesel_para_,
      const Field3D &i_n,
      const Field3D &i_phi,
      const Field3D &i_vort,
      const Field3D &i_Pi,
      const Field3D &i_pe,
      const Field2D &i_B
  );

  ///
  Solver *solver;
  const HeselParameters &hesel_para_;         //object holding HESEL collisional parameters
  const Field3D &n; //plasma density
  const Field3D &phi; //electric potential
  const Field3D &vort; //vorticity
  const Field3D &Pi; //ion temperature
  const Field3D &pe;  //electron pressure
  const Field2D &B;  //Magnetic field

  BoutReal mu;          // Ion-electron mass ratio
  BoutReal nu_parl; 	// Parallel collision frequency
  BoutReal nu_e_parl;
  Field3D te, ti; ///electron, ion temperatures
  Field3D j_parl, nui_parl; // parallel current and ion velocity density
  Field3D ue_parl, ui_parl; // electron, ion parallel velocities
  Field3D n_terms, vort_terms, pe_terms, pi_terms; // Parallel transport terms to add to rhs of ddt_n, ddt_vort, ddt_pe, ddt_pi equations

  ///<Methods
  int InitParallel();
  int RhsParallel(BoutReal t);

  bool parallel_transport = false;
  int parallel_transport_type;
  bool diffusive_transport = false;
  bool jmbols_transport = false;
  BoutReal par_diff_coeff;
};


#endif /* PARALLEL_HXX_ */
