/*
 * Parallel.cxx
 *
 *  Created on: Oct 1, 2018
 *      Author: alec
 */

#include "Parallel.hxx"

int Parallel::InitParallel(){
  /* --- Import switches from input file --- */
  Options *paralleloptions = Options::getRoot()->getSection("hesel");

  /* parallelDynamics : if false, no parallel transport. */
  paralleloptions->get("parallel_transport",parallel_transport,0);
  dump.add(parallel_transport, "parallel_transport", 0);

  /* parallelDynamicsType : bitwise switch for parallel transport terms, (diffusion) <-> (1) */
  paralleloptions->get("parallel_transport_type",parallel_transport_type,0);
  dump.add(parallel_transport_type, "parallel_transport_type", 0);

  /* Constants */
  // Ion-electron mass ratio
  mu      = hesel_para_.mi()/Me;

  // Electron-ion collision frequency (normalized)
  nu_parl 	= 0.51*hesel_para_.nuei()/hesel_para_.oci();
  nu_e_parl 	= 2.92*(hesel_para_.rhoe()*hesel_para_.rhoe())*hesel_para_.oce()/(3*(hesel_para_.rhos()*hesel_para_.rhos())*hesel_para_.nuei());

  /* Initialize fields. */
  j_parl = n*(ui_parl - ue_parl);

  n_terms	= 0.;
  vort_terms 	= 0.;
  pe_terms	= 0.;
  pi_terms	= 0.;

  /* Declare other fields to be saved. */
  SAVE_REPEAT(j_parl);

  return 0;
}

int Parallel::RhsParallel(BoutReal UNUSED(t)){
/* Update fields */
  te = pe / n;    // Electron temperature
  ti = Pi / n;    // Ion temperature

  j_parl = n*(ui_parl - ue_parl);

/* Evolve parallel fields */
  ddt(ue_parl)  = 0.;
  ddt(ui_parl) 	= 0.;

/* Calculate sources to perpendicular equations */
  n_terms	= 0.;
  vort_terms 	= 0.;
  pe_terms	= 0.;
  pi_terms	= 0.;

/* Parallel transport equations */
  diffusive_transport   = parallel_transport_type & 1;
  jmbols_transport      = parallel_transport_type & 2;
  if(diffusive_transport){
    par_diff_coeff = 0.1;

    n_terms	+= par_diff_coeff*Laplace_par(n);

    pe_terms	= te*n_terms;
    pi_terms	= ti*n_terms;  
  }
  else if(jmbols_transport){
    n_terms	-= Grad_par(n*ue_parl);

    vort_terms  -= ui_parl*Grad_par(vort);
    vort_terms  += Grad_par(j_parl)/n;

    pe_terms	= te*n_terms;
    pi_terms	= ti*n_terms;

    ddt(ue_parl) -= ue_parl*Grad_par(ue_parl);
    ddt(ue_parl) += mu*Grad_par(phi);
    ddt(ue_parl) -= mu*Grad_par(n)/n;
    ddt(ue_parl) += nu_parl*(ui_parl - ue_parl);
    // SOURCE TERM TO BE INCLUDED
    ddt(ue_parl) += mu/nu_e_parl*Laplace_par(ue_parl);

    ddt(ui_parl) -= ui_parl*Grad_par(ui_parl);
    ddt(ui_parl) -= Grad_par(phi);
    ddt(ui_parl) -= nu_parl/mu*(ui_parl - ue_parl);
    // SOURCE TERM TO BE INCLUDED    
  }

  

  return 0;
}

Parallel::Parallel(
    Solver *i_solver,
    const HeselParameters &i_hesel_para_,
    const Field3D &i_n,
    const Field3D &i_phi,
    const Field3D &i_vort,
    const Field3D &i_Pi,
    const Field3D &i_pe,
    const Field2D &i_B) :
      ///The class member below are references and hence must be initialized at object instantiation
      solver(i_solver),
      hesel_para_(i_hesel_para_),
      n(i_n),
      phi(i_phi),
      vort(i_vort),
      Pi(i_Pi),
      pe(i_pe),
      B(i_B){
}
