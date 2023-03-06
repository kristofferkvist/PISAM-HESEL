/*
 * hesel.hxx
 *
 *  Created on: Sep 29, 2015
 *      Author: yolen
 */

#ifndef INCLUDE_HESEL_HXX_
#define INCLUDE_HESEL_HXX_

#include <bout/physicsmodel.hxx>
#include <invert_laplace.hxx>
#include <initialprofiles.hxx>
#include <derivs.hxx>
#include <bout/constants.hxx>
#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <chrono>
#include "HeselParameters/HeselParameters.hxx" //Class calculating collisional parameters, parametrization of parallel dynamics etc.
#include "Neutrals/Neutrals.hxx" //Class calculating neutral dynamics
#include "Parallel/Parallel.hxx" //Class calculating parallel dynamics
#include "BoutFastOutput/fast_output.hxx"
#include "KineticNeutrals/KineticNeutrals.hxx"

//enum DIRECTION {down = -1, up = 1}; // magnetic field direction, currently not been used
enum COLLISIONAL_MODEL {off = 0, simple = 1, standard = 2, full = 3}; // off; standard: standard HESEL model; full: full collisional drift ordered model (see J. Madsen et al pop 2016)

//#define SAVE_BOOL_ONCE(var) dump.add((int &) var, #var, 0)

class hesel : public PhysicsModel {
public:
    hesel();
    uint64_t rhs_wall_time_not_including_neutrals{0};

protected:
    int init(bool restart);
    int rhs(BoutReal t);
    //Helper method
    BoutReal SumField(Field3D f);

private:
    //Time measurement
    uint64_t t1, t2;

    // code test flags
    bool compact_output;
    bool test_blob;
    bool test_blob_use_gen_vort;
    bool test_laplace;

	  // Fast output object
	  FastOutput fast_output;

    // 3D fields, in HESEL is 2D
    Field3D lnn, lnpe, lnpi, vort; // evolving fields: density, electron pressure, ion pressure, and generalized vorticity logarithm
    Field3D lnte, lnti;
    Field3D ddt_lnn, ddt_lnpe, ddt_lnpi, ddt_vort;
    Field3D n, pe, pi, te, ti, csHot, tau; // physical fields: density, temperature, pressure and ion sound speed
    Field3D pert_n, pert_te, pert_phi; // perturbed fields: n, Te and phi
    Field3D tau_n; // damping rate of advection

    // 2D fields, in HESEL is 1D
    Field2D avg_n, avg_phi, avg_te, avg_ti, avg_csHot, avg_tau; // average fields: n, phi, Te, csHot

    // 2D static fields, in HESEL is 1D
    Field2D init_n, init_pe, init_pi; // initial profile: n, pe, pi
    Field3D seed_n, seed_pe, seed_pi; // initial field perturbations
    Field2D init_te, init_ti; // helping 2D field
    Field2D sigma_open, sigma_closed, sigma_force; // step function for open and closed field lines and for forcing profiles
    Field2D B, invB; // Magnetic field, 1/B
    Field2D invR;

    // 3D fields, dissipating terms
    Field3D Qdelta_Rcppe;// electron and ion heat exchange term due to elastic collisions
    Field3D Qresist_Rcppi;// electron and ion heat exchange term due to resistivity
    Field3D Qviscous; // viscous heating in ion pressure equation
    Field3D DampSheath;// sheath damping
    Field3D DampAdvection; // advection damping
    Field3D DampShe, DampShi; // Spitzer-Harm conduction
    Field3D DriftWave; // drift wave term in Nielsen 2017 Equation 7

    // Vorticity and potential related fields
    Field3D evort, delpi, phi, phi_star, vort_residual;

    // thermal energy
    Field3D qe_para_advection,  qi_para_advection;
    Field3D qe_para_conduction, qi_para_conduction;
    Field3D qe_perp_neoclassic, qi_perp_neoclassic;
    Field3D qe_perp_turbulence, qi_perp_turbulence;
    Field3D part_turb_flux, part_diff_flux;

    Field2D avg__qe_para_advection, avg__qi_para_advection;
    Field2D avg__qe_para_conduction, avg__qi_para_conduction;
    Field2D avg__qe_perp_neoclassic, avg__qi_perp_neoclassic;
    Field2D avg__qe_perp_turbulence, avg__qi_perp_turbulence;
    Field2D avg__part_turb_flux, avg__part_diff_flux;

    // 3D vector fields
    Vector3D gradp_phi, gradp_pi; //gradients of phi and Pi for use in vorticity equation
    Vector3D gradp_lnn, gradp_lnte, gradp_lnti;

    // drift velocities
    Vector3D uR;//resistive drift velocity
    Vector3D uExB;//ExB drift velocity

    // logarithm terms: avoid explicit division
    Field3D Deln_Rcpn, Delte_Rcpte, Delti_Rcpti; // Delp2(f)/f
    Field3D Gradn_Gradte_Rcppe, Gradn_Gradti_Rcppi; // Grad_perp(n)*Grad_perp(te)/pe, Grad_perp(n)*Grad_perp(ti)/pi

    // full model
    Field3D De, Di, Dn;
    Field3D Div_gammaR_Rcpn;
    Field3D ti_rcpte;

    // rampup initial profiles
    BoutReal n_inner, pe_inner, pi_inner;
    BoutReal rampup_init_time;
    Field2D init_n_fake, init_pe_fake, init_pi_fake;
    Field2D init_n0, init_pe0, init_pi0;

	// Synthetic probes
	std::fstream fs;
    BoutReal x_pos;
    std::list<int> probe_list;
    std::string datadir, probe_file;

//******************************************************************************

    // Group fields for calculating and communicating
    FieldGroup field_group_ln;
    FieldGroup field_group_ddt;
    FieldGroup field_group_pert;
    FieldGroup field_group_avg;
    FieldGroup field_group_gradp;
    FieldGroup field_group_phyd;

//******************************************************************************

    // Instances of HESEL-Common classes
    HeselParameters HeselPara;// HESEL collisional parameters
    Neutrals neutrals;// HESEL neutral model
    Parallel parallel;// HESEL parallel transport model
    KineticNeutrals kinetic_neutrals;

//******************************************************************************

    // Switches
    bool use_grid_file;
    bool right_handed_coord;
//    int coord_sign; // 1. for right handed coordinate system, -1. for left-handed, used to determine sign for C() and Brackets()
    bool invert_w_star; //should Laplacian inversion be on w or w^*?
    bool interchange_dynamics; // switch for interchange dynamics
    bool test_vort_cross_term; // switch for {Grad(phi), Grad(pi)}
    bool parallel_dynamics; // parallel dynamics
    bool perpendicular_dynamics; // perpendicular dynamics
    int plasma_neutral_interactions;
    int kinetic_plasma_neutral_interactions;
    bool source_n, source_pe, source_pi, source_vort;
    bool parallel_transport;
    bool force_profiles; //should profiles be forced in inner edge region ?
    bool floor_profiles; //should we force fields to stay above given level
    bool rampup_ti0; //ramp up ion temperature during run
    COLLISIONAL_MODEL collisional_model_enum;
    int collisional_model;
    int static_plasma; //make plasma field static. bit digits (n,pe,pi,vort), 0=all, 1=vort, 2=pi, 4=pe, 8=n
    int rampup_init_prof; // ramp up initial profiles
    bool diagnostics;
    bool init_zero_potential;
    bool Lc_from_grid;

//******************************************************************************

    // Parameters
    BoutReal Bt, q, Rmajor, Rminor, A, lblob, Z, Mach; // physical and engineering parameters
    Field2D lconn; // allow for lconn profile
    BoutReal n0, Te0, Ti0, B0; // reference values at LCFS
    BoutReal Z_eff; // Effective multiplier for diffusion coefficients
    BoutReal force_time; //#rate by which profiles are forced. d/dt lnn = (n-n_profile)/force_rate
    BoutReal floor_n, floor_pe, floor_pi, floor_time; //lower bounds for fields. if floor_fields == true field will we forced to floor level
    BoutReal rampup_ti0_timax; //ion temperature at t=rampup_time, for ramping up ion temperature
    BoutReal rampup_ti0_time; //time-scale for ramping up the ion temperature at the inner boundary
    BoutReal nuii, nuei, nuee; //variables holding parameters
    BoutReal oci, rhoe, rhos;// ion gyro-frequency, electron gyro-radius, ion hybrid gyro-radius
    BoutReal x_lcfs, x_wall;// variables define edge, SOL and wall regions
    int lcfs_ind; // index of lcfs along x axis
//    BoutReal thermal_energy;
    int parallel_conduction;
    int parallel_sheath_damping;
    int parallel_advection_damping;
    int parallel_drift_wave;
    bool perpend_heat_exchange;
    bool perpend_viscous_heating;
    bool perpend_hyper_diffusion;
    int ti_over_te; // setting for approximation for ti/te
    int diffusion_coeff; // approximation for diffusion coefficients
    int qdelta_approx; // approximation of nuei in Qdelta
    int reciprocal_approx; // approximation for reciprocals in parallel dynamics
    BoutReal neoclass_correction_factor;
    bool static_n, static_pe, static_pi, static_vort, static_te, static_ti;
    bool rampup_init_n, rampup_init_pe, rampup_init_pi;
    BoutReal hyper_dc_n, hyper_dc_pe, hyper_dc_pi, hyper_dc_vort;
    int hyper_order;
    int diag_thermal;
    bool diag_thermal_para_ad, diag_thermal_para_sh;
    bool diag_thermal_perp_neo, diag_thermal_perp_turb;
    bool diag_particle_flux;
    bool double_curvature_coeff;
    int curvature_coeff;

//******************************************************************************

    // Vorticity and potential related variables and class
    class Laplacian* phiSolver;           // Laplacian solver for vort -> phi
    BRACKET_METHOD bracket_method_enum; // Bracket method for advection terms
    int bracket_method;

//******************************************************************************

    // Methods related to init()
    int InitMesh(); // load information about the mesh e.g. domain size Lx and Lz
    int InitParameters(); // load parameters specific for the Hesel model (e.g. background temperature)
    int InitFieldGroups(); // initialize different field groups
    int InitIntialProfiles();// read initial profiles
    int InitRampInitProfiles();
    int InitEvolvingFields(); // initialize evolving fields
    int InitOtherFields(); // initialize various fields
    int InitMagneticField(); // calculate slab magnetic field
    int InitDiagnostics(); // setup diagnostics, currently unused
    int InitPhiSolver(); // setup poisson equaton solver used to calculate phi from vorticity (\nabla^2_{\perp} \phi)
    int InitNeutrals(); // setup nertural model
    int InitParallel(); // setup nertural model
    int InitKineticNeutrals(bool restart);
    int InitTestBlob();

//******************************************************************************

    // Methods related to rhs()
    int RhsEvolvingFieldsStart();
    int RhsInterchangeDynamics(); // non-collisional part of HESEL equation on LHS
    int RhsPerpendicularDynamics(); // collional part of HESEL equation on RHS
    int RhsCollisionalSimple();
    int RhsCollisionalStandard(); // standard particaly linearized collisional model (HESEL in Jens's paper)
    int RhsCollisionalFull(); // full nonlinear drift ordered Braginskii based collisional model (Full Model in Jens's paper)
    int RhsParallelDynamics();
    int RhsParallelDampSheath();
    int RhsParallelDampAdvection();
    int RhsParallelDriftWave(); // drift waves dynamics according to Ander's paper
    int RhsDiagnostics(BoutReal t);
    int RhsPlasmaNeutralInteraction(BoutReal t); // add neutral sources, sinks etc to plasma field rhs's (e.g. ionization particle density source )
    int RhsKineticPlasmaNeutralInteractionInit(BoutReal t);
    int RhsKineticPlasmaNeutralInteractionFinal();
    int RhsRampInitProfiles(BoutReal t); // ramp up initial profiles
    int RhsPlasmaFieldsForce(); // force plasma fields. Typical in the inner edge region
    int RhsPlasmaFieldsFloor(); // elevate plasma fields if they are below certain threshold. the method used here is perhaps not very good since the underlyning newton method used in the PVOIDE and CVode time integration schemes requires continous fields, probably violated by flooring plasma. Instead use Sigmoids of something similar.
    int RhsParallelTransport(BoutReal t); // Add full parallel transport dynamics
    int RhsRampTiReference(BoutReal t); // ramp up reference ion temperature to a maximum value with certain time
    int RhsPlasmaFieldsStatic(); // make plasma field static, typical for testing neutral model
    int RhsTestBlob();
    int RhsEvolvingFieldsEnd();

 //******************************************************************************

    // Methods calculating fields as a bundle in field group
    int CalculatePhi(); // invert vorticity to calculate \phi^* and \phi
    int CalculateFieldGroupAvg(); // average fields
    int CalculateFieldGroupGradp(); // vectors
    int CalculateFieldGroupPert(); // perturbed fields
    int CalculateFieldGroupPhyd(); // physical and derived fields
    int CalculateDiffusionCoeff();// De for full model
    int Calculate_uR(int c);// uR, resistive drift velocity
    int Calculate_uExB(); // uExB, ExB drift velocity
 //******************************************************************************

    // Operators
    Field3D Brackets(const Field3D &f, const Field3D &g); // self defined bracket
    Field2D ZAvg(Field3D f); //poloidal average -- z-average
    Field3D C(const Field3D &f); //curvature operator
    Field3D Delp4(const Field3D &var);
    Field3D Delp6(const Field3D &var);
    std::string bool2str(bool s);
    int bool2int(bool b);
    Field2D LinearSlope(BoutReal ls_intercept, BoutReal ls_turning_x, BoutReal ls_turning_y);
    Vector3D b0xGrad(const Field3D &var);
    int sigma_adjust(Field3D &var, int region);

};



#endif /* INCLUDE_HESEL_HXX_ */
