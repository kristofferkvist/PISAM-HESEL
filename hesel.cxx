/*
 * hesel.cxx
 *
 * This file holds the main codes for HESEL model based on Jen. Madsen's POP 2016
 * paper. Two collisional model is included: standard simplified HESEL model
 * and full model that contains higher order drifts. Plasma neutral interaction
 * is also included based on Alexander's work. Drift wave dynamics is included
 * according to Ander's PPCF 2017 paper.
 *
 * Created on: Sep 29, 2015
 *     Author: yolen
 *
 * Change Log
 *     2017-01-10 by alexander:
 *         1. Include neutral models.
 *         2. Rearrange code structure.
 *     2017-04-28 by Jens Madsen:
 *         1. Upgrade to BOUT++ version 4
 *         2. Rearrange code structure.
 *         3. Separate HeselParameters to be a submodule.
 *     2017-06-06 by alexander:
 *         1. Add plasma neutral interaction model.
 *     2018-02-05 by jeppe:
 *         1. Correct mistakes in LFS of vorticity equation, add missing terms in
 *            RHS of ion pressure equation and correct mistakes in HeselParameters.
 *     2018-07-24 by xiang liu:
 *         1. Clean and restructure the code.
 *         2. Correct bug in initialization of magnetic fields.
 *         3. Solve problem that no boundary value for using bracket.
 *         4. Check through and correct several mistakes according to Jens Madsen
 *            POP paper.
 *         5. Add drift wave dynamics.
 *         6. Revise initial profile functions, make pressure profile linear drops
 *            in the edge.
 *         7. Adjust C() and Brackets() to suite for right-handed coordinate system
 *         8. Rewrite full collisional model.
 *         9. Separate the equations and add switches for different dynamics
 *         10. Add invert_w_star
 *         11. Reformulate and not using reciprocal explicitly, which might lead to
 *             sharp points near steep gradient and stop the time solver.
 *         12. reorganize physical and derived fields, avoid using reciprocal.
 *     2018-08-07 by xiang liu
 *         1. find a solution for convergence error for full model and for low q or
 *            high Te in std model
 *         2. add a bunch of controls for approximations
 *     2018-08-20 by xiang liu
 *         1. add feature: use experiment profiles stored in grid file
 *     2018-09-03 by xiangl
 *         add feature: ramp up initial profiles
 *     2018-09-07 by xiangl
 *         add hyper diffusion with 4th oder
 *     2018-09-21 by xiangl
 *         reorganize hyper diffusion code with switches added and add 6th order
 *     2018-09-27 by xiangl
 *         add parallel thermal energy diagnostics and reorganize parallel dynamics
 *     2018-09-28 by xiangl
 *         add perpendicular thermal energy diagnostics
 *     2018-12-04 by xiangl
 *         fix bugs in DriftWave and C(), adjust viscous heating \eta with full variation
 *
 * TO DO: neutral model cleanup and test
 */

#include "hesel.hxx"

/// ****** Constructor ******
hesel::hesel() : neutrals(solver,HeselPara,n,phi,pi,pe,B),
		 parallel(solver,HeselPara,n,phi,vort,pi,pe,B),
		 kinetic_neutrals(HeselPara, n, te, ti, phi, B)
{}

/// ****** Overwrite Initialization Function ******
int hesel::init(bool restart) {
    InitMesh();
    InitParameters();
    InitMagneticField();
    InitFieldGroups();
    InitIntialProfiles();
    InitRampInitProfiles();
    InitEvolvingFields();
    InitOtherFields();
    InitPhiSolver();
    InitDiagnostics();
    InitNeutrals();
    InitParallel();
		InitKineticNeutrals(restart);

		if (fast_output.enabled) {

		  // Add monitor if necessary
		    if (fast_output.enable_monitor) {
		      solver->addMonitor(&fast_output);
		    }

		  // Add points from the input file
		  int i = 0;
		  BoutReal xpos, ypos, zpos;
		  int ix, iy, iz;
		  Options* fast_output_options = Options::getRoot()->getSection("fast_output");
		  while (true) {
		    // Add more points if explicitly set in input file
		    fast_output_options->get("xpos"+std::to_string(i), xpos, -1.);
		    fast_output_options->get("ypos"+std::to_string(i), ypos, -1.);
		    fast_output_options->get("zpos"+std::to_string(i), zpos, -1.);
		    if (xpos<0. || ypos<0. || zpos<0.) {
		      output.write("\tAdded %i fast_output points\n", i);
		      break;
		    }
		    ix = int(xpos*mesh->GlobalNx);
		    iy = int(ypos*mesh->GlobalNy);
		    iz = int(zpos*mesh->GlobalNz);

		    // Add fields to be monitored
		    fast_output.add("n"+std::to_string(i), n, ix, iy, iz);
		    fast_output.add("phi"+std::to_string(i), phi, ix, iy, iz);
		    fast_output.add("te"+std::to_string(i), te, ix, iy, iz);
		    fast_output.add("ti"+std::to_string(i), ti, ix, iy, iz);
			if (plasma_neutral_interactions){
		      fast_output.add("Ncold"+std::to_string(i), neutrals.Ncold, ix, iy, iz);
		      fast_output.add("Nwarm"+std::to_string(i), neutrals.Nwarm, ix, iy, iz);
		      fast_output.add("Nhot"+std::to_string(i), neutrals.Nhot, ix, iy, iz);
		    }
			i++;
		  }
		}

		/*if (intercomm_monitor.enabled){
			solver->addMonitor(&intercomm_monitor);
		}*/

    return 0;
}

/// ****** Overwrite Time Evolution Function ******
int hesel::rhs(BoutReal t) {

    RhsEvolvingFieldsStart(); // Communicate evolving fields
		RhsKineticPlasmaNeutralInteractionInit(t);
		t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    CalculateFieldGroupPhyd();// calculate derived physical fields

    CalculatePhi(); // calculate phi from generalized vorticity
    RhsTestBlob(); // blob testing code, must put just after CalculatePhi()

    CalculateFieldGroupGradp();// calculate vector fields, only call after phi has been calculated
    CalculateFieldGroupAvg(); // calculate average fields, only call after phi has been calculated

    RhsInterchangeDynamics(); // non collisional part of equations
    RhsPerpendicularDynamics(); // add collisional effects to the RHS's of the plasma fields e.g. diffusion to particle density field.
    RhsParallelDynamics(); // parallel damping dynamics
    RhsDiagnostics(t);
		//RhsKineticPlasmaNeutralInteraction(t);
    RhsPlasmaNeutralInteraction(t);
    RhsRampInitProfiles(t);
    RhsParallelTransport(t);
    RhsPlasmaFieldsForce(); // force field towards fixed profiles in inner edge region
    RhsPlasmaFieldsFloor(); // if required make sure that field stay above a minimum level (a floor)
    RhsRampTiReference(t); // ramp up reference ion temperature
    RhsPlasmaFieldsStatic(); // special case were plasma i static and neutral are evolved
		t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		rhs_wall_time_not_including_neutrals += t2 - t1;
		RhsKineticPlasmaNeutralInteractionFinal();
    RhsEvolvingFieldsEnd();
    if (fast_output.enabled){
	    fast_output.monitor_method(t); // Store fast output in BOUT.fast.<processor_no.>
    }
		/*
		std::cout << "###############  ddt_lnn = " << SumField(ddt_lnn) << std::endl;
		std::cout << "###############  ddt_lnpe = " << SumField(ddt_lnpe) << std::endl;
		std::cout << "###############  ddt_lnpi = " << SumField(ddt_lnpi) << std::endl;
		*/
		return 0;
}



BoutReal hesel::SumField(Field3D f){
	BoutReal sum = 0;
	int loc_proc_nx = n.getNx()-2;
	int loc_proc_ny = n.getNy();
	int loc_proc_nz = n.getNz();
	int N_per_proc = loc_proc_nx*loc_proc_ny*loc_proc_nz;
	for(int i = 0; i < loc_proc_nx; i++){
		for(int j = 0; j < loc_proc_ny; j++){
			for(int k = 0; k < loc_proc_nz; k++){
				//mxg guard cells are placed in the x-dimension. Dont read any data into these cells.
				 sum += abs(f(i+1, j, k))/(N_per_proc);
				 //std::cout << "My field at " << i << ", " << j << ", " << k << " is " << buffer_field(i, j, k) << std::endl;
			}
		}
	}
	return sum;
}


//******************************************************************************

int hesel::InitMesh(){
    // TO DO: would it possible to set metric tensor to change to right-handed
    // coordinate system?
    TRACE("Updating geometry");

    // Options *meshoptions = Options::getRoot()->getSection("mesh");

    // set toroidal magnetic field
    mesh->getCoordinates()->Bxy = 1. ;

    // covariant metric tensor and magnetic field
    mesh->getCoordinates()->g11 = 1.0 ;
    mesh->getCoordinates()->g22 = 1.0 ;
    mesh->getCoordinates()->g33 = 1.0 ;
    mesh->getCoordinates()->g12 = 0.0 ;
    mesh->getCoordinates()->g13 = 0.0 ;
    mesh->getCoordinates()->g23 = 0.0 ;

    //...contravariant
    mesh->getCoordinates()->g_11 = 1.0 ;
    mesh->getCoordinates()->g_22 = 1.0 ;
    mesh->getCoordinates()->g_33 = 1.0 ;
    mesh->getCoordinates()->g_12 = 0.0;
    mesh->getCoordinates()->g_13 = 0.0;
    mesh->getCoordinates()->g_23 = 0.0;

    mesh->getCoordinates()->geometry();

    if (mesh->sourceHasVar("init_n")){
        use_grid_file = true;
        output << ">>>>>>> Using Initial Profiles in Grid File <<<<<<<" << endl;
    } else {
        use_grid_file = false;
    }

    return 0;
}

int hesel::InitParameters(){
    TRACE("Loading parameters");

    Options *opt_hesel = Options::getRoot()->getSection("hesel");

    // switches and options
    OPTION(opt_hesel, compact_output,              false);
    OPTION(opt_hesel, test_blob,                   false);
    OPTION(opt_hesel, test_laplace,                false);
    OPTION(opt_hesel, right_handed_coord,          false);
    OPTION(opt_hesel, interchange_dynamics,        true);
    OPTION(opt_hesel, parallel_dynamics,           true);
    OPTION(opt_hesel, perpendicular_dynamics,      true);
    OPTION(opt_hesel, invert_w_star,               false);
    OPTION(opt_hesel, plasma_neutral_interactions, 0);
		OPTION(opt_hesel, kinetic_plasma_neutral_interactions, 0);
		if (plasma_neutral_interactions && kinetic_plasma_neutral_interactions){
			throw BoutException("ERROR: Kinetic AND fluid neutrals is switched on. Choose one or neither");
		}

    OPTION(opt_hesel, parallel_transport,          false);
    OPTION(opt_hesel, force_profiles,              true);
    OPTION(opt_hesel, floor_profiles,              false);
    OPTION(opt_hesel, rampup_ti0,                  false);
    OPTION(opt_hesel, static_plasma,               false);
    OPTION(opt_hesel, rampup_init_prof,            false);
    OPTION(opt_hesel, diagnostics,                 false);
    OPTION(opt_hesel, init_zero_potential,         false);


    // sub switches and options
    OPTION(opt_hesel, test_blob_use_gen_vort,      false);
    OPTION(opt_hesel, test_vort_cross_term,        false);
    OPTION(opt_hesel, collisional_model,           1);
    OPTION(opt_hesel, perpend_heat_exchange,       true);
    OPTION(opt_hesel, perpend_viscous_heating,     true);
    OPTION(opt_hesel, perpend_hyper_diffusion,     false);
    OPTION(opt_hesel, parallel_conduction,         1);
    OPTION(opt_hesel, parallel_sheath_damping,     2);
    OPTION(opt_hesel, parallel_advection_damping,  3);
    OPTION(opt_hesel, parallel_drift_wave,         1);
    OPTION(opt_hesel, ti_over_te,                  3);
    OPTION(opt_hesel, diffusion_coeff,             1);
    OPTION(opt_hesel, reciprocal_approx,           3);
    OPTION(opt_hesel, neoclass_correction_factor,  -1); // use default value: R/a*q^2
    OPTION(opt_hesel, qdelta_approx,               1);
    OPTION(opt_hesel, diag_particle_flux,          false);
    OPTION(opt_hesel, diag_thermal,                0);
    OPTION(opt_hesel, double_curvature_coeff,      false);
    OPTION(opt_hesel, Lc_from_grid,                false);


    OPTION(opt_hesel, Z_eff,                       1);

    // bracket method
    OPTION(opt_hesel, bracket_method, 2);

    // settings for modifying profiles
    OPTION(opt_hesel, force_time, 50.);
    OPTION(opt_hesel, floor_time, 0.);
    OPTION(opt_hesel, floor_n,    0.);
    OPTION(opt_hesel, floor_pe,   0.);
    OPTION(opt_hesel, floor_pi,   0.);
    OPTION(opt_hesel, rampup_init_time, 1000.);
    SAVE_ONCE2(force_time, floor_time);
    SAVE_ONCE3(floor_n, floor_pe, floor_pi);
    SAVE_ONCE(rampup_init_time);

    // settings for hyper diffusion
    OPTION(opt_hesel, hyper_dc_n,    1);
    OPTION(opt_hesel, hyper_dc_pe,   1);
    OPTION(opt_hesel, hyper_dc_pi,   1);
    OPTION(opt_hesel, hyper_dc_vort, 1);
    OPTION(opt_hesel, hyper_order,   4);
    SAVE_ONCE2(hyper_dc_n, hyper_dc_pe);
    SAVE_ONCE2(hyper_dc_pi, hyper_dc_vort);
    SAVE_ONCE(hyper_order);

    // physical and engineering parameters
    OPTION(opt_hesel, Rmajor, 1.90);
    OPTION(opt_hesel, Rminor, 0.45);
    OPTION(opt_hesel, Bt,     2.3);
    OPTION(opt_hesel, q,      5);
    OPTION(opt_hesel, Te0,    20.);
    OPTION(opt_hesel, Ti0,    20.);
    OPTION(opt_hesel, n0,     1.5E19);
    OPTION(opt_hesel, A,      2.);
    OPTION(opt_hesel, Z,      1.);
    if (Lc_from_grid){
        mesh->get(lconn,  "lconn" );
    }else{
        OPTION(opt_hesel, lconn,  15.);
    }
    OPTION(opt_hesel, lblob,  -1); // use default value: q*R
    OPTION(opt_hesel, Mach,   1.);

    B0 = Bt*Rmajor/(Rmajor+Rminor); // should use reference value at LCFS

    HeselPara.R(Rmajor);
    HeselPara.a(Rminor);
    HeselPara.B(B0);
    HeselPara.q(q);
    HeselPara.Te(Te0);
    HeselPara.Ti(Ti0);
    HeselPara.n(n0);
    HeselPara.A(A);
    HeselPara.Z(Z);
    HeselPara.lconn(lconn);
    HeselPara.Mach(Mach);
    HeselPara.lblob(lblob);
    HeselPara.neocorr_factor(neoclass_correction_factor);
    lconn = HeselPara.lconn();
    lblob = HeselPara.lblob();
    neoclass_correction_factor = HeselPara.neocorr_factor();
    SAVE_ONCE4(Bt, Rmajor, Rminor, q);
    SAVE_ONCE4(Te0, Ti0, n0, B0);
    SAVE_ONCE2(A, Z)
    SAVE_ONCE3(lconn, lblob, Mach);
    SAVE_ONCE(neoclass_correction_factor);

    // ramp up ion temperatures
    OPTION(opt_hesel, rampup_ti0_timax,  Ti0);
    OPTION(opt_hesel, rampup_ti0_time, 0.);
    SAVE_ONCE2(rampup_ti0_timax, rampup_ti0_time);

    // regions separating parameters
    OPTION(opt_hesel, x_lcfs, 0.33333);
    OPTION(opt_hesel, x_wall, 0.66666);
    lcfs_ind = (mesh->GlobalNx - 2*mesh->xstart) * x_lcfs + mesh->xstart;
    SAVE_ONCE2(x_lcfs, x_wall);

    // curvature operator
    curvature_coeff = 1;
    if (double_curvature_coeff){
        curvature_coeff = 2;
    }

    // dump calculated parameters
    oci  = HeselPara.oci();
    nuii = HeselPara.nuii();
    nuei = HeselPara.nuei();
    nuee = HeselPara.nuee();
    rhoe = HeselPara.rhoe();
    rhos = HeselPara.rhos();
    SAVE_ONCE(oci);
    SAVE_ONCE3(nuii, nuei, nuee);
    SAVE_ONCE2(rhoe, rhos);

// *****************************************************************************

    output << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;

    // warning message for blob testing
    if (test_blob){
        output << ">>>>>>> [WARNING]: BLOB TESTING IS TURNED ON!" << endl;
    }

    // warning message for Laplace testing
    if (test_laplace){
        output << ">>>>>>> [WARNING]: LAPLACE TESTING IS TURNED ON!" << endl;
    }

    // output for switches part 1
    output << ">>>>>>> Use right-handed coordinate: "   << bool2str(right_handed_coord) << endl;
    output << ">>>>>>> Invert Omega star: "   << bool2str(invert_w_star) << endl;
    output << ">>>>>>> Interchange dynamics: " << bool2str(interchange_dynamics) << endl;
    if (interchange_dynamics && test_vort_cross_term){
        output << ">>>>>>> [WARNING]: cross term of vorticity equation in interchange dynamics is turned off!" << endl;
    }

    // output for perpendicular dynamics
    output << ">>>>>>> Perpendicular dynamics: "   << bool2str(perpendicular_dynamics) << endl;
    if (perpendicular_dynamics){
        // output for collisional models
        switch (collisional_model){
            case 0:{
                collisional_model_enum = off;
                output << ">>>>>>>     1) collisional model: off" << endl;
                break;
            }
            case 1:{
                collisional_model_enum = simple;
                output << ">>>>>>>     1) collisional model: simple" << endl;
                break;
            }
            case 2:{
                collisional_model_enum = standard;
                output << ">>>>>>>     1) collisional model: standard" << endl;
                break;
            }
            case 3:{
                collisional_model_enum = full;
                output << ">>>>>>>     1) collisional model: full" << endl;
                break;
            }
            default:{
                throw BoutException("ERROR: Collisional model not recognized!\n");
            }
        }
        output << ">>>>>>>     2) heat   exchange: "   << bool2str(perpend_heat_exchange) << endl;
        output << ">>>>>>>     3) viscous heating: "   << bool2str(perpend_viscous_heating) << endl;
        output << ">>>>>>>     4) hyper diffusion: "   << bool2str(perpend_hyper_diffusion) << endl;
        output << ">>>>>>>     5) approx of ti/te: "   << ti_over_te << endl;
        output << ">>>>>>>     6) diffusion coeff: "   << diffusion_coeff << endl;
        output << ">>>>>>>     7) Qdelta   approx: "   << qdelta_approx << endl;
    }

    // output for parallel dynamics
    output << ">>>>>>> Parallel dynamics: "   << bool2str(parallel_dynamics) << endl;
    if (parallel_dynamics){
        output << ">>>>>>>     1) sheath    : "   << parallel_sheath_damping << endl;
        output << ">>>>>>>     2) advection : "   << parallel_advection_damping << endl;
        output << ">>>>>>>     3) conduction: "   << parallel_conduction << endl;
        output << ">>>>>>>     4) drift wave: "   << parallel_drift_wave << endl;
        output << ">>>>>>>     5) rcp approx: "   << reciprocal_approx << endl;
    }

    // output for switches part 2
    output << ">>>>>>> Plasma neutral interactions: "   << bool2str(plasma_neutral_interactions) << endl;
		output << ">>>>>>> Kinetic Plasma neutral interactions: "   << bool2str(kinetic_plasma_neutral_interactions) << endl;
    output << ">>>>>>> Plasma parallel transport: "   << bool2str(parallel_transport) << endl;
    output << ">>>>>>> Force profiles: "   << bool2str(force_profiles) << endl;
    output << ">>>>>>> Floor profiles: "   << bool2str(floor_profiles) << endl;
    output << ">>>>>>> Ramp up Ti0: "   << bool2str(rampup_ti0) << endl;

    // output for static plasma
    static_n    = static_plasma & 1;
    static_pe   = static_plasma & 2;
    static_pi   = static_plasma & 4;
    static_vort = static_plasma & 8;
    static_te   = static_plasma & 16;
    static_ti   = static_plasma & 32;
    if (static_plasma == 0){
        output << ">>>>>>> Static plasma: " << bool2str(false) << endl;
    }else{
        if (static_pi && static_ti){
            throw BoutException("ti and pi can not be both static, please set n and ti static instead!");
        }
        if (static_pe && static_te){
            throw BoutException("te and pe can not be both static, please set n and ti static instead!");
        }
        output << ">>>>>>> Static plasma: | n | pe | pi | vort | te | ti |"   << endl;
        output << "                       | ";
        output <<                           static_n;
        output <<                         " | "<< static_pe;
        output <<                         "  | "<< static_pi;
        output <<                         "  |  "<< static_vort;
        output <<                                          "   | " << static_te;
        output <<                                          "  | " << static_ti;
        output <<                                          "  |" << endl;
    }

    // output for ramping up init profiles
    rampup_init_n  = rampup_init_prof & 1;
    rampup_init_pe = rampup_init_prof & 2;
    rampup_init_pi = rampup_init_prof & 4;
    if (rampup_init_prof == 0){
        output << ">>>>>>> Ramp up Initial Profiles: " << bool2str(false) << endl;
    }else{
        output << ">>>>>>> Ramp up Initial Profiles:  | n | pe | pi |"   << endl;
        output << "                                   | ";
        output <<                                       rampup_init_n;
        output <<                                     " | "<< rampup_init_pe;
        output <<                                     "  | "<< rampup_init_pi;
        output <<                                          "  |" << endl;
    }

    // output for diagnostics
    diag_thermal_para_ad   = diag_thermal & 1;
    diag_thermal_para_sh   = diag_thermal & 2;
    diag_thermal_perp_neo  = diag_thermal & 4;
    diag_thermal_perp_turb = diag_thermal & 8;
    output << ">>>>>>> Diagnostics: "   << bool2str(diagnostics) << endl;
    if (diagnostics) {
        output << ">>>>>>>     1) thermal energy (ad, sh, neo, turb): "   <<
                bool2int(diag_thermal_para_ad) <<
                bool2int(diag_thermal_para_sh) <<
                bool2int(diag_thermal_perp_neo) <<
                bool2int(diag_thermal_perp_turb) << endl;
    }

    // output for bracket methods
    switch(bracket_method) {
    case 0: {
        bracket_method_enum = BRACKET_STD;
        output << ">>>>>>> Brackets for ExB: default differencing" << endl;
        break;
    }
    case 1: {
        bracket_method_enum = BRACKET_SIMPLE;
        output << ">>>>>>> Brackets for ExB: simplified operator" << endl;
        break;
     }
    case 2: {
        bracket_method_enum = BRACKET_ARAKAWA;
        output << ">>>>>>> Brackets for ExB: Arakawa scheme" << endl;
        break;
    }
    case 3: {
        bracket_method_enum = BRACKET_CTU;
        output << ">>>>>>> Brackets for ExB: Corner Transport Upwind method" << endl;
        break;
    }
    default:
        throw BoutException("ERROR: Invalid choice of bracket method. Must be 0 - 3!\n");
    }

    // print parameters
    output << endl;
    HeselPara.printvalues(); //print to standard output

    output << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;

    return 0;
}

int hesel::InitMagneticField(){
    //Calculate magnetic field - gyro-Bohm normalized
    TRACE("Initialize magnetic field");

    B.allocate();
    invR.allocate();
    for (auto i:B){
        invR[i] = 1. / ( HeselPara.R() + HeselPara.a()
            + HeselPara.rhos() * mesh->getCoordinates()->dx[i]
            * (mesh->GlobalNx * mesh->GlobalX(i.x()) - lcfs_ind) );
        B[i] = (HeselPara.R()+HeselPara.a()) * invR[i];
    }

    SAVE_ONCE(B);

    invB = 1./B;

    return 0;
}

int hesel::InitFieldGroups(){
    //parallel communication of fields
    field_group_ln.add(lnn, lnpe, lnpi);
    field_group_ddt.add(ddt_lnn, ddt_lnpe, ddt_lnpi, ddt_vort);
    field_group_phyd.add(n, pe, pi, te, ti, lnte, lnti, csHot, tau);
    field_group_gradp.add(gradp_phi, gradp_pi);
    field_group_gradp.add(gradp_lnn, gradp_lnte, gradp_lnti);
    field_group_avg.add(avg_n, avg_te, avg_phi, avg_csHot, avg_tau);
    field_group_pert.add(pert_n, pert_te, pert_phi);

    return 0;
}

int hesel::InitIntialProfiles(){
    TRACE("Load initial profiles");

    // read initial profiles
    if (use_grid_file){
        mesh->get(init_n,  "init_n" );
        mesh->get(init_pe, "init_pe");
        mesh->get(init_pi, "init_pi");
    } else {
        initial_profile("init_n",  init_n);
        initial_profile("init_pe", init_pe);
        initial_profile("init_pi", init_pi);
    }
    initial_profile("seed_n",  seed_n);
    initial_profile("seed_pe",  seed_pe);
    initial_profile("seed_pi",  seed_pi);

    // Check for negatives
    if (min(init_n, true) < 0.0) {
      throw BoutException("Starting density is negative");
    }
    if (min(init_pe, true) < 0.0) {
      throw BoutException("Starting electron pressure is negative");
    }
    if (min(init_pi, true) < 0.0) {
      throw BoutException("Starting ion pressure is negative");
    }
    if (min(seed_n, true) < 0.0) {
      throw BoutException("Starting density perturbation is negative");
    }

    // save initial profiles
    if (!rampup_init_prof){
        SAVE_ONCE(init_n);
        SAVE_ONCE(init_pe);
        SAVE_ONCE(init_pi);
    }
    SAVE_ONCE3(seed_n, seed_pe, seed_pi);

    // gather inner edge value
    n_inner  = init_n (0,0);
    pe_inner = init_pe(0,0);
    pi_inner = init_pi(0,0);

    // derived initial variables
    init_te = init_pe/init_n;
    init_ti = init_pi/init_n;

    return 0;
}

int hesel::InitRampInitProfiles(){
    TRACE("Initialize process of ramping up initial profiles");
    if (rampup_init_prof){
        init_n0 = 0; init_pe0 = 0; init_pi0 = 0;
        init_n_fake = 0; init_pe_fake = 0; init_pi_fake = 0;
        for (auto i:init_n){
            init_n0[i]  = init_n[i];
            init_pe0[i] = init_pe[i];
            init_pi0[i] = init_pi[i];

            init_n_fake[i]  = init_n[i];
            init_pe_fake[i] = init_pe[i];
            init_pi_fake[i] = init_pi[i];

            if (rampup_init_n && init_n_fake[i] > 1) {init_n_fake[i] = 1.;}
            if (rampup_init_pe && init_pe_fake[i] > 1) {init_pe_fake[i] = 1.;}
            if (rampup_init_pi && init_pi_fake[i] > 1) {init_pi_fake[i] = 1.;}
        }
        dump.add(init_n0, "init_n", 0);
        dump.add(init_pe0, "init_pe", 0);
        dump.add(init_pi0, "init_pi", 0);

        RhsRampInitProfiles(0);
    }

    return 0;
}

int hesel::InitEvolvingFields(){
    ///Fields that are time-integrated.
    ///Time integrated fields are saved to datafile automatically
    SOLVE_FOR3(lnn, lnpe, lnpi);
    SOLVE_FOR(vort);

    lnn  = log(init_n + seed_n);
    lnpe = log(init_pe + seed_pe);
    lnpi = log(init_pi + seed_pi);

    lnte  = lnpe - lnn;
    lnti  = lnpi - lnn;

    if(init_zero_potential){
      vort = Laplace_perp(init_pi + seed_pi);
    }

    return 0;
}

int hesel::InitOtherFields(){
    // initialize physical and derived fields
    CalculateFieldGroupPhyd();
    if(!compact_output){
      SAVE_REPEAT5(n, pe, pi, te, ti);
    }
    //step functions for open and closed field lines and for forcing profiles
    initial_profile("sigma_open",   sigma_open);
    initial_profile("sigma_closed", sigma_closed);
    initial_profile("sigma_force",  sigma_force);
    SAVE_ONCE(sigma_open);
    SAVE_ONCE(sigma_closed);
    SAVE_ONCE(sigma_force);

    return 0;
}

int hesel::InitPhiSolver(){
    //INITIALIZE Laplace inversion
    phiSolver = Laplacian::create();
    phiSolver->setCoefA(0.);

    // initialize phi related fields
    phi = 0.; // Starting phi
    if(!compact_output){
      SAVE_REPEAT(phi);
    }
    if (invert_w_star){
        phi_star = phi;
        if(!compact_output){
          SAVE_REPEAT(phi_star);
        }
//    }else{
//        evort = Delp2(phi);
//        delpi = Delp2(pi);
//        mesh->communicate(evort, delpi);
//        SAVE_REPEAT2(evort, delpi);
    }

//    vort = 0.;// has to be zero, now set by input file

    // laplace testing
    if(test_laplace){
        vort_residual = 0.;
        if(!compact_output){
          SAVE_REPEAT(vort_residual);
        }
    }

    return 0;
}

int hesel::InitDiagnostics(){
    TRACE("Initialize diagnostics!");

    if (diagnostics) {
        // ExB drift velocity
        Calculate_uExB();
        if(!compact_output){
          SAVE_REPEAT(uExB);
        }
        // thermal diagnostics
        if (diag_thermal_para_ad){
            qe_para_advection  = 0.;
            qi_para_advection  = 0.;
            //if(!compact_output){
            //  SAVE_REPEAT2(qe_para_advection,  qi_para_advection);
            //}

            avg__qe_para_advection = 0.;
            avg__qi_para_advection = 0.;
            SAVE_REPEAT2(avg__qe_para_advection,  avg__qi_para_advection);
        }
        if (diag_thermal_para_sh){
            qe_para_conduction = 0.;
            qi_para_conduction = 0.;
            //if(!compact_output){
            //  SAVE_REPEAT2(qe_para_conduction, qi_para_conduction);
            //}

            avg__qe_para_conduction = 0.;
            avg__qi_para_conduction = 0.;
            SAVE_REPEAT2(avg__qe_para_conduction,  avg__qi_para_conduction);
        }
        if (diag_thermal_perp_neo){
            qe_perp_neoclassic = 0.;
            qi_perp_neoclassic = 0.;
            //if(!compact_output){
            //  SAVE_REPEAT2(qe_perp_neoclassic, qi_perp_neoclassic);
            //}

            avg__qe_perp_neoclassic = 0.;
            avg__qi_perp_neoclassic = 0.;
            SAVE_REPEAT2(avg__qe_perp_neoclassic,  avg__qi_perp_neoclassic);
        }
        if (diag_thermal_perp_turb){
            qe_perp_turbulence = 0.;
            qi_perp_turbulence = 0.;
            //if(!compact_output){
            //  SAVE_REPEAT2(qe_perp_turbulence, qi_perp_turbulence);
            //}

            avg__qe_perp_turbulence = 0.;
            avg__qi_perp_turbulence = 0.;
            SAVE_REPEAT2(avg__qe_perp_turbulence,  avg__qi_perp_turbulence);
        }

        // particle flux diagnostics
        if (diag_particle_flux){
            part_turb_flux  = 0.;
            part_diff_flux  = 0.;

            avg__part_turb_flux = 0.;
            avg__part_diff_flux = 0.;
            SAVE_REPEAT2(avg__part_turb_flux,  avg__part_diff_flux);
        }
    }

    return 0;
}

int hesel::InitNeutrals(){
    if (plasma_neutral_interactions){
        SOLVE_FOR3(neutrals.lnNcold,neutrals.lnNwarm,neutrals.lnNhot);
        SOLVE_FOR(neutrals.lnNhelium);
        neutrals.InitNeutrals();
    }

    return 0;
}

int hesel::InitParallel(){
    if (parallel_transport){
        SOLVE_FOR2(parallel.ue_parl, parallel.ui_parl);
        parallel.InitParallel();
    }

    return 0;
}

int hesel::InitKineticNeutrals(bool restart){
    if (kinetic_plasma_neutral_interactions){
        kinetic_neutrals.InitKineticNeutrals(restart);
    }
    return 0;
}

//******************************************************************************

int hesel::RhsEvolvingFieldsStart(){
    ddt_lnn  = 0.;
    ddt_lnpe = 0.;
    ddt_lnpi = 0.;
    ddt_vort = 0.;

    mesh->communicate(field_group_ln);
    mesh->communicate(vort);

    return 0;
}

int hesel::RhsInterchangeDynamics(){
    TRACE("Interchange dynamics");

    if (interchange_dynamics){
        // particle density
        ddt_lnn +=
                - invB*Brackets(phi,lnn)
                - C(phi)
                + C(te)
                + C(lnn)*te;

        // electron pressure
        ddt_lnpe +=
                - invB*Brackets(phi,lnpe)
                - 5./3.*C(phi)
                + 5./3.*C(te)
                + 5./3.*C(lnpe)*te;

        // ion pressure
        ddt_lnpi +=
                - invB*Brackets(phi,lnpi)
                - 5./3.*C(phi)
                - 5./3.*C(ti)
                - 5./3.*C(lnpi)*ti
                + 2./3.*C(pe+pi);

        // vorticity
        ddt_vort +=
                - Brackets(phi, vort)
                + C(pe+pi);

        // cross term of vorticity
        if (!test_vort_cross_term){
            ddt_vort +=
                    - Brackets(gradp_phi.x,gradp_pi.x)
                    - Brackets(gradp_phi.z,gradp_pi.z);
        }
    }

     return 0;
}

int hesel::RhsPerpendicularDynamics(){
    TRACE("Perpendicular dynamics");

    if (perpendicular_dynamics){
        // calculate ti_rcpte
        switch (ti_over_te){
            case 1:{
                ti_rcpte = HeselPara.tau();
                break;
            }
            case 2:{
                ti_rcpte = avg_tau; // [approx]
                break;
            }
            case 3:{
                ti_rcpte = tau; // [approx]
                break;
            }
            default:{
                throw BoutException("no support settings for ti_over_te\n");
            }
        }
        // neo-classical collisions
        switch(collisional_model_enum){
            case off:{
                break;
            }
            case simple:{
                RhsCollisionalSimple();
                break;
            }
            case standard:{
                CalculateDiffusionCoeff();
                Calculate_uR(0);
                RhsCollisionalStandard();
                break;
            }
            case full:{
                CalculateDiffusionCoeff();
                Calculate_uR(1);
                RhsCollisionalFull();
                break;
            }
            default:{
                throw BoutException("Collisional model not recognized\n");
            }
        }

        // electron-ion heat exchanging
        if (perpend_heat_exchange){
            Qresist_Rcppi = uR*(gradp_lnn + gradp_lnti);// UR*gradp_pi/pi, heat exchange term due to resistivity, J. Madsen Equation 42

            switch (qdelta_approx){
                case 0:{
                    Qdelta_Rcppe = 0.;
                    break;
                }
                case 1:{
                    Qdelta_Rcppe = 3*HeselPara.me()/HeselPara.mi()*HeselPara.nnuei()*(1-ti_rcpte);// [approx], Qdelta/pe, heat exchange term due to elastic collisions, J. Madsen Equation 26
                    break;
                }
                case 2:{
                    Qdelta_Rcppe = 3*HeselPara.me()/HeselPara.mi()*HeselPara.nnuei()*(1-ti_rcpte) * avg_n / avg_te / sqrt(avg_te); // [approx]
                    break;
                }
                case 3:{
                    Qdelta_Rcppe = 3*HeselPara.me()/HeselPara.mi()*HeselPara.nnuei()*(1-ti_rcpte) * n / te / sqrt(te); // [division]
                    break;
                }
                case 4:{
                    Qdelta_Rcppe = 3*HeselPara.me()/HeselPara.mi()*HeselPara.nnuei()*(1-ti_rcpte) * n / avg_te / sqrt(avg_te); // [approx]
                    break;
                }
                default:{
                    throw BoutException("Qdelta approximation is not recognized\n");
                }
            }

            ddt_lnpe +=
                    - 2./3. * Qresist_Rcppi* ti_rcpte
                    - 2./3. * Qdelta_Rcppe;
            ddt_lnpi +=
                    + 2./3. * Qresist_Rcppi
                    + 2./3. * Qdelta_Rcppe / ti_rcpte; // [division]
        }

        // ion viscous heating
        if (perpend_viscous_heating){
            Qviscous = 3./10. * Di*( pow(D2DX2(phi+pi)-D2DZ2(phi+pi),2)+4*pow(D2DXDZ(phi+pi),2) )/ti; // [division]
//            Qviscous = HeselPara.normEta()*( pow(D2DX2(phi+pi)-D2DZ2(phi+pi),2)+4*pow(D2DXDZ(phi+pi),2) ); // [approx], not approximate n*Ti in eta, approximate nuii
            mesh->communicate(Qviscous);

            ddt_lnpi += 2./3. * Qviscous;
        }

        // hyper diffusion
        if (perpend_hyper_diffusion){
            if (hyper_order == 4) {
                ddt_lnn +=
                        -         hyper_dc_n  * Dn       * Delp4(n)/avg_n;
                ddt_lnpe +=
                        - 2./3. * hyper_dc_pe * Dn       * Delp4(pe)/avg_n/avg_te;
                ddt_lnpi +=
                        - 2./3. * hyper_dc_pi * 5./2.*Dn * Delp4(pi)/avg_n/avg_ti;
                ddt_vort +=
                        -         hyper_dc_vort * HeselPara.normEta() * Delp4(vort);
            } else if (hyper_order == 6) {
                ddt_lnn +=
                        +         hyper_dc_n  * Dn       * Delp6(n)/avg_n;
                ddt_lnpe +=
                        + 2./3. * hyper_dc_pe * Dn       * Delp6(pe)/avg_n/avg_te;
                ddt_lnpi +=
                        + 2./3. * hyper_dc_pi * 5./2.*Dn * Delp6(pi)/avg_n/avg_ti;
                ddt_vort +=
                        +         hyper_dc_vort * HeselPara.normEta() * Delp6(vort);
            } else {
                throw BoutException("Invalid hyper differential order!");
            }

            mesh->communicate(ddt_lnn, ddt_lnpe, ddt_lnpi, ddt_vort);
        }
    }

    return 0;
}

int hesel::RhsCollisionalSimple(){
    // for testing, only diffusion is included
    TRACE("Collisional effects: simple");

    ddt_lnn +=
            + HeselPara.normDn()*Delp2(lnn);
    ddt_lnpe +=
            + 2./3.*HeselPara.normDn()*Delp2(lnpe);
    ddt_lnpi +=
            + 2./3.*2.*HeselPara.normDi()*Delp2(lnti);
    ddt_vort +=
            + HeselPara.normEta()*Delp2(vort);

    mesh->communicate(field_group_ddt);

    return 0;
}

int hesel::RhsCollisionalStandard(){
    TRACE("Collisional effects: standard");

    Deln_Rcpn   = Delp2(lnn)  + gradp_lnn *gradp_lnn;
    Delte_Rcpte = Delp2(lnte) + gradp_lnte*gradp_lnte;
    Delti_Rcpti = Delp2(lnti) + gradp_lnti*gradp_lnti;
    Gradn_Gradte_Rcppe = gradp_lnn*gradp_lnte;
    Gradn_Gradti_Rcppi = gradp_lnn*gradp_lnti;
    mesh->communicate(Deln_Rcpn, Delte_Rcpte, Delti_Rcpti);

    ddt_lnn +=
            +                  Dn* Deln_Rcpn;
    ddt_lnpe +=
            + 2./3. *          Dn*(Deln_Rcpn   + Gradn_Gradte_Rcppe)
            + 2./3. * 29./12.* De*(Delte_Rcpte + Gradn_Gradte_Rcppe);
    ddt_lnpi +=
            + 2./3. * 5./2.*   Dn*(Deln_Rcpn   + Gradn_Gradti_Rcppi)
            + 2./3. * 2. *     Di*(Delti_Rcpti + Gradn_Gradti_Rcppi);
    ddt_vort +=
            +                  HeselPara.normEta()*Delp2(vort);

    mesh->communicate(ddt_vort);

    return 0;
}

int hesel::RhsCollisionalFull(){
    TRACE("Collisional effects: full");

    Div_gammaR_Rcpn = gradp_lnn*uR + Div(uR);
    Delte_Rcpte     = Delp2(lnte) + gradp_lnte*gradp_lnte;
    Delti_Rcpti     = Delp2(lnti) + gradp_lnti*gradp_lnti;
    mesh->communicate(Div_gammaR_Rcpn);
    mesh->communicate(Delte_Rcpte, Delti_Rcpti);

    ddt_lnn +=
            -                 Div_gammaR_Rcpn;
    ddt_lnpe +=
            - 2./3. *        (Div_gammaR_Rcpn + gradp_lnte*uR)
            + 2./3. * 29./12*((Grad_perp(De)+De*gradp_lnn)*gradp_lnte + De*Delte_Rcpte);
    ddt_lnpi +=
            - 2./3. * 5./2.* (Div_gammaR_Rcpn + gradp_lnti*uR)
            + 2./3. * 2.*    ((Grad_perp(Di)+Di*gradp_lnn)*gradp_lnti + Di*Delti_Rcpti);
    ddt_vort +=
            + HeselPara.normEta()*Delp2(vort);

    mesh->communicate(ddt_lnpe, ddt_lnpi, ddt_vort);

    return 0;
}

int hesel::RhsParallelDynamics(){
    TRACE("Parallel dynamics");

    if (parallel_dynamics){
        // parallel advection damping
        RhsParallelDampAdvection();
        ddt_lnn  -=               sigma_open*DampAdvection;
        ddt_lnpe -= 2./3. * 9./2.*sigma_open*DampAdvection;
        ddt_lnpi -= 2./3. * 9./2.*sigma_open*DampAdvection;
        ddt_vort -=               sigma_open*DampAdvection*vort;

        // parallel sheath damping
        RhsParallelDampSheath();
        ddt_lnpi += 2./3. * sigma_open*DampSheath;
        ddt_vort +=         sigma_open*DampSheath;

        // parallel Spitzer-Harm conduction
        switch(parallel_conduction){
            case 0:{
                DampShe = 0.;
                DampShi = 0.;
                break;
            }
            case 1:{
                DampShe = pow(te,2)*sqrt(te)/HeselPara.normTaushe();
                DampShi = 0.;
                break;
            }
            case 2:{
                DampShe = pow(te,2)*sqrt(te)/HeselPara.normTaushe();
                DampShi = pow(ti,2)*sqrt(ti)/HeselPara.normTaushi();
                break;
            }
            default:{
                throw BoutException("Not support settings for parallel Spitzer-Harm conduction!\n");
            }
        }

        // drift wave dynamics
        RhsParallelDriftWave();
        ddt_vort -= sigma_closed*DriftWave;

        // reciprocal approximation
        switch (reciprocal_approx){
            case 1:{
                ddt_lnpe -= 2./3. * sigma_open*DampShe;
                ddt_lnpi -= 2./3. * sigma_open*DampShi;
                ddt_lnn  -=         sigma_closed*DriftWave;
                ddt_lnpe -= 2./3. * 3.21*sigma_closed*DriftWave*avg_te;
                ddt_lnpi -= 2./3. * sigma_closed*DriftWave*avg_n*avg_ti;
                break;
            }
            case 2:{
                ddt_lnpe -= 2./3. * sigma_open*DampShe/avg_n; // [approx]
                ddt_lnpi -= 2./3. * sigma_open*DampShi/avg_n; // [approx]
                ddt_lnn  -=         sigma_closed*DriftWave/avg_n; // [approx], n ~ avg_n
                ddt_lnpe -= 2./3. * 3.21*sigma_closed*DriftWave/avg_n; // [approx], pe ~ avg_n*avg_te
                ddt_lnpi -= 2./3. * sigma_closed*DriftWave; // [approx]
                break;
            }
            case 3:{
                ddt_lnpe -= 2./3. * sigma_open*DampShe/n; // [division]
                ddt_lnpi -= 2./3. * sigma_open*DampShi/n; // [division]
                ddt_lnn  -=         sigma_closed*DriftWave/n; // [division]
                ddt_lnpe -= 2./3. * 3.21*sigma_closed*DriftWave*avg_te/pe; // [division]
                ddt_lnpi -= 2./3. * sigma_closed*DriftWave*avg_n*avg_ti/pi; // [division]
                break;
            }
            default:{
                throw BoutException("Not support settings for reciprocal_approx!\n");
            }
        }
    }

    return 0;
}

int hesel::RhsParallelDampSheath(){
    TRACE("Parallel sheath damping");

    switch (parallel_sheath_damping){
        case 0:{ // no sheath damping
            DampSheath = 0.;
            break;
        }
        case 1:{
            DampSheath =        1. / HeselPara.normLc()*(1-exp(HeselPara.bohm_potential()-avg_phi/avg_te));// Anders paper Equation 8, and J. Madsen Equation 88
            break;
        }
        case 2:{
            DampSheath = avg_csHot / HeselPara.normLc()*(1-exp(HeselPara.bohm_potential()-avg_phi/avg_te));// Anders paper Equation 8, and J. Madsen Equation 88
            break;
        }
        case 3:{
            /* [division] */
            DampSheath =     csHot / HeselPara.normLc()*(1-exp(HeselPara.bohm_potential()-phi/te));// Anders paper Equation 8
            break;
        }
        default:{
            throw BoutException("Not recognized sheath damping options!\n");
        }
    }

    return 0;
}

int hesel::RhsParallelDampAdvection(){
    TRACE("Parallel advection");

    switch (parallel_advection_damping){
        case 0:{
            DampAdvection = 0.;
            break;
        }
        case 1:{
            DampAdvection =        1. / HeselPara.normTaun();
            break;
        }
        case 2:{
            DampAdvection = avg_csHot / HeselPara.normTaun();
            break;
        }
        case 3:{
            DampAdvection =     csHot / HeselPara.normTaun();
            break;
        }
        default:{
            throw BoutException("Not recognized advection damping options!\n");
        }
    }

    return 0;
}

int hesel::RhsParallelDriftWave(){
    TRACE("Drift wave dynamics");

    CalculateFieldGroupPert();

    switch (parallel_drift_wave){
        case 0:{
            DriftWave = 0.;
            break;
        }
        case 1:{
            DriftWave = (pert_te + pert_n*(avg_te/avg_n) - pert_phi)/HeselPara.normTaudw();
            break;
        }
        case 2:{
            DriftWave = (pert_te + pert_n*(avg_te/avg_n) - pert_phi)/HeselPara.normTaudw() * avg_te * sqrt(avg_te);
            break;
        }
        case 3:{
            DriftWave = (pert_te + pert_n*(avg_te/avg_n) - pert_phi)/HeselPara.normTaudw() * te * sqrt(te); // the n in nuei cancels out
            break;
        }
        default:{
            throw BoutException("ERROR: Invalid input for parallel_drift_wave!\n");
        }
    }

    return 0;
}

int hesel::RhsDiagnostics(BoutReal UNUSED(t)){
    TRACE("HESEL Diagnostics");
		/*std::cout << "Electron density size." << " x: " << n.getNx() << " y: " << n.getNy() << " z: " << n.getNz() << std::endl;
		std::cout << "Whatever parameter " << mesh->getCoordinates()->dx(1,0) << std::endl;*/
    if (diagnostics) {
        // ExB drift velocity
        Calculate_uExB();

        // thermal diagnostics
        if (diag_thermal_para_ad){
            qe_para_advection  = 9./2.*DampAdvection*pe * HeselPara.normLb();
            qi_para_advection  = 9./2.*DampAdvection*pi * HeselPara.normLb();
//            sigma_adjust(qe_para_advection, 1);
//            sigma_adjust(qi_para_advection, 1);
        }
        if (diag_thermal_para_sh){
            qe_para_conduction =       DampShe*te       * HeselPara.normLb();
            qi_para_conduction =       DampShi*ti       * HeselPara.normLb();
//            sigma_adjust(qe_para_conduction, 1);
//            sigma_adjust(qi_para_conduction, 1);
        }
        if (diag_thermal_perp_neo){
            switch(collisional_model_enum){
            case off:{
                qe_perp_neoclassic = 0.;
                qi_perp_neoclassic = 0.;
                break;
            }
            case simple:{
                qe_perp_neoclassic = - HeselPara.normDn()*DDX(pe);
                qi_perp_neoclassic = - 2.*HeselPara.normDn()*n*DDX(ti);
                break;
            }
            case standard:{
                qe_perp_neoclassic = -       Dn*te*DDX(n) - 29./12.*De*n*DDX(te);
                qi_perp_neoclassic = - 5./2.*Dn*ti*DDX(n) -      2.*Di*n*DDX(ti);
                break;
            }
            case full:{
                qe_perp_neoclassic =              pe*uR.x - 29./12.*De*n*DDX(te);
                qi_perp_neoclassic =        5./2.*pi*uR.x -      2.*Di*n*DDX(ti);
                break;
            }
            default:{
                throw BoutException("Collisional model not recognized\n");
            }
            }
            mesh->communicate(qe_perp_neoclassic, qi_perp_neoclassic);
        }
        if (diag_thermal_perp_turb){
            qe_perp_turbulence = 3./2.*pe*uExB.x;
            qi_perp_turbulence = 3./2.*pi*uExB.x;
        }

        // particle flux diagnostics
        if (diag_particle_flux){
            part_turb_flux  = n*uExB.x;

            switch(collisional_model_enum){
            case off:{
                part_diff_flux  = 0.;
                break;
            }
            case simple:{
                part_diff_flux = -HeselPara.normDn()*DDX(n);
                break;
            }
            case standard:{
                part_diff_flux = -Dn*DDX(n);
                break;
            }
            case full:{
                part_diff_flux = n*uR.x;
                break;
            }
            default:{
                throw BoutException("Collisional model not recognized\n");
            }
            }
        }
    }

    return 0;
}

int hesel::RhsPlasmaNeutralInteraction(BoutReal t){
    /* If true, neutral sources are included in plasma transport equations.
     * If false, the plasma fields evolve as if there were no neutrals.
     */
    TRACE("Neutrals");
    source_n    = plasma_neutral_interactions & 1;
    source_pe   = plasma_neutral_interactions & 2;
    source_pi   = plasma_neutral_interactions & 4;
    source_vort = plasma_neutral_interactions & 8;

    if(plasma_neutral_interactions){
        neutrals.RhsNeutrals(t);
        mesh->communicate(neutrals.uSi);

	/* nHESEL neutral source terms as in alec thesis eqns. (2.128) - (2.131) */
        if (source_n)   {//std::cout << "S_n = " << SumField(neutrals.Sn/n) << std::endl;
												 //std::cout << "S_pe = " << 2./3*SumField(neutrals.Spe/pe) << std::endl;
												 //std::cout << "S_pi = " << 2./3*SumField(neutrals.Spi/pi) << std::endl;
												 //std::cout << "n = " << SumField(n) << std::endl;
												 ddt_lnn  += neutrals.Sn/n;}
        if (source_pe)  {//std::cout << "S_pe = " << neutrals.Spe(20, 0, 60) << std::endl;
												 ddt_lnpe += 2./3.*neutrals.Spe/pe;}
        if (source_pi)  {ddt_lnpi += 2./3.*neutrals.Spi/pi - Div(pi*neutrals.uSi)/pi - 2./3.*Div(neutrals.uSi);}
        if (source_vort){ddt_vort -= Div(n*neutrals.uSi);}
    }

    return 0;
}

/* If true, neutral sources are included in plasma transport equations.
 * If false, the plasma fields evolve as if there were no neutrals.
 */
 /* nHESEL neutral source terms as in alec thesis eqns. (2.128) - (2.131) */
 /*
int hesel::RhsKineticPlasmaNeutralInteraction(BoutReal t){

    TRACE("Kinetic Neutrals");
		if (kinetic_plasma_neutral_interactions){
			kinetic_neutrals.RhsCommunicateRoutine(t);


	    ddt_lnn  += kinetic_neutrals.Sn/n;
	    ddt_lnpe += 2./3.*kinetic_neutrals.Spe/pe;
	    ddt_lnpi += 2./3.*kinetic_neutrals.Spi/pi;
	    ddt_vort -= 0;
		}
    return 0;
}*/

int hesel::RhsKineticPlasmaNeutralInteractionInit(BoutReal t){
    /* If true, neutral sources are included in plasma transport equations.
     * If false, the plasma fields evolve as if there were no neutrals.
     */
    TRACE("Kinetic Neutrals Init");
		if (kinetic_plasma_neutral_interactions){
			kinetic_neutrals.RhsSend(t);
		}
    return 0;
}

int hesel::RhsKineticPlasmaNeutralInteractionFinal(){
    /* If true, neutral sources are included in plasma transport equations.
     * If false, the plasma fields evolve as if there were no neutrals.
     */
    TRACE("Kinetic Neutrals Final");
		if (kinetic_plasma_neutral_interactions){
			kinetic_neutrals.RhsReceive();

		/* nHESEL neutral source terms as in alec thesis eqns. (2.128) - (2.131) */
	    ddt_lnn  += kinetic_neutrals.Sn/n;
	    ddt_lnpe += 2./3.*kinetic_neutrals.Spe/pe;
			ddt_lnpi += 2./3.*kinetic_neutrals.Spi/pi - Div(pi*kinetic_neutrals.uSi)/pi - 2./3.*Div(kinetic_neutrals.uSi)
	    	      - 2./3.*(kinetic_neutrals.u0_x_ion*kinetic_neutrals.Sux + kinetic_neutrals.u0_z_ion*kinetic_neutrals.Suz)/pi
	    	      + 1./3.*kinetic_neutrals.Sn*(kinetic_neutrals.u0_x_ion*kinetic_neutrals.u0_x_ion + kinetic_neutrals.u0_z_ion*kinetic_neutrals.u0_z_ion)/pi
                      + 2./3.*Div(n*kinetic_neutrals.uSi);
	    ddt_vort += Div(n*kinetic_neutrals.uSi);
		}
    return 0;
}

int hesel::RhsParallelTransport(BoutReal t){
    /* Full parallel evolution of plasma fields
     */
    TRACE("Parallel transport");
    if(parallel_transport){
        parallel.RhsParallel(t);

        ddt_lnn  += parallel.n_terms/n;
        ddt_vort += parallel.vort_terms;
        ddt_lnpe += parallel.pe_terms/pe;
        ddt_lnpi += parallel.pi_terms/pi;
    }

    return 0;
}

int hesel::RhsRampInitProfiles(BoutReal t){
    TRACE("Ramp up initial profiles");

    if (rampup_init_prof && t <= rampup_init_time){
        BoutReal r = t/rampup_init_time;
	std::string bnd_in;
        if(rampup_init_n) {
            init_n   = init_n_fake + (init_n0-init_n_fake)*r;
            n_inner  = init_n (0,0);
            bnd_in   = "bndry_xin=dirichlet_o2(" + std::to_string(log(n_inner)) + ")";
            lnn.setBoundary(bnd_in);
//            HeselPara.n(init_n(lcfs_ind, 0));
        }
        if(rampup_init_pe){
            init_pe  = init_pe_fake + (init_pe0-init_pe_fake)*r;
            init_te  = init_pe / init_n;
            pe_inner = init_pe(0,0);
            bnd_in   = "bndry_xin=dirichlet_o2(" + std::to_string(log(pe_inner)) + ")";
            lnpe.setBoundary(bnd_in);
//            HeselPara.Te(init_te(lcfs_ind, 0));
        }
        if(rampup_init_pi){
            init_pi  = init_pi_fake + (init_pi0-init_pi_fake)*r;
            init_ti  = init_pi / init_n;
            pi_inner = init_pi(0,0);
            bnd_in   = "bndry_xin=dirichlet_o2(" + std::to_string(log(pi_inner)) + ")";
            lnpi.setBoundary(bnd_in);
//            HeselPara.Ti(init_ti(lcfs_ind, 0));
        }
    }

    return 0;
}

int hesel::RhsPlasmaFieldsForce(){
    // TO DO: avoid using explicit reciprocal. However, it act in force region,
    // shouldn't exist very small value
    if (force_profiles){
        ddt_lnn  += sigma_force*(init_n /n  - 1)/force_time;
        ddt_lnpe += sigma_force*(init_pe/pe - 1)/force_time;
        ddt_lnpi += sigma_force*(init_pi/pi - 1)/force_time;
    }

    return 0;
}

int hesel::RhsPlasmaFieldsFloor(){
    //keep field above given level
    if(floor_profiles){
        for (auto i:vort.getRegion(RGN_NOBNDRY)){
            if(n[i]  < floor_n)///change these if's to sigmoids!!
                ddt_lnn[i]  += (floor_n /n[i]  - 1)/floor_time;
            if(pe[i] < floor_pe)
                ddt_lnpe[i] += (floor_pe/pe[i] - 1)/floor_time;
            if(pi[i] < floor_pi)
                ddt_lnpi[i] += (floor_pi/pi[i] - 1)/floor_time;
        }
    }

    return 0;
}

int hesel::RhsPlasmaFieldsStatic(){
    /* If true, transport of plasma is deactivated. Use for initialization of
     * neutral field to existing plasma run.
     */
    if (static_plasma) {
        if (static_n)   {ddt_lnn  = 0.;}
        if (static_pe)  {ddt_lnpe = 0.;}
        if (static_pi)  {ddt_lnpi = 0.;}
        if (static_vort){ddt_vort = 0.;}
        if (static_te){ddt_lnpe = ddt_lnn;}
        if (static_ti){ddt_lnpi = ddt_lnn;}
    }

    return 0;
}

int hesel::RhsRampTiReference(BoutReal t){
    TRACE("Ramp up Ti0");
    if(rampup_ti0){
        HeselPara.Ti(Ti0 + (rampup_ti0_timax - Ti0)*sin(0.5*PI*t/rampup_ti0_time));
    }

    return 0;
}

int hesel::RhsTestBlob(){
    if (test_blob){
        static_te = 1;
        static_ti = 1;

        if (!test_blob_use_gen_vort){
            phi = phiSolver->solve(vort, phi);
        }

        mesh->communicate(phi);
    }

    return 0;
}

int hesel::RhsEvolvingFieldsEnd(){
    mesh->communicate(field_group_ddt);

    ddt(lnn)  = ddt_lnn;
    ddt(lnpe) = ddt_lnpe;
    ddt(lnpi) = ddt_lnpi;
    ddt(vort) = ddt_vort;

    return 0;
}

//******************************************************************************

int hesel::CalculatePhi(){
    if(invert_w_star){//invert w^* = w + \nabla^2 pi
        //boundary condition for vort is a combination of boundary condition for phi and pi
        phi_star = phiSolver->solve(vort, phi_star);
        mesh->communicate(phi_star);
        phi = phi_star - pi + pi_inner;
    }
    else{//invert w
        delpi = Delp2(pi);
        mesh->communicate(delpi);

        //ExB vorticity
        evort = vort - delpi;

        // Solve for potential
        phi = phiSolver->solve(evort, phi);
        mesh->communicate(phi);
    }

    if (test_laplace){
        vort_residual = vort - Delp2(pi) - Delp2(phi);
        mesh->communicate(vort_residual);
    }

    return 0;
}

int hesel::CalculateFieldGroupPhyd(){
    // physical and derived group
    lnte  = lnpe - lnn;
    lnti  = lnpi - lnn;
    n     = exp(lnn );
    pe    = exp(lnpe);
    pi    = exp(lnpi);
    te    = exp(lnte); // avoid using pe/n
    ti    = exp(lnti); // avoid using pi/n
    csHot = sqrt(ti + te);
    tau   = exp(lnti - lnte);

    return 0;
}

int hesel::CalculateFieldGroupAvg(){
    // average field group
    avg_n     = ZAvg(n);
    avg_te    = ZAvg(te);
    avg_ti    = ZAvg(ti);
    avg_csHot = ZAvg(csHot);
    avg_phi   = ZAvg(phi);
    avg_tau   = ZAvg(tau);
//    avg_tau   = avg_ti/avg_te;

    if(diagnostics){
        if (diag_thermal_para_ad){
            avg__qe_para_advection = ZAvg(qe_para_advection);
            avg__qi_para_advection = ZAvg(qi_para_advection);
        }
        if (diag_thermal_para_sh){
            avg__qe_para_conduction = ZAvg(qe_para_conduction);
            avg__qi_para_conduction = ZAvg(qi_para_conduction);
        }
        if (diag_thermal_perp_neo){
            avg__qe_perp_neoclassic = ZAvg(qe_perp_neoclassic);
            avg__qi_perp_neoclassic = ZAvg(qi_perp_neoclassic);
        }
        if (diag_thermal_perp_turb){
            avg__qe_perp_turbulence = ZAvg(qe_perp_turbulence);
            avg__qi_perp_turbulence = ZAvg(qi_perp_turbulence);
        }
        if (diag_thermal_perp_turb){
            avg__part_turb_flux = ZAvg(part_turb_flux);
            avg__part_diff_flux = ZAvg(part_diff_flux);
        }
    }

    return 0;
}

int hesel::CalculateFieldGroupGradp(){
    // vector field group, only call after phi has been calculated!
    gradp_phi   = Grad_perp(phi);
    gradp_pi    = Grad_perp(pi);
    gradp_lnn   = Grad_perp(lnn);
    gradp_lnte  = Grad_perp(lnte);
    gradp_lnti  = Grad_perp(lnti);

    mesh->communicate(field_group_gradp);

    return 0;
}

int hesel::CalculateFieldGroupPert(){
    pert_n   = n   - avg_n;
    pert_te  = te  - avg_te;
    pert_phi = phi - avg_phi;

    return 0;
}

int hesel::CalculateDiffusionCoeff(){
    switch (diffusion_coeff){
        case 1:{
            De = HeselPara.normDe(); // [approx]
            Di = HeselPara.normDi(); // [approx]
            break;
        }
        case 2:{
            De = HeselPara.normDe()*avg_n/sqrt(avg_te)/pow(B,2); // [approx]
            Di = HeselPara.normDi()*avg_n/sqrt(avg_ti)/pow(B,2); // [approx]
            break;
        }
        case 3:{
            De = HeselPara.normDe()*n/sqrt(te)/pow(B,2); // [division]
            Di = HeselPara.normDi()*n/sqrt(ti)/pow(B,2); // [division]
            break;
        }
        case 4:{
            De = HeselPara.normDe()*avg_n/sqrt(te)/pow(B,2); // [division] [approx]
            Di = HeselPara.normDi()*avg_n/sqrt(ti)/pow(B,2); // [division] [approx]
            break;
        }
        case 5:{
            De = HeselPara.normDe()*n/sqrt(avg_te)/pow(B,2); // [approx]
            Di = HeselPara.normDi()*n/sqrt(avg_ti)/pow(B,2); // [approx]
            break;
        }
        case 6:{
            De = HeselPara.normDe()/sqrt(avg_te)/pow(B,2); // [approx]
            Di = HeselPara.normDi()/sqrt(avg_ti)/pow(B,2); // [approx]
            break;
        }
        default:{
            throw BoutException("not support diffusion_coeff!\n");
        }
    }

    De *= Z_eff;
    Di *= Z_eff;

    Dn = De*(1+ti_rcpte);

    return 0;
}

int hesel::Calculate_uR(int c){
    if (c == 1){
//        uR = -De/te*((te+ti)*gradp_lnn+Grad_perp(ti)-0.5*Grad_perp(te)); // [division]
        uR = -De*((1+ti_rcpte)*gradp_lnn+gradp_lnti*ti_rcpte-0.5*gradp_lnte);
    } else if (c == 0){
        uR = -Dn*gradp_lnn;
    } else {
        throw BoutException("Wrong input parameters!\n");
    }

    return 0;
}

int hesel::Calculate_uExB(){
    uExB = invB*b0xGrad(phi);
    return 0;
}

//******************************************************************************
// Bracket and C operators are decomposed to X-Z components and depend on coordinate
// system. By default BOUT++ slab geometry is left-handed coordinate system and
// HESEL equations are decomposed in right-handed coordinate system. One should be
// careful the sign of the operators here. Since only these two operators are
// influenced during reduction process. One should be careful of the sign and make
// them consistent.


Field3D hesel::Brackets(const Field3D &f, const Field3D &g){
    // we use BRACKET_ARAKAWA, it decomposes vector with X-Z components in
    // left-hand coordinate system (BOUT++ default). since HESEL is decomposed
    // in right-hand system, we add a minus in font.
    Field3D res;

    if (right_handed_coord){
        res = -bracket(f, g, bracket_method_enum);
    }else{
        res =  bracket(f, g, bracket_method_enum);
    }

    mesh->communicate(res);

    return res;
}

Field3D hesel::C(const Field3D &f) {
    // Jens Madsen Equation 78, decomposed in right-hand coordinate system, has a
    // minus in front. By default, BOUT++ is a left-hand coordinate system, if
    // one want to decompose it in this coordinate, there is no minus in front.
    Field3D res;

    if (right_handed_coord){
//        res = -invB*HeselPara.rhos()/HeselPara.R() * DDZ(f);
//        res = -2*invB*HeselPara.rhos()*invR * DDZ(f);
        res = -curvature_coeff*HeselPara.rhos()/(HeselPara.R()+HeselPara.a()) * DDZ(f); // B*R=B0*(Rmajor+Rminor)
    }else{
//        res =  invB*HeselPara.rhos()/HeselPara.R() * DDZ(f);
//        res =  2*invB*HeselPara.rhos()*invR * DDZ(f);
        res =  curvature_coeff*HeselPara.rhos()/(HeselPara.R()+HeselPara.a()) * DDZ(f); // B*R=B0*(Rmajor+Rminor)
    }

    mesh->communicate(res);

    return res;
}

Field2D hesel::ZAvg(Field3D f) {
    Field2D res;
    res = 0.;

    for (auto i:f){
        res(i.x(), i.y()) += f(i.x(), i.y(), i.z())/mesh->LocalNz;
    }

    return res;
}

Field3D hesel::Delp4(const Field3D &var){
    // Simple implementation of 4th order perpendicular Laplacian
    // Copy from BOUT-dev/examples/hasegawa-wakatani
    Field3D tmp;

    tmp = Delp2(var);
    mesh->communicate(tmp);
    tmp.applyBoundary("neumann");

    return Delp2(tmp);
}

Field3D hesel::Delp6(const Field3D &var){
    Field3D tmp;

    tmp = Delp4(var);
    mesh->communicate(tmp);
    tmp.applyBoundary("neumann");

    return Delp2(tmp);
}

std::string hesel::bool2str(bool b){
    return b ? "on" : "off";
}

int hesel::bool2int(bool b){
    return b ? 1 : 0;
}

Field2D hesel::LinearSlope(BoutReal ls_intercept, BoutReal ls_turning_x, BoutReal ls_turning_y){
    Field2D res=0.;
    BoutReal x, y, slope, dx;
    int i, j;

    slope = (ls_turning_y - ls_intercept) / ls_turning_x;

    x = 0.;
    y = ls_intercept;
    for (i=mesh->xstart-1; i >= 0; i--){
        dx = mesh->getCoordinates()->dx(i,0);
        x -= dx;
        y -= slope * dx;
    }

    dx = mesh->getCoordinates()->dx(0,0);
    x -= dx;
    y -= slope * dx;
    for (i=0; i<mesh->GlobalNx; i++){
        dx = mesh->getCoordinates()->dx(i,0);
        x += dx;
        y += slope * dx;
        for (j=0; j<mesh->GlobalNy; j++){
            if (x > ls_turning_x){
                res(i,j) = ls_turning_y;
            } else {
                res(i,j) = y;
            }
//            output << '(' << x << ',' << res(i,j) << ')' << endl;  // test
        }
    }

    return res;
}

Vector3D hesel::b0xGrad(const Field3D &var){
    Vector3D tmp;
    tmp = Grad_perp(var);
    mesh->communicate(tmp);

    Vector3D res;
    res.y = 0.;
    if (right_handed_coord){
        res.x = -tmp.z;
        res.z = tmp.x;
    }else{
        res.x = tmp.z;
        res.z = -tmp.x;
    }

    return res;
}

int hesel::sigma_adjust(Field3D &var, int region){
    if (region == 1){// core region
        for (auto i:var){
            if (i.x() < lcfs_ind){
                var[i] = 0.;
            }
        }
    }else if (region == 2){// edge region
        for (auto i:var){
            if (i.x() > lcfs_ind){
                var[i] = 0.;
            }
        }
    }

    return 0;
}
//******************************************************************************


int main(int argc, char **argv) {
	/* Split world communicator */
	MPI_Comm sub_comm;
	int flag, world_rank, app_id;
	void* appnum;
	BoutComm* boutcomm = BoutComm::getInstance();

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &appnum, &flag);
	app_id = *(int*)appnum;
	MPI_Comm_split(MPI_COMM_WORLD, app_id, world_rank, &sub_comm);
	/*Set the subcommunicator of the c++ side as the communicator in boutcomm*/
	boutcomm -> setComm(sub_comm);
	/* Init Bout and hesel */
  BoutInitialise(argc, argv); // Initialise BOUT++
  hesel *model = new hesel(); // Create a model
  Solver *solver = Solver::create(); // Create a solver
  solver->setModel(model); // Specify the model to solve
  auto bout_monitor = bout::utils::make_unique<BoutMonitor>();
  solver->addMonitor(bout_monitor.get(), Solver::BACK);
  solver->solve(); // Run the solver
	//std::cout << "############rhs_wall_time_not_including_neutrals = " << model->rhs_wall_time_not_including_neutrals << std::endl;
  delete model;
  delete solver;
	MPI_Comm_free(&sub_comm);
  BoutFinalise(); // Finished with BOUT++
	MPI_Finalize();
  return 0;
}
