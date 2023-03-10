#include "KineticNeutrals.hxx"


KineticNeutrals::KineticNeutrals(
  const HeselParameters &i_hesel_para_,
  const Field3D &i_n,
  const Field3D &i_te,
  const Field3D &i_ti,
  const Field3D &i_phi,
  const Field2D &i_B
) :
hesel_para_(i_hesel_para_),
n(i_n),
te(i_te),
ti(i_ti),
phi(i_phi),
B(i_B)
{}

/// ****** Destructor ******
KineticNeutrals::~KineticNeutrals(){
	delete [] recv_buf;
	delete [] send_buf;
}

int KineticNeutrals::InitKineticNeutrals(bool restart){
		ReadKineticNeutralParams();
		//Init source fields
		Sn = 0, Spe = 0, Sux = 0, Suz = 0, Spi = 0;
    u_perp0_x = 0, u_perp0_z = 0;
    uSi.x = 0;
    uSi.y = 0;
    uSi.z = 0;
    //uSi.setBoundary("dirichlet_o2(0.)");
    oci = hesel_para_.oci();
    rhos = hesel_para_.rhos();
		//Allocate Buffer
		Allocate_buffers();
		InitIntercommunicator();
		//Make the initial communication.
		//This will give the initial fields to the kinetic model which will run
		//for an initial period to reach an equilibrium. Then source terms for the
		//first timestep will be returned.
		if (local_rank == 0){
      MPI_Bcast(&oci, 1, MPI_DOUBLE, MPI_ROOT, intercomm);
			MPI_Bcast(&rhos, 1, MPI_DOUBLE, MPI_ROOT, intercomm);
		}
    Mpi_send(send_buf, n);
    Mpi_send(send_buf, te);
    Mpi_send(send_buf, ti);
    if (restart){
      MPI_Bcast(&step_starttime, 1, MPI_DOUBLE, 0, intercomm);
      t_end += step_starttime;
    }
    Mpi_receive(recv_buf, Sn);
		Mpi_receive(recv_buf, Spe);
    Mpi_receive(recv_buf, Sux);
    Mpi_receive(recv_buf, Suz);
		Mpi_receive(recv_buf, Spi);
    //Monitor fields
    SAVE_REPEAT3(Sn,Spe,Spi);
  return 0;
}

int KineticNeutrals::ReadKineticNeutralParams(){
    Options* root_options = Options::getRoot();
    root_options->get("mxg", mxg, 1);
    root_options->get("t_end", t_end, 0);
		Options* kinetic_neutral_options = Options::getRoot()->getSection("kinetic_neutrals");
		OPTION(kinetic_neutral_options, dt_min, 0.0);
    return 0;
}

int KineticNeutrals::InitIntercommunicator(){
    sub_comm = BoutComm::get();
    MPI_Comm_size(sub_comm, &app_size);
    MPI_Intercomm_create(sub_comm, 0, MPI_COMM_WORLD, app_size, 1, &intercomm);
    MPI_Comm_rank(intercomm, &local_rank);
    return 0;
}

//Allocating buffers corresponding to the field sizes in each processor.
int KineticNeutrals::Allocate_buffers(){
	loc_proc_nx = Sn.getNx()-2*mxg;
	loc_proc_ny = Sn.getNy();
	loc_proc_nz = Sn.getNz();
	//std::cout << "############NX, NY, NZ: " << loc_proc_nx << ", " << loc_proc_ny << ", " << loc_proc_nz << std::endl;
	N_per_proc = loc_proc_nx*loc_proc_ny*loc_proc_nz;
	recv_buf = new BoutReal[N_per_proc];
	send_buf = new BoutReal[N_per_proc];
  return 0;
}

int KineticNeutrals::SendFieldsReceiveSources(){
		Mpi_send(send_buf, n);
		Mpi_send(send_buf, te);
		Mpi_send(send_buf, ti);
		Mpi_receive(recv_buf, Sn);
		Mpi_receive(recv_buf, Spe);
    Mpi_receive(recv_buf, Sux);
    Mpi_receive(recv_buf, Suz);
		Mpi_receive(recv_buf, Spi);
		return 0;
}

int KineticNeutrals::Rhs_calc_velocity_sources(){
  u_perp0_x  = -1/B*DDZ(phi);  // ExB drift, x-component
  u_perp0_z  =  1/B*DDX(phi);  //            z-component

  uSi.x =  Suz/n - u_perp0_z*Sn/n;
  uSi.z = -Sux/n + u_perp0_x*Sn/n;
  //uSi.applyBoundary();
  return 0;
}

int KineticNeutrals::RhsSend(BoutReal t){
  if ((step_starttime+dt_min < t) or (t_end <= t)){
    if (run_flag){
      step_starttime = t;
      step_kinetic = 1;
      if (local_rank == 0){
        MPI_Bcast(&t, 1, MPI_DOUBLE, MPI_ROOT, intercomm);
      }
      Mpi_send(send_buf, n);
  		Mpi_send(send_buf, te);
  		Mpi_send(send_buf, ti);
    }
    if (t > t_end){
      run_flag = 0;
    }
  }
  return 0;
}

int KineticNeutrals::RhsReceive(){
  if (step_kinetic){
    step_kinetic = 0;
    Mpi_receive(recv_buf, Sn);
		Mpi_receive(recv_buf, Spe);
    Mpi_receive(recv_buf, Sux);
    Mpi_receive(recv_buf, Suz);
		Mpi_receive(recv_buf, Spi);
    Rhs_calc_velocity_sources();
  }
  return 0;
}

//Receive the input data from the remote group of the intercommunicator, and read the data into the relevant field.
int KineticNeutrals:: Mpi_receive(double* buffer, Field3D input_field){
	MPI_Scatter(NULL, 0, MPI_DOUBLE, buffer, N_per_proc, MPI_DOUBLE, 0, intercomm);
	for(int i = 0; i < loc_proc_nx; i++){
		for(int j = 0; j < loc_proc_ny; j++){
			for(int k = 0; k < loc_proc_nz; k++){
				//mxg guard cells are places in the x-dimension. Dont put any data into these cells.
				input_field(i+mxg, j, k) = buffer[Get_ind_3D(i, j, k)];
				//std::cout << "My field at " << i << ", " << j << ", " << k << " is " << buffer_field(i, j, k) << std::endl;
			}
		}
	}
  return 0;
}

//Send the output data to the remote group of the intercommunicator.
int KineticNeutrals::Mpi_send(double* buffer, Field3D output_field){
	for(int i = 0; i < loc_proc_nx; i++){
		for(int j = 0; j < loc_proc_ny; j++){
			for(int k = 0; k < loc_proc_nz; k++){
				//mxg guard cells are placed in the x-dimension. Dont read any data into these cells.
				 buffer[Get_ind_3D(i, j, k)] = output_field(i+mxg, j, k);
				 //std::cout << "My field at " << i << ", " << j << ", " << k << " is " << buffer_field(i, j, k) << std::endl;
			}
		}
	}
	MPI_Gather(buffer, N_per_proc, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, intercomm);
  return 0;
}

//Return the index of a flattened array of dimension nx, ny, nz where nx is the
//outer dimension and z is the inner dimension.
int KineticNeutrals::Get_ind_3D(int x_ind, int y_ind, int z_ind){
	return x_ind*loc_proc_ny*loc_proc_nz + y_ind*loc_proc_nz + z_ind;
}

int KineticNeutrals::PrintField(Field3D f){
  int nx = loc_proc_nx + 2*mxg;
  std::cout << "[";
    for (int i = 0; i < nx; i++){
      std::cout << "[";
      for (int j = 0; j < loc_proc_nz; j++){
        std::cout << f(i, 0, j) << " ";
        if (j == loc_proc_nz-1){std::cout << "]" << std::endl;}
      }
    }
  std::cout << "]" << std::endl;
  return 0;
}
