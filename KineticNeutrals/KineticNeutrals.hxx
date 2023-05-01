#include <bout.hxx>
#include <boutcomm.hxx>
#include <derivs.hxx>
#include "../HeselParameters/HeselParameters.hxx"

class KineticNeutrals{
public:

	KineticNeutrals(
    const HeselParameters &i_hesel_para_,
    const Field3D &i_n,
    const Field3D &i_te,
    const Field3D &i_ti,
		const Field3D &i_phi,
		const Field2D &i_B
  );

  ~KineticNeutrals();

  int InitKineticNeutrals(bool restart);
	int RhsSend(BoutReal t);
	int RhsReceive();

	Field3D u0_x_ion, u0_z_ion, u0_x_electron, u0_z_electron;
	//Electron sources
	Field3D Sn, Spe;
	//Ion sources
	Field3D Sux, Suz, Spi;
	Field3D u_E_x, u_E_z;
	Vector3D uSi;                   // Leading order ion drifts due to neutrals

private:

	const HeselParameters &hesel_para_; //  Object holding HESEL collisional parameters
	const Field3D &n;                   //  Plasma density
	const Field3D &te;                  //  Ion temperature
	const Field3D &ti;                  //  Electron pressure
	const Field3D &phi;                  //  Electric potential
	const Field2D &B;                  //  Magnetic field

	int InitIntercommunicator();
	int Mpi_receive(double* buffer, Field3D buffer_field, int root);
	int Mpi_send(double* buffer, Field3D buffer_field);
	int Allocate_buffers();
	int Get_ind_3D(int x, int y, int z);
	int ReadKineticNeutralParams();
	int Calculate_ion_fluid_speed();
	int Rhs_calc_velocity_sources();
	int PrintField(Field3D f);

	//Variables supporting MPI intercommunicator
  MPI_Comm intercomm, sub_comm;
  BoutReal* send_buf;
  BoutReal* recv_buf;
  BoutReal oci, rhos;
	BoutComm* boutcomm;
	int local_rank, app_size, python_size;
  //The following variables hold the size of each dimension per cpu not including guard cells.
  int loc_proc_nx, loc_proc_ny, loc_proc_nz, N_per_proc;
  //Number of gurd_cells
  int mxg;
  bool run_flag{1}, step_kinetic{0};
  BoutReal step_starttime{0.0};
  BoutReal dt_min, t_end;

};
