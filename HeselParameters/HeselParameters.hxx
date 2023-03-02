/*
 * HeselParameters.hxx
 *
 *  Created on: Oct 22, 2014
 *      Author: yolen
 *
 *      This class store and calculate the diffusivity from representative parameters such
 *      as temperature, particle density, safety factor. Usually this is taken at the LCFS
 *
 *		Notation is that of the manuscript submitted to PRL plus the notation use in the HESEL writeup
 *		Collision frequencies follow Sigmar Helander
 */

#ifndef HESELPARAMETERS_HXX_
#define HESELPARAMETERS_HXX_

#include <bout.hxx>
#include <bout/constants.hxx>
#include <math.h>


//useful constants
const BoutReal e    = 1.60E-19;//unit charge
const BoutReal kb   = 1.38E-23;//Boltzmann constant
const BoutReal epso = 8.85E-12;// Permittivity of free space
const BoutReal muo  = 4.0*PI*1E-7;// Permeability of free space
const BoutReal Me  = 9.1093816e-31;      // Electron mass
const BoutReal Mp  = 1.67262158e-27; // Proton mass


class BoutRealVar {
public:
    BoutRealVar():value_(0),valset_(false){}
	BoutRealVar(BoutReal val):value_(val),valset_(true){}
	void set(BoutReal val){value_ = val;valset_=true;}
	BoutReal get() const {return value_;}
	bool valset() const {return valset_;}
	void valset(bool valset){valset_ = valset;}
	void operator=(BoutReal &val){this->value_ = val;this->valset_ = true;}
private:
        BoutReal value_;
        bool valset_;
};

class FieldVar {
public:
    FieldVar():value_(0.),valset_(false){}
	FieldVar(Field2D val):value_(val),valset_(true){}
	void set(Field2D val){value_ = val;valset_=true;}
	Field2D get() const {return value_;}
	bool valset() const {return valset_;}
	void valset(bool valset){valset_ = valset;}
	void operator=(Field2D &val){this->value_ = val;this->valset_ = true;}
private:
        Field2D value_;
        bool valset_;
};



class HeselParameters {
public:
	HeselParameters(){}
	//print values
	void printvalues() const;
	//GETTERS
	//Derived quantities getters
	BoutReal D() const; 	//Classical diffusion coefficient as derived in writeup
	BoutReal Dn() const;	//neo-classical diffusion coefficient (1+1.6q^2)Dn as derived in writeup
	BoutReal nuei() const;//electron-ion collision frequency
	BoutReal nuii() const;//ion-ion collision frequency
	BoutReal nuee() const;//electron-electron collision frequency
	BoutReal taun() const;//parameterized parallel damping coefficient of particle density
	BoutReal tauw() const;//parameterized parallel damping coefficient of generalized vorticity
	BoutReal taupe() const;//parameterized parallel damping coefficient of electron pressure
	BoutReal taupi() const;//parameterized parallel damping coefficient of ion pressure
	Field2D taushe() const;
	Field2D taushi() const;
	BoutReal taudw() const;
	BoutReal eta() const;//viscosity coefficient on generalized vorticity
	BoutReal chi_i_perp() const; //perpendicular ion heat conduction coefficient
	BoutReal chi_e_perp() const; //perpendicular electron heat conduction coefficient
	BoutReal chi_i_par() const; //parallel ion heat conduction coefficient
	BoutReal chi_e_par() const; //parallel electron heat conduction coefficient
	BoutReal me() const {return Me;};
	BoutReal mi() const;
	BoutReal rhoe() const;//electron gyro-radius
	BoutReal rhoi() const;//ion gyro-radius
	BoutReal rhos() const;//hybrid gyro radius c_s/\Omega_i
	BoutReal oce() const;
	BoutReal oci () const;//Ion gyro-frequency
	BoutReal cs() const; //ion sound speed Ti = 0
	BoutReal vte() const;//electron thermal speed
	BoutReal vti() const;//ion thermal speed
	BoutReal collog() const;//Coulomb Logarithm
	BoutReal tau() const;//Ti0/Te0
        BoutReal bohm_potential() const;// Bohm potential
        BoutReal neocorr_factor() const;// common factor for neoclassical correction
        BoutReal lblob() const;// parallel blob (ballooning) length, ~q*R, J. Madsen Equation 66
        BoutReal De() const; // De as in Jens Paper
        BoutReal Di() const; // Di as in Jens paper

	//gyro-Bohm Normalized derived quantities
	BoutReal nD() const {return D()/(rhos()*rhos()*oci());}; 	//Classical diffusion coefficient as derived in writeup
	BoutReal nDn() const {return Dn()/(rhos()*rhos()*oci());};	//neo-classical diffusion coefficient (1+1.6q^2)Dn as derived in writeup
	BoutReal nnuei() const {return nuei()/oci();};//electron-ion collision frequency
	BoutReal nnuii() const {return nuii()/oci();};//ion-ion collision frequency
	BoutReal nnuee() const {return nuee()/oci();};//electron-electron collision frequency
	BoutReal ntaun() const {return taun()*oci();};//parameterized parallel damping coefficient of particle density
	BoutReal ntauw() const {return tauw()*oci();};//parameterized parallel damping coefficient of generalized vorticity
	BoutReal ntaupe() const {return taupe()*oci();};//parameterized parallel damping coefficient of electron pressure
	BoutReal ntaupi() const {return taupi()*oci();};//parameterized parallel damping coefficient of ion pressure
//	BoutReal neta() const {return (1.+q()*q()*R()/a())*3./10.*tau()*nnuii();};//viscosity coefficient on generalized vorticity. extra n() due to normalization with respect to n_0
	BoutReal neta() const {return 3./10.*nchi_i_perp()/2;};//viscosity coefficient on generalized vorticity, nchi_i_perp = 2*nDi, neta = 3/10*nDi.
        BoutReal nchi_i_perp() const {return chi_i_perp()/(rhos()*rhos()*oci()*n());}; //perpendicular ion heat conduction coefficient
	BoutReal nchi_e_perp() const {return chi_e_perp()/(rhos()*rhos()*oci()*n());}; //perpendicular electron heat conduction coefficient
	BoutReal nchi_i_par() const {return chi_i_par()/(rhos()*rhos()*oci()*n());}; //parallel ion heat conduction coefficient
	BoutReal nchi_e_par() const {return chi_e_par()/(rhos()*rhos()*oci()*n());}; //parallel electron heat conduction coefficient
        
        // new set of normalized parameters, keep above for compatible reason
        Field2D normLc() const {return lconn()/rhos();};
        BoutReal normLb() const {return lblob()/rhos();};
        BoutReal normDe() const {return De()/(rhos()*rhos()*oci());};
        BoutReal normDi() const {return Di()/(rhos()*rhos()*oci());};
        BoutReal normDn() const {return (1.+tau())*normDe();};
        BoutReal normEta() const {return 3./10.*normDi();};
        BoutReal normTaun() const {return taun()*oci();};
        Field2D normTaushi() const {return taushi()*oci();};
        Field2D normTaushe() const {return taushe()*oci();};
        BoutReal normTaudw() const {return taudw()*oci();}; // parameterized drift wave damping rate

	//input parameters
	BoutReal q() const {return q_.get();}; //safety factor
	BoutReal B() const {return B_.get();};
	BoutReal R() const {return R_.get();};
	BoutReal a() const {return a_.get();};
	BoutReal Te() const {return Te_.get();};
	BoutReal Ti() const {return Ti_.get();};
	BoutReal n() const {return n_.get();};
	/*lconn[m]: parallel Connection length, collog: Coulomb logarithm
	 * A: ion nuclei, Z: ion charge, mp[kg]: proton mass, me [kg]: electron mass, mi[kg]: ion mass    */
	Field2D lconn() const {return lconn_.get();};// parallel connection length, parallel gradient length: Lc = Te/(dTe/dz) ~ Te,u/(Te,u-Te,d)*Lconn, When Te,u >> Te,d, Lc = Lconn, [P.C. Stangeby et al NF 2010 vol. 50 ] Equations 1 and 2
	//BoutReal collog(){return collog_.get();};
	BoutReal Mach() const {return Mach_.get();};
	BoutReal A() const {return A_.get();};
	BoutReal Z() const {return Z_.get();};

	//STOP GETTERS

	//SETTERS
	//Only possible to set fundamental parameters
	/*q: safety factor, B[tesla]: magnetic field, R[m]: Major radius, a[m]: minor radius,
	 * Te[eV] electron temperature, Ti[eV]: ion temperature, n[m^-3]*/
	void q(BoutReal var){q_.set(var);};//safety factor
	void B(BoutReal var){B_.set(var);};
	void R(BoutReal var){R_.set(var);};
	void a(BoutReal var){a_.set(var);};
	void Te(BoutReal var){Te_.set(var);};
	void Ti(BoutReal var){Ti_.set(var);};
	void n(BoutReal var){n_.set(var);};
	/*lconn[m]: parallel Connection length, collog: Coulomb logarithm
	 * A: ion nuclei, Z: ion charge, mp[kg]: proton mass, me [kg]: electron mass, mi[kg]: ion mass    */
	void lconn(Field2D var){lconn_.valset(true); lconn_.set(var);};
	void Mach(BoutReal var){Mach_.valset(true); Mach_.set(var);};
	void A(BoutReal var){A_.set(var);};
	void Z(BoutReal var){Z_.set(var);};
        
        void neocorr_factor(BoutReal nc_cor_fac) {neoclass_corr_fac = nc_cor_fac;};
        void lblob(BoutReal lb) {l_balloon = lb;};


private:
	//input parameters
	/*q: safety factor, B[tesla]: magnetic field, R[m]: Major radius, a[m]: minor radius,
	 * Te[eV] electron temperature, Ti[eV]: ion temperature, n[m^-3]*/
	BoutRealVar q_;//safety factor
	BoutRealVar B_;
	BoutRealVar R_;
	BoutRealVar a_;
	BoutRealVar Te_;
	BoutRealVar Ti_;
	BoutRealVar n_;
	/*lconn[m]: parallel Connection length, collog: Coulomb logarithm
	 * A: ion nuclei, Z: ion charge, mp[kg]: proton mass, me [kg]: electron mass, mi[kg]: ion mass    */
	FieldVar lconn_;
	BoutRealVar Mach_;
	BoutRealVar A_;
	BoutRealVar Z_;
        BoutReal neoclass_corr_fac = -1;
        BoutReal l_balloon = -1;

	//flag set true if all inputs are set
	//bool allset;
};


#endif /* HESELPARAMETERS_HXX_ */
