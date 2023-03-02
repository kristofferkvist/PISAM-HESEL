/*
 * HeselParameters.cxx
 *
 *  Created on: Oct 22, 2014
 *      Author: yolen
 */

#include "HeselParameters.hxx"
//#include "/home/yolen/numerics/BOUT-dev/include/bout/constants.hxx"


/*HeselParameters::HeselParameters() {
	allset = false;
	//collog_.set(13.3969); //Default value for Coulomb logarithm
}

HeselParameters::~HeselParameters() {
	// TODO Auto-generated destructor stub
}
*/

//GETTERS
/*
 * Calculate and return the collision frequencies
 */
BoutReal HeselParameters::collog() const {
	//check that input parameters are set
	if(n_.valset() && Z_.valset() && Te_.valset())
	{
		BoutReal debye = sqrt(epso*e*Te()/(e*e*n()));
		return log(12.*PI*n()*pow(debye,3) / Z());// F.F.Chen P227
	}
	else
	{
		throw BoutException("not all values set in collog\n");
	}

}

BoutReal HeselParameters::nuei() const {
	//check that input parameters are set
	if(n_.valset() && Z_.valset()  && Te_.valset())
	{
		return sqrt(2)*n()*Z()*Z()*pow(e,4)*collog()/(12.0*sqrt(pow(PI,3))*sqrt(me())*sqrt(pow(e*Te(),3))*epso*epso);// J. Madsen Equation 4
	}
	else
	{
		throw BoutException("not all values set in nuei\n");
	}

}

BoutReal HeselParameters::nuii() const {
	if(n_.valset() && Z_.valset()  && Ti_.valset())
	{
		return n()*pow(Z(),4)*pow(e,4)*collog()/(12.0*sqrt(pow(PI,3))*(epso*epso)*sqrt(mi())*sqrt(pow(e*Ti(),3)));//ion-ion collision frequency, J. Madsen Equation 7
	}
	else
	{
		throw BoutException("not all values set in nuii\n");

	}
}

BoutReal HeselParameters::nuee() const {
	if(Z_.valset())
	{
//		return sqrt(2)*n()*pow(e,4)*collog()/(12.0*sqrt(pow(PI,3))*(epso*epso)*sqrt(Me)*sqrt(pow(e*Te(),3)));//electron-electron collision frequency nuee = sqrt(2)*n*(e^4)*collog/(12.0*sqrt(pi^3)*(epso^2)*sqrt(me)*sqrt((e*Te)^3))
                return nuii()/pow(Z(),4)*sqrt(mi()/me())*sqrt(pow(tau(),3));// F.F.Chen P398
	}
	else
	{
		throw BoutException("Error in HeselParameters: not all values set in nuee\n");

	}
}

BoutReal HeselParameters::D() const {
		return  nuei() * rhoe()*rhoe(); 	//Classical electron diffusion coefficient, J. Madsen next line below equation 50
}

//neo-classical diffusion coefficient (1+1.6q^2)Dn as derived in writeup
BoutReal HeselParameters::Dn() const {
	if(q_.valset()){
		return D()*neocorr_factor();// J. Madsen Equation 86: De, normalization factor should be ignored
	}else
	{
		throw BoutException("Error in HeselParameters: not all values set in Dn\n");

	}
}

BoutReal HeselParameters::De() const {
    return neocorr_factor()*rhoe()*rhoe()*nuei();// Jens Equation 96
} 

BoutReal HeselParameters::Di() const {
    return neocorr_factor()*rhoi()*rhoi()*nuii();// Jens Equation 96
} 

//parameterized parallel damping coefficient of particle density
BoutReal HeselParameters::taun() const {
    if (Mach_.valset()){
	return lblob() /(2.*Mach()*cs());// J. Madsen Equation 87, in hesel.cxx should be divided by normalized ion sound speed
    } else {
        throw BoutException("Error in HeselParameters: not all values set in taun\n");
    }
}

//parameterized parallel damping coefficient of generalized vorticity
BoutReal HeselParameters::tauw() const {
    return taun();// J.Madsen Equation 82
}

//parameterized parallel damping coefficient of electron pressure
BoutReal HeselParameters::taupe() const {
    //	return lconn()*lconn()/(chi_e_par());
//    return lconn()*lconn()*n()/(chi_e_par()); // change unit to time, so multiply by n(), J. Madsen Equation 87 t_SH
    return lblob()*lblob()*n()/(chi_e_par()); // change unit to time, so multiply by n(), J. Madsen Equation 87 t_SH, according to Anders, Lc is approximated to ballooning length
}

Field2D HeselParameters::taushi() const {
    return lconn()*lconn()*n()/(chi_i_par());
}

Field2D HeselParameters::taushe() const {
    return lconn()*lconn()*n()/(chi_e_par()); // change unit to time, so multiply by n(), J. Madsen Equation 87 t_SH, according to Anders, Lc is approximated to ballooning length
}

BoutReal HeselParameters::taudw() const {
    return lblob()*lblob()*me()*nuei()/(2.*e*Te());
}

BoutReal HeselParameters::taupi() const {
//parameterized parallel damping coefficient of ion pressure
    return 2./9.*taun();// J. Madsen Equation 84
}

//viscosity coefficient on generalized vorticity
BoutReal HeselParameters::eta() const {
	return 3./10. *e*Ti()*n()*nuii()/(oci()*oci())/(Z()*e)*neocorr_factor();
}

//neo-classical perpendicular ion heat conduction coefficient
BoutReal HeselParameters::chi_i_perp() const {
	return 2.0*n()*nuii()*rhoi()*rhoi()*neocorr_factor();// J. Madsen Equation 28 ki,perp
}

//neo-classical perpendicular electron heat conduction coefficient
BoutReal HeselParameters::chi_e_perp() const {
	return 4.66*n()*nuei()*rhoe()*rhoe()*neocorr_factor();// J. Madsen Equation21 ke,perp
}

BoutReal HeselParameters::chi_i_par() const {
    return  3.9*n()*e*Ti()/(mi()*nuii());// J. Madsen Equation 28 ki,//
}

//classical parallel electron heat conduction coefficient, not enhanced by neo-classical transport.
BoutReal HeselParameters::chi_e_par() const {
    return  3.16*n()*e*Te()/(me()*nuei());// J. Madsen Equation 21 ke,//
}

BoutReal HeselParameters::mi() const {
	if(A_.valset())
	{
		return A()*Mp;
	}else
	{
		throw BoutException("In mi() in HeselParameters.cxx: A not set\n exiting!!");

	}
}

BoutReal HeselParameters::rhoe() const {
		return vte()/oce();
}

BoutReal HeselParameters::rhoi() const {
		return vti()/oci();
}

BoutReal HeselParameters::rhos() const {
		return cs()/oci();// J. Madsen Equation 74
}

BoutReal HeselParameters::oce() const {
	if(B_.valset())
	{
		return e*B()/me();
	}else
	{
		throw BoutException("In Heselparameters: not all values set in oce\nexiting\n");

	}
}

BoutReal HeselParameters::oci() const {
	if(B_.valset() )
	{
		return e*Z()*B()/mi();
	}else
	{
		throw BoutException("In Heselparameters: not all values set in oci\nexiting\n");

	}
}

BoutReal HeselParameters::vte() const {
	if(Te_.valset() )
	{
		return sqrt( e*Te()/me());
	}else
	{
		throw BoutException("In Heselparameters: not all values set in vte\nexiting\n");

	}
}

BoutReal HeselParameters::vti() const {
	if(Ti_.valset() )
	{
		return sqrt( e*Ti()/mi()) ;
	}else
	{
		throw BoutException("In Heselparameters: not all values set in vti\nexiting\n");

	}
}

BoutReal HeselParameters::cs() const {
	if(Te_.valset() )
	{
		return sqrt( e*Te()/mi());
	}else
	{
		throw BoutException("In Heselparameters: not all values set in cs\nexiting\n");

	}
}

BoutReal HeselParameters::tau() const {
	if(Te_.valset() && Ti_.valset())
	{
		return Ti()/Te();// J. Madsen next line below Equation 84
	}else
	{
		throw BoutException("In Heselparameters: not all values set in tau\nexiting\n");

	}
}

BoutReal HeselParameters::bohm_potential() const{
    return log(sqrt(mi()/(2.*PI*me())));//J. Madsen next line below Equation 68
}

BoutReal HeselParameters::neocorr_factor() const{
    if ( neoclass_corr_fac >= 0 ){
        return 1. + neoclass_corr_fac;
    }else{
        return 1. + R()/a()*q()*q();// J. Madsen Equation 64
    }
    return 1.;
}

BoutReal HeselParameters::lblob() const{
    if (l_balloon > 0){
        return l_balloon;
    }else{
        return q()*R(); // J. Madsen Equation 66
    }
}

void HeselParameters::printvalues() const {
	output.write("Coulomb Log = %e\n",collog());
	output.write("vte         = %e\t vti       = %e\t cs      = %e\n",vte(),vti(),cs());
	output.write("oci         = %e\t oce       = %e\n",oci(),oce());
	output.write("rhoe        = %e\t rhoi      = %e\t rhos    = %e\n",rhoe(),rhoi(),rhos());
	output.write("mi          = %e\t me        = %e\n",mi(),me());
	output.write("n0          = %e\t Te0       = %e\t Ti0     = %e\n",n(),Te(),Ti());
	output.write("B0          = %e\t q95       = %e\t Rmajor  = %e\n",B(),q(),R());
        
	output.write("nuei        = %e\t nuii      = %e\t nuee    = %e\n",nuei(),nuii(),nuee());
	//output.write("Lc          = %e\t Lb        = %e\n",lconn(), lblob());
    output.write("neo-classical correction factor: %e\n", neocorr_factor());
	output.write("De          = %e\t Di        = %e\n",De(), Di());
	output.write("De(1+tau)   = %e\n",(1.+Ti()/Te())*De());
	output.write("eta         = %e\n",eta());
	output.write("chi_i_perp  = %e\n",chi_i_perp());
	output.write("chi_e_perp  = %e\n",chi_e_perp());
	output.write("chi_i_par   = %e\n",chi_i_par());
	output.write("chi_e_par   = %e\n",chi_e_par());
	output.write("taun        = %e\t 1/taun    = %e\n",taun(), 1./taun());
//	output.write("taushe      = %e\t 1/taushe  = %e\n",taushe(),1./taushe());
//	output.write("taushi      = %e\t 1/taushi  = %e\n",taushi(),1./taushi());
	output.write("taudw       = %e\t 1/taudw   = %e\n",taudw(),1./taudw());

	//gyrobohm normalized parameters
	output.write("\nGyroBohm normalized parameters\n");
	output.write("normalized diffusion coeff: %e\n", rhos()*rhos()*oci());
	output.write("nnuei       = %e\t nnuii     = %e\t nnuee   = %e\n",nnuei(),nnuii(),nnuee());
//	output.write("nLc         = %e\t nLb       = %e\n",normLc(),normLb());
	output.write("nDe         = %e\n",normDe());
	output.write("nDi         = %e\n",normDi());
	output.write("neta        = %e\n",normEta());
	output.write("nchi_i_perp = %e\n",nchi_i_perp());
	output.write("nchi_e_perp = %e\n",nchi_e_perp());
	output.write("nchi_i_par  = %e\n",nchi_i_par());
	output.write("nchi_e_par  = %e\n",nchi_e_par());
	output.write("ntaun       = %e\t 1/ntaun   = %e\n",normTaun(),1./normTaun());
//	output.write("ntaushe     = %e\t 1/ntaushe = %e\n",normTaushe(),1./normTaushe());
//	output.write("ntaushi     = %e\t 1/ntaushi = %e\n",normTaushi(),1./normTaushi());
	output.write("ntaudw      = %e\t 1/ntaudw  = %e\n",normTaudw(),1./normTaudw());

}
