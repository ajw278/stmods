
import os
from definitions import *




def write_pdr_input(FUV_front, FUV_back, model_name='test', ifisob=0, chem_dir=CHEM_DIR, \
ifafm=20, densh=1e4, d_sour=0.0, fmrc=10.0, ieqth=1, tgaz=500.0, UV_units='Habing', itrfer=3, jfgkh2=2, vturb=2.0, gratio_0=0.01, \
cdunit_0=5.8e21, Z=1.0, q_pah=4.6e-2, rgrmin=1e-7, rgrmax=3e-5, F_DUST_P=0, iforh2=0, istic=4, F_W_ALL_IFAF=0, srcpp='O 8 V', input_filename='pdr.in', density_profile_filename=None):
	"""
	Writes the input file for the Meudon PDR code, specifying a user-defined density profile.

	Args:
		input_filename (str): The name of the PDR input file to be written.
		density_profile_filename (str): The name of the density profile file.
		FUV_front (float): The FUV luminosity at the front side (scaling factor for ISRF).
		FUV_back (float): The FUV luminosity at the back side (scaling factor for ISRF).
	"""



	if UV_units=='Habing':
		#Convert to Mathis field (see PDR manual -- I have used 1.6e-3 erg cm^2 s^-1 for Habing unit)
		FUV_back /= (1.92/1.6)
		FUV_front /= (1.92/1.6)
		

	with open(input_filename, 'w') as f:
		f.write("! Input file for Meudon PDR code with user-defined density profile\n")
		f.write(f"modele {model_name}\n")
		f.write(f"chimie {chem_dir}\n")
		f.write(f"ifafm {ifafm:d}  ! Number of iterations\n")
		f.write(f"AVmax {AVmax:%.2lf} ! Maximum extinction\n")
		f.write(f"densh {densh:.2e}  ! Constant hydrogen density\n")
		f.write(f"F_ISRF 1  ! Use the Mathis ISRF\n")
		f.write(f"radm {FUV_front:.2e}  ! FUV scaling factor for the front side\n")
		f.write(f"radp {FUV_back:.2e}  ! FUV scaling factor for the back side\n")
		f.write(f"dsour {dsour:.2e}  ! Distance to star in parsec (-ve observer side, +ve far side, 0 is none)\n")
		f.write(f"srcpp {srcpp} ! Stellar spectrum -- either spectral type (data/Astrodata/star.dat) or ASCII file input")

		"""
		To provide a specific stellar spectrum, build a file containing the flux as a function of wavelength. This file has to be stored in the data/Astrodata directory. The name of this file must begin with F_. Then provide the name of this file in the pdr.in file in the srcpp parameter. The format for this file is:
		• first line: radius of the star in solar radius
		• second line: effective temperature in K
		• third line:number of points in the spectrum
		• forth line: comment
		• then the file must contain the spectrum in two columns with, on the first column, wavelengths in nm and on the second one the flux in erg cm−2 s−1 nm−1 sr−1.
		"""

		f.write(f"fmrc {fmrc:.2e}  ! Cosmic ray ionisation rate in units of 10^-17 s^-1 -- typical value = 10\n")
		f.write(f"ieqth {ieqth:d}  ! Thermal balance flag - 0=isothermal, 1=compute T\n")
		f.write(f"tgaz {tgaz:d}  ! Initial guess (or fixed) gas temperature in K\n")
		f.write(f"ifisob {ifisob:d}  ! Use user-defined density profile\n")
		if not density_profile_filename is None:
			f.write(f"fprofil {density_profile_filename}  ! Path to the user-defined density profile\n")
		f.write(f"presse  {presse:%.2e}! Pressure, only for isobaric model\n")
		f.write(f"vturb {vturb:.2f} ! Turbulent velocity for line broadening (km/s)\n")
		f.write(f"itrfer {itrfer:d} ! Radiative transfer method (exact for H - 1 , +H2 - 2, 12CO - 3, ... etc.)\n")
		f.write(f"jfgkh2 {jfgkh2:d} ! Rotational H2 quantum number (J) below which exact UV radiative transfer is solved in the UV lines\n")
		f.write(f"los_ext Galaxy ! extinction curve (see line_of_sight.dat in Astrodata)\n")
		f.write(f"rrr 3.1 ! RV = AV / E(B-V). Total to selective extinction ratio. (see line_of_sight.dat in Astrodata)\n")
		f.write(f"metal {Z:.2e} ! Metallicity to scale grains and elemental abundances by (typically 1)\n")
		f.write(f"cdunit_0 {cdunit_0:.2e} ! N(H) / E(B-V) - typical value for the Galaxy: 5.8 1021 - Bohlin et al. (1978)\n")
		f.write(f"gratio_0 {gratio_0:.2e} ! Dust to gas mass ratio for Z = 1\n")
		f.write(f"q_pah {q_pah:.2e} ! PAH to dust mass ratio for Z = 1, typical value: 4.6E-2\n")
		f.write(f"alpgr {alpgr:.2f} ! index of grain size distribution -- 3.5 according to Mathis, Rumpl & Nordsieck (1977)\n")
		f.write(f"rgrmin {rgrmin:.2f} ! Minimum grain radius excluding PAHs (typically 1E-7 cm)\n")
		f.write(f"rgrmax {rgrmax:.2f} ! Maximum grain radius excluding PAHs (typically 3E-5 cm)\n")
		f.write(f"F_DUST_P {F_DUST_P:d} ! Grain model -- Flag to activate or not the coupling of the PDR code with DustEM.\n")
		f.write(f"iforh2 {iforh2:d} ! H2 excitation formation model. Default value is 0.\n")
		f.write(f"istic {istic:d} ! H sticking on grains model. Default value is 4.\n")
		f.write(f"F_W_ALL_IFAF {F_W_ALL_IFAF:d} ! Flag for writing outputs for all simulations\n")

	return input_filename

def write_density_profile(density_profile, r_non_ionized, av_profile, Temp=500.0, filename='density_profile.pfl'):
    """
    Write the density profile to a file that can be used by the Meudon PDR code.
    
    Parameters:
    -----------
    density_profile : 1D array
        The density profile along the line of sight (non-ionized region).
    r_non_ionized : 1D array
        The radial distances corresponding to the density profile.
    av_profile : 1D array
        The visual extinction profile.
    filename : str
        The name of the output density profile file.
    """
    with open(filename, 'w') as f:
        f.write(f"{len(r_non_ionized)}\n")  # Number of points in the profile
        for i in range(len(r_non_ionized)):
            f.write(f"{av_profile[i]:.3f} {Temp:.1f} {density_profile[i]:.3e}\n")  # A_V, temp (placeholder), density
    return filename


def write_stellar_spectrum(wavelength, flux, filename='stellar_spectrum.dat'):
    """
    Write the stellar spectrum file for the Meudon PDR code.
    
    Parameters:
    -----------
    wavelength : 1D array
        Wavelengths of the stellar spectrum (in nm).
    flux : 1D array
        Flux values (in erg cm^-2 s^-1 nm^-1 sr^-1).
    filename : str
        The name of the output spectrum file.
    """
    with open(filename, 'w') as f:
        f.write(f"1.0  # Stellar radius in solar radius\n")  # Placeholder radius
        f.write(f"10000.0  # Effective temperature in K\n")  # Placeholder temperature
        f.write(f"{len(wavelength)}  # Number of points\n")
        f.write("# Wavelength (nm)   Flux (erg cm^-2 s^-1 nm^-1 sr^-1)\n")
        for wl, fl in zip(wavelength, flux):
            f.write(f"{wl:.3f} {fl:.3e}\n")
    return filename

"""
def gen_PDR(FUV_front, FUV_back, AV_prof, dense_prof, isobaric=False, T=500.0, **kwargs)

	if isobaric:
		
		kwargs['ieqth'] = 1 
		kwargs['AVmax'] = AV_prof[-1]
		kwargs['ifisob'] = 2
		if not 'presse' in kwargs:
			rho = np.mean(dense_prof)
			nH = rho*Msol2g/(mu * mH * pc2cm**3)
			kwargs['presse'] = nH*T
	else:
		write_density_profile(filename, AV, density_Msolpc, T=500.0)
"""
	
		
def run_pdr_model(FUV_front, FUV_back, r_non_ionized, density_profile, av_profile, wavelength, flux, 
					model_name='test_pdr_model', input_filename='pdr.in'):
	"""
	Runs the PDR model by writing necessary input files and running the code.

	Parameters:
	-----------
	FUV_front : float
		The FUV luminosity at the front.
	FUV_back : float
		The FUV luminosity at the back.
	r_non_ionized : 1D array
		Radial distances in the non-ionized region.
	density_profile : 1D array
		Density profile along the non-ionized region.
	av_profile : 1D array
		Visual extinction profile along the line of sight.
	wavelength : 1D array
		Wavelengths of the stellar spectrum.
	flux : 1D array
		Flux of the stellar spectrum.
	model_name : str
		Name of the model.
	input_filename : str
		Name of the PDR input file.

	Returns:
	--------
	input_filename : str
		The name of the PDR input file.
	"""
	# Step 1: Write the density profile file
	density_profile_filename = write_density_profile(density_profile, r_non_ionized, av_profile, filename='density_profile.pfl')

	# Step 2: Write the stellar spectrum file
	stellar_spectrum_filename = write_stellar_spectrum(wavelength, flux, filename='stellar_spectrum.dat')

	# Step 3: Generate the PDR input file
	write_pdr_input(FUV_front, FUV_back, model_name=model_name, input_filename=input_filename, 
					density_profile_filename=density_profile_filename, srcpp=stellar_spectrum_filename)

	return input_filename

import subprocess

def execute_pdr_code(pdr_input_filename):
	"""
	Execute the Meudon PDR code.

	Parameters:
	-----------
	pdr_input_filename : str
		The input file for the PDR model.
	"""
	result = subprocess.run([PDR_CDIR+'PDR', pdr_input_filename], capture_output=True, text=True)
	return result.stdout, result.stderr






if __name__=='__main__':

	# Filenames
	input_filename = "pdr_input_with_density.in"
	density_profile_filename = "user_density_profile.pfl"

	# Write the PDR input file pointing to the user-defined density profile
	write_pdr_input_with_density_file(input_filename, density_profile_filename, FUV_front, FUV_back)

	# Write the user-defined density profile file
	write_density_profile(density_profile_filename, z_prof, rho_prof)

	# Execute the Meudon PDR code
	run_pdr_code(input_filename)
	



