
import os

CWD = os.getcwd()
PDR_CDIR = '/home/awinter/PDR1.5.4_210817_rev2095/'
CHEM_DIR = 'data/Chimie/'

def write_pdr_input_with_density_file(input_filename, density_profile_filename, FUV_front, FUV_back, model_name='test', chem_dir=CHEM_DIR), ifafm=20, densh=1e4, \
d_sour=0.0, fmrc=10.0, ieqth=1, tgaz=500.0, UV_units='Habing', itrfer=3, jfgkh2=2, vturb=2.0, gratio_0=0.01, cdunit_0=5.8e21, Z=1.0, q_pah=4.6e-2, rgrmin=1e-7, \
rgrmax=3e-5, F_DUST_P=0, iforh2=0, istic=4, F_W_ALL_IFAF=0, srcpp='O 8 V'):
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
		f.write(f"modele {model_name}")
		f.write(f"chimie {chem_dir}")
		f.write(f"ifafm {ifafm:d}  ! Number of iterations\n")
		#f.write(f"AVmax 20.0  ! Maximum extinction\n")
		#f.write(f"densh {densh:.2e}  ! Maximum extinction\n")
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
		f.write(f"ifisob 1  ! Use user-defined density profile\n")
		f.write(f"fprofil {density_profile_filename}  ! Path to the user-defined density profile\n")
		#f.write(f"presse  {presse:%.2e}! Pressure, only for isobaric model\n")
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


def write_density_profile(filename, DeltaZ, density):
	"""
	Writes a user-defined density profile for the Meudon PDR code.

	Args:
		filename (str): The name of the density profile file.
		DeltaZ (float): The thickness of the slab (in parsecs or similar unit).
		density (float): The constant density in cm^-3.
	"""
	# Number of points in the profile
	num_points = 100  # Adjust as needed for more or fewer points
	step_av = DeltaZ / num_points

	with open(filename, 'w') as f:
		f.write(f"{num_points}  ! Number of points in the density profile\n")
		for i in range(num_points):
			Av = i * step_av
			temperature = 10.0  # Example constant temperature, modify as needed
			f.write(f"{Av:.5f} {temperature:.2f} {density:.2e}\n")


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
	



