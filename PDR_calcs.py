
import os
from definitions import *

import subprocess


def write_pdr_input(FUV_front, FUV_back, model_name='test', ifisob=0, chem_dir=CHEM_DIR, \
ifafm=20, densh=1e4, d_sour=0.0, fmrc=10.0, ieqth=1, tgaz=500.0, UV_units='Habing', itrfer=3, jfgkh2=2, vturb=2.0, gratio_0=0.01, dsour=0.0, AVmax=20.0, \
cdunit_0=5.8e21, Z=1.0, q_pah=4.6e-2, rgrmin=1e-7, rgrmax=3e-5, F_DUST_P=0, iforh2=0, istic=4, alpgr=3.5, F_W_ALL_IFAF=0, presse=0.0, ichh2 = 2,srcpp='O 8 V', \
input_filename='pdr.in', density_profile_filename=None):
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
		f.write(f"{model_name.ljust(47)}  ! modele   : Output files radix\n")
        f.write(f"{chem_dir.ljust(47)}  ! chimie   : Chemistry file\n")
        f.write(f"{str(ifafm).ljust(47)}  ! ifafm    : Number of global iterations\n")
        f.write(f"{f'{AVmax:.2f}'.ljust(47)}  ! Avmax    : Integration limit (Av)\n")
        f.write(f"{f'{densh:.2e}'.ljust(47)}  ! densh    : Initial density (nH in cm-3)\n")
        f.write(f"{'1'.ljust(47)}  ! F_ISRF   : Use Mathis ISRF (1 = Mathis, 2 = Draine)\n")
        f.write(f"{f'{FUV_front:.2e}'.ljust(47)}  ! radm     : FUV scaling factor (Observer side)\n")
        f.write(f"{f'{FUV_back:.2e}'.ljust(47)}  ! radp     : FUV scaling factor (Back side)\n")
        f.write(f"{srcpp.ljust(47)}  ! srcpp    : Stellar spectrum or none\n")
        f.write(f"{f'{d_sour:.2e}'.ljust(47)}  ! d_sour   : Star distance (pc)\n")
        f.write(f"{f'{fmrc:.2f}'.ljust(47)}  ! fmrc     : Cosmic rays ionization rate (10^-17)\n")
        f.write(f"{str(ieqth).ljust(47)}  ! ieqth    : Thermal balance flag (1 = solve, 0 = fixed T)\n")
        f.write(f"{f'{tgaz:.2f}'.ljust(47)}  ! tgaz     : Initial temperature (K)\n")
        f.write(f"{str(ifisob).ljust(47)}  ! ifisob   : Equation of state (0 = constant density, 1 = user profile)\n")
        f.write(f"{density_profile_filename.ljust(47)}  ! fprofil  : Density profile file\n")
        if presse is not None:
            f.write(f"{f'{presse:.2e}'.ljust(47)}  ! presse   : Pressure, only for isobaric model\n")
        f.write(f"{f'{vturb:.2f}'.ljust(47)}  ! vturb    : Turbulent velocity (km/s)\n")
        f.write(f"{str(itrfer).ljust(47)}  ! itrfer   : Radiative transfer method\n")
        f.write(f"{str(jfgkh2).ljust(47)}  ! jfgkh2   : Use FGK approximation for J >= jfgkh2\n")
        f.write(f"{str(2).ljust(47)}  ! ichh2    : H + H2 collision rate model (2 is standard)\n")
        f.write(f"Galaxy".ljust(47) + "  ! los_ext  : Line of sight extinction curve\n")
        f.write(f"{f'{3.1:.2f}'.ljust(47)}  ! rrr      : Rv = Av / E(B-V) (Typical value 3.1)\n")
        f.write(f"{f'{Z:.2e}'.ljust(47)}  ! metal    : Metallicity (automatically scales grains and abundances)\n")
        f.write(f"{f'{cdunit_0:.2e}'.ljust(47)}  ! cdunit_0 : NH / E(B-V) (typical: 5.8e21 for Galaxy)\n")
        f.write(f"{f'{gratio_0:.2e}'.ljust(47)}  ! gratio_0 : Dust to gas mass ratio (Z=1)\n")
        f.write(f"{f'{q_pah:.2e}'.ljust(47)}  ! q_pah    : PAH to dust mass ratio (default = 4.6e-2)\n")
        f.write(f"{f'{alpgr:.2f}'.ljust(47)}  ! alpgr    : Grain size distribution index (MRN dist. = 3.5)\n")
        f.write(f"{f'{rgrmin:.2e}'.ljust(47)}  ! rgrmin   : Minimum grain radius (typically 1e-7 cm)\n")
        f.write(f"{f'{rgrmax:.2e}'.ljust(47)}  ! rgrmax   : Maximum grain radius (typically 3e-5 cm)\n")
        f.write(f"{str(F_DUST_P).ljust(47)}  ! F_DUST_P : Activate DUSTEM (1=yes, 0=no)\n")
        f.write(f"{str(iforh2).ljust(47)}  ! iforh2   : H2 formation model (0 = standard)\n")
        f.write(f"{str(istic).ljust(47)}  ! istic    : H2 sticking model (4 = standard)\n")
        f.write(f"{str(F_W_ALL_IFAF).ljust(47)}  ! F_W_ALL_IFAF : Write outputs for all iterations\n")

		"""
		To provide a specific stellar spectrum, build a file containing the flux as a function of wavelength. This file has to be stored in the data/Astrodata directory. The name of this file must begin with F_. Then provide the name of this file in the pdr.in file in the srcpp parameter. The format for this file is:
		• first line: radius of the star in solar radius
		• second line: effective temperature in K
		• third line:number of points in the spectrum
		• forth line: comment
		• then the file must contain the spectrum in two columns with, on the first column, wavelengths in nm and on the second one the flux in erg cm−2 s−1 nm−1 sr−1.
		"""

	return input_filename

def write_density_profile(density_profile, av_profile, Temp=500.0, filename='density_profile.pfl'):
    """
    Write the density profile to a file that can be used by the Meudon PDR code.
    
    Parameters:
    -----------
    density_profile : 1D array
        The density profile along the line of sight (non-ionized region).
    av_profile : 1D array
        The visual extinction profile.
    filename : str
        The name of the output density profile file.
    """
    with open(filename, 'w') as f:
        f.write(f"{len(av_profile)}\n")  # Number of points in the profile
        for i in range(len(av_profile)):
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


PDR_PATH = '/home/awinter/PDR1.5.4_210817_rev2095/'
IDAT_PATH = PDR_PATH+'AnalysisTools/IDAT/idat'

def run_idat_extraction(hdf5_file, selection_file, output_file, idat_path=IDAT_PATH):
		"""
		Runs IDAT to extract quantities from an HDF5 file and save them as a CSV or ASCII file.

		Args:
			hdf5_file (str): Path to the input HDF5 file.
			selection_file (str): Path to the selection file listing the quantities to extract.
			output_file (str): Path to the output file where the extracted data will be saved.
			idat_path (str): Path to the IDAT executable (default is './idat').
		"""
		# Construct the command
		command = [idat_path, 'extract', hdf5_file, selection_file, output_file]

		try:
			# Execute the command
			result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

			# Output the command results
			print(f"IDAT Extraction completed. Output saved to {output_file}")
			print(result.stdout.decode())

		except subprocess.CalledProcessError as e:
			# Handle any errors during the extraction process
			print(f"Error during IDAT extraction: {e.stderr.decode()}")



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
	
		
def run_pdr_model_tcalc(FUV_front, FUV_back, r_non_ionized, density_profile, av_profile, wavelength, flux, 
					model_name='test_pdr_model', input_filename='pdr.in', wl_units='angstrom', run_dir='test_dir'):
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

	if wl_units=='angstrom':
		wavelength /= 10.0
		flux *= 10.0
	elif not wl_units in ['nanometers', 'nm']:
		raise Warning('Wavelength unit not recognised.')

	dsour= -1.*r_non_ionized
	avmax = av_profile[-1]
	ifisob = 1


	
	

	# Step 1: Write the density profile file
	density_profile_filename = write_density_profile(density_profile, av_profile, filename='density_profile.pfl')

	# Step 2: Write the stellar spectrum file
	stellar_spectrum_filename = write_stellar_spectrum(wavelength, flux, filename='stellar_spectrum.dat')

	# Step 3: Generate the PDR input file
	write_pdr_input(FUV_front, FUV_back, model_name=model_name, input_filename=input_filename, 
					density_profile_filename=density_profile_filename, srcpp=stellar_spectrum_filename,\
					dsour=dsour, AVmax = avmax, ifisob=ifisob)

	
	result = subprocess.run([PDR_CDIR+'src/PDR', input_filename, '> pdr_out.out'], capture_output=True, text=True)
	with open("pdr_out.out", "w") as text_file:
		text_file.write(result.stdout)
		text_file.write(result.stderr)
	return input_filename


def execute_pdr_code(pdr_input_filename):
	"""
	Execute the Meudon PDR code.

	Parameters:
	-----------
	pdr_input_filename : str
		The input file for the PDR model.
	"""
	result = subprocess.run([PDR_CDIR+'PDR' + pdr_input_filename + '> pdr_out.out'], capture_output=True, text=True)
	print(result)
	exit()
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
	stdout, run_pdr_code(input_filename)
	

	exit()



