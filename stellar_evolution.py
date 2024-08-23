import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pysynphot as S

from astropy.coordinates import SkyCoord
import astropy.units as u

G = 6.67e-8
Msol  =1.989e33
au = 1.496e13
sig_SB = 5.67e-5 #K^-4
mp = 1.6726e-24
k_B = 1.81e-16 #cgs K^-1
Lsol = 3.828e33 #erg/s
year = 365.*24.*60.0*60.0
#Rsol in cm
Rsol = 6.957e10
h = 6.626e-27  # Planck's constant in erg*s
c = 2.998e10   # Speed of light in cm/s
eV = 1.602e-12


h = 6.626e-27  # Planck's constant in erg*s
c = 2.998e10   # Speed of light in cm/s
k_B = 1.381e-16 # Boltzmann constant in erg/K

def find_closest_mass_file(directory, target_mass, tol= 0.5):
	closest_mass = None
	closest_filename = None

	for filename in os.listdir(directory):
		if filename.endswith('.track.eep'):
			try:
				mass_str = filename.split('M')[0]
				mass = float(mass_str) / 100  # Convert to solar masses assuming filename like '00037M.track.eep'
				if closest_mass is None or abs(mass - target_mass) < abs(closest_mass - target_mass):
					closest_mass = mass
					closest_filename = filename
			except ValueError:
				continue

	
	if closest_mass is None:
		raise Warning('No stellar model file found.')
	if abs(closest_mass-target_mass)/target_mass>tol and target_mass>0.1:
		
		raise Warning('No stellar model within %d percent of specified initial mass found.'%(tol*100))
		

	return closest_filename, closest_mass

def extract_eep_data(filepath):
	# Define the columns based on the README
	columns = [
	'star_age',  'star_mass', 'log_L', 'log_Teff', 'log_R', 'log_g'
	]

	# Read the data
	data = pd.read_csv(filepath, delim_whitespace=True, comment='#', names=columns, usecols=[0, 1, 6, 11, 13, 14])

	# Convert log values to their actual values
	data['L'] = 10**data['log_L']
	data['Teff'] = 10**data['log_Teff']
	data['R'] = 10**data['log_R']

	return data
    
def interpolate_stellar_properties(data, target_age):
	# Ensure the data is sorted by age
	data = data.sort_values('star_age')

	# Interpolation functions
	interpolate_Teff = interp1d(data['star_age'], data['Teff'], kind='linear', fill_value="extrapolate")
	interpolate_log_g = interp1d(data['star_age'], data['log_g'], kind='linear', fill_value="extrapolate")
	interpolate_log_L = interp1d(data['star_age'], data['log_L'], kind='linear', fill_value="extrapolate")
	interpolate_R = interp1d(data['star_age'], data['R'], kind='linear', fill_value="extrapolate")
	interpolate_mass = interp1d(data['star_age'], data['star_mass'], kind='linear', fill_value="extrapolate")

	# Interpolate values at target age
	Teff = interpolate_Teff(target_age)
	log_g = interpolate_log_g(target_age)
	log_L = interpolate_log_L(target_age)
	R = interpolate_R(target_age)
	star_mass = interpolate_mass(target_age)


	return Teff, log_g, log_L, R, star_mass

def example_plot(directory, target_mass):
	closest_filename, closest_mass = find_closest_mass_file(directory, target_mass)

	print(f"Closest mass file: {closest_filename} with mass {closest_mass} M_sun")

	filepath = os.path.join(directory, closest_filename)
	eep_data = extract_eep_data(filepath)
	# Plotting the results
	import matplotlib.pyplot as plt
	plt.rc('text', usetex=True)
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(eep_data['star_age'], eep_data['Teff'])
	plt.xlabel('Age [years]')
	plt.ylabel('Effective Temperature [K]')
	plt.yscale('log')
	plt.xscale('log')

	plt.subplot(2, 2, 2)
	plt.plot(eep_data['star_age'], eep_data['log_g'])
	plt.xlabel('Age [years]')
	plt.ylabel('$\log g$')
	plt.xscale('log')

	plt.subplot(2, 2, 3)
	plt.plot(eep_data['star_age'], eep_data['star_mass'])
	plt.xlabel('Age [years]')
	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel('Stellar Mass [$M_\odot$]')
	plt.subplot(2, 2, 4)
	plt.plot(eep_data['star_age'], eep_data['L'])
	plt.xlabel('Age (years)')
	plt.ylabel('Luminosity [$L_\odot$]')
	plt.yscale('log')
	plt.xscale('log')
	plt.tight_layout()
	plt.show()

def fetch_stellar_properties(minit, age_years, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
	closest_filename, closest_mass = find_closest_mass_file(directory, minit)
	
	filepath = os.path.join(directory, closest_filename)
	eep_data = extract_eep_data(filepath)
	print(filepath)

	Teff, log_g, log_L, R, star_mass = interpolate_stellar_properties(eep_data, age_years)

	print('Mass, age, Teff, logL:', minit, Teff, log_L, R, star_mass)
	

	return Teff, log_g, log_L, R, star_mass

def find_all_mass_files(directory):
    """
    Find all filenames in the directory that correspond to different initial masses.
    Assumes filenames are in the format 'MIST_mass_X.XX.dat', where X.XX is the mass.
    """
    filenames = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.track.eep'):  # Assuming .dat files are the ones to be used
            filenames.append(os.path.join(directory, filename))
    
    return filenames


def find_closest_age(data, target_age):
    """
    Find the closest available age in the data to the target age.
    """
    return data['star_age'][np.argmin(np.abs(data['star_age'] - target_age))]

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_all_isochrones(directory, target_ages, mlims=[20.,100.], observational_data=None):
	"""
	Plot the HR diagram with isochrones for all available stellar mass files in the directory.

	Parameters:
	- directory: The directory containing the MIST data files.
	- target_ages: List of ages (in log years) for which to plot the isochrones.
	"""
	files = find_all_mass_files(directory)
	mass_labels = []

	plt.figure(figsize=(8, 8))

	massfile = {}
	mass_list = [] 

	for file in files:
		mass = float(file.split('/')[-1].replace('M.track.eep', '')) / 100.0  # Extract mass from filename
		mass_labels.append(mass)
		massfile[mass] = file
		mass_list.append(mass)

	mass_list = np.array(mass_list)
	mass_list = np.sort(mass_list)
	mass_list = mass_list[mass_list <= mlims[1]]
	mass_list = mass_list[mass_list >= mlims[0]]
	while len(mass_list) > 20:
		mass_list = mass_list[::2]

	# Normalize the color map based on the mass range
	norm = mcolors.Normalize(vmin=mass_list.min(), vmax=mass_list.max())
	cmap = cm.get_cmap('viridis')

	# Plot isochrones for each mass (solid lines)
	for mass in mass_list:
		file = massfile[mass]
		data = extract_eep_data(file)
		color = cmap(norm(mass))  # Get color for the mass
		selected_data = data[(data['star_age']>=np.amin(target_ages)*1e6)&(data['star_age']<=np.amax(target_ages)*1e6)]
		plt.plot(selected_data['log_Teff'], selected_data['log_L'], color=color, label=f'Mass = {mass:.2f} M_sun')

	# Plot dotted lines for constant age
	for target_age in target_ages:
		age_label_data = []
		for mass in mass_list:
			file = massfile[mass]
			data = extract_eep_data(file)
			if np.amax(data['star_age'])>=target_age*1e6 and np.amin(data['star_age'])<=target_age*1e6:
				closest_age = find_closest_age(data, target_age * 1e6)
				
				selected_data = data[data['star_age'] == closest_age]
				if len(selected_data) > 0:
					age_label_data.append((selected_data.iloc[0]['log_Teff'], selected_data.iloc[0]['log_L']))
			
		if len(age_label_data) > 1:
			age_label_data = np.array(age_label_data)
			plt.plot(age_label_data[:, 0], age_label_data[:, 1], 'k:', linewidth=1)
			plt.scatter(age_label_data[:, 0], age_label_data[:, 1], color='k', linewidth=1)
			# Add labels to the iso-age lines
			plt.text(age_label_data[-1, 0], age_label_data[-1, 1], f'{target_age:.1f}', fontsize=8, verticalalignment='bottom')


	# Initialize the dictionary to store the best-fit results
	best_fit_dict = {'Object': [] , 'mstar': [], 'mstar_lower': [], 'mstar_upper': [] , 'age': [], 'age_upper': [], 'age_lower': []}

	# Plot the observational data points with error bars
	if observational_data is not None:
		for _, row in observational_data.iterrows():
			plt.errorbar(
				np.log10(row['Teff']), row['logL'], 
				xerr=row['Ter']/row['Teff']/np.log(10), yerr=row['logLerr'], 
				fmt='o', color='red', ecolor='gray', capsize=3
			)
			if np.isnan(row['logLerr']):
				row['logLerr'] = 0.2
			# Find the best-fit mass and age and their ranges
			best_fit, (lower_mass, upper_mass), (lower_age, upper_age) = find_best_fit_mass_age(
				directory, row['logL'], row['logLerr'], row['Teff'], row['Ter'], mass_limits = [10., 100.], age_limits=[0.1, 10.]
			)

			obj = row['Object']
			if len(obj.split('_'))>1:
				obj = row['Object'].split('_')[1]
			# Create a LaTeX formatted label with superscript and subscript for best-fit values
			label_text = (
				f"  {obj}: "
				f"${best_fit[0]:.0f}_{{{lower_mass:.0f}}}^{{{upper_mass:.0f}}}$ $M_{{\odot}}$, "
				f"${best_fit[1]/1e6:.1f}_{{{lower_age/1e6:.1f}}}^{{{upper_age/1e6:.1f}}}$ Myr"
			)

			# Label the observational point with the best-fit mass and age in LaTeX format
			plt.text(
				np.log10(row['Teff']), row['logL'], label_text,
				fontsize=6, verticalalignment='top', horizontalalignment='left', color='r'
			)

			best_fit_dict['Object'].append(row['Object'])
			best_fit_dict['mstar'].append(best_fit[0])
			best_fit_dict['age'].append(best_fit[1]/1e6)

			best_fit_dict['mstar_lower'].append(lower_mass)
			best_fit_dict['mstar_upper'].append(upper_mass)

			best_fit_dict['age_lower'].append(lower_age/1e6)
			best_fit_dict['age_upper'].append(upper_age/1e6)

			df=pd.DataFrame.from_dict(best_fit_dict,orient='index').transpose()
			df.to_csv('mass_age.csv', sep=',', index=False)
	plt.gca().invert_xaxis()  # Effective temperature decreases to the right
	plt.xlabel(r'$\log T_{eff}$')
	plt.ylabel(r'$\log L/L_{\odot}$')

	plt.tick_params(which='both', top=True, left=True, right=True, bottom=True)

	# Add a colorbar for mass
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = plt.colorbar(sm)
	cbar.set_label('Mass (M_sun)')
	plt.xlim([4.7, 4.3])

	plt.ylim([4.0,6.5])
	plt.savefig('isochrone_fit.pdf', bbox_inches='tight', format='pdf'
	)
	#plt.legend()
	plt.show()

	return best_fit_dict


def find_best_fit_mass_age(directory, logL, logL_err, T_eff, T_eff_err, age_limits=None, mass_limits=None):
	"""
	Find all stellar masses and ages consistent within 1-sigma of given luminosity and effective temperature.

	Parameters:
	- directory: The directory containing the MIST data files.
	- L: Measured luminosity.
	- L_err: Uncertainty in measured luminosity.
	- T_eff: Measured effective temperature.
	- T_eff_err: Uncertainty in measured effective temperature.

	Returns:
	- results: A list of tuples, where each tuple is (mass, age) consistent with the provided measurements.
	"""
	files = find_all_mass_files(directory)

	consistent_masses_ages = []
	best_fit = None
	min_chi2 = np.inf


	if age_limits is None:
		age_limits=  [0., np.inf]
	else:
		age_limits = np.array(age_limits)*1e6

	# Loop over each file (each mass) and extract the relevant data
	for file in files:
		data = extract_eep_data(file)
		mass = float(file.split('/')[-1].replace('M.track.eep', '')) / 100.0  # Extract mass from filename
		if not mass_limits is None:
			if mass>mass_limits[1] or mass<mass_limits[0]:
				continue

		consistent_data = data[
			(data['log_L'] >= logL - logL_err) & (data['log_L'] <=logL + logL_err) &
			(data['Teff'] >= T_eff - T_eff_err) & (data['Teff'] <= T_eff + T_eff_err) &
			(data['star_age'] >= age_limits[0]) & (data['star_age'] <= age_limits[1])
		]
		if len(consistent_data)>=1:
			# Calculate chi2 for each point in the track
			chi2 = ((consistent_data['Teff'] - T_eff) / T_eff_err)**2 + \
					((consistent_data['log_L'] - logL) / logL_err)**2
			min_idx = np.argmin(chi2)
			if chi2.iloc[min_idx] < min_chi2:
				min_chi2 = chi2.iloc[min_idx]
				best_fit = (mass, consistent_data.iloc[min_idx]['star_age'])

			
			
			# Collect the consistent mass and age values
			for _, row in consistent_data.iterrows():
				consistent_masses_ages.append((mass, row['star_age']))
		

	if consistent_masses_ages:
		consistent_masses_ages = np.array(consistent_masses_ages)
		lower_mass, upper_mass = np.min(consistent_masses_ages[:, 0]), np.max(consistent_masses_ages[:, 0])
		lower_age, upper_age = np.min(consistent_masses_ages[:, 1]), np.max(consistent_masses_ages[:, 1])
	else:
		lower_mass, upper_mass = best_fit[0], best_fit[0]
		lower_age, upper_age = best_fit[1], best_fit[1]

	return best_fit, (lower_mass, upper_mass), (lower_age, upper_age)


def load_observational_data(file_path):
    """
    Load observational data from the provided file.
    
    Parameters:
    - file_path: Path to the data file.
    
    Returns:
    - df: DataFrame containing Teff, Ter, logL, logLerr for each star.
    """
    column_names = [
        "Object", "ra", "dec", "spt_comb", "Teff", "Ter", "logL", "logLerr",
        "J", "eJ", "H", "eH", "K", "eK", "m36", "em36", "m45", "em45",
        "m58", "em58", "m80", "em80", "Ak", "Aker", "flag", "stage",
        "disPis24_1", "disPis24_2", "disPis24_17"
    ]
    
    # Load the data into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, names=column_names, skiprows=1)
    
    # Extract the relevant columns
    extracted_data = df[["Object", "Teff", "Ter", "logL", "logLerr"]]
    
    return extracted_data

def get_spectra(mstar, age, metallicity=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
	
	# Get stellar properties
	Teff, log_g, log_L, R, star_mass = fetch_stellar_properties(mstar, age*1e6, directory=directory)
	print('log Teff:',np.log10(Teff), mstar, age)
	
	# Compute the stellar spectrum using Castelli & Kurucz atmosphere models
	print(Teff, log_g, log_L, R, star_mass)
	
	atm_mod = 'Castelli & Kurucz 2004'
	try:

		try:
			print('Trying phoenix model...')
			sp = S.Icat('phoenix', Teff, metallicity, log_g)
			atm_mod = 'Phoenix'
		except:
			sp = S.Icat('ck04models', Teff, metallicity, log_g)
	except:
		try:
			print('Trying K93 model... ')
			sp = S.Icat('k93models', Teff, metallicity, log_g)
			atm_mod = 'Kurucz 1993'
		except:
			print('Warning: using blackbody spectrum because stellar parameters outside of atmosphere model range')
			sp = S.BlackBody(Teff)
			atm_mod = 'Blackbody'


		
	#sp = S.Icat('k93models', Teff, metallicity, log_g)
	
	#Renormalize given the stellar luminosity 
	Ltot = np.trapz(sp.flux*np.pi*4.0*R*R*Rsol*Rsol, sp.wave)
	
	Lnorm = Lsol*10.**log_L / Ltot


	return sp.wave, sp.flux*Lnorm, R*Rsol, atm_mod

def compute_luminosity(wave, flux, Rstar, wavelength_start=0.0, wavelength_end = np.inf):
	"""
	Compute the luminosity of the star between given wavelengths.

	Parameters:
	wave (array): Wavelength array in Angstroms.
	flux (array): Flux array in erg/cm^2/s/Å.
	wavelength_start (float): Starting wavelength in Angstroms.
	wavelength_end (float): Ending wavelength in Angstroms.

	Returns:
	float: Luminosity in erg/s.
	"""
	# Mask to select the wavelength range
	mask = (wave >= wavelength_start) & (wave <= wavelength_end)

	# Integrate the flux over the selected wavelength range
	integrated_flux = np.trapz(flux[mask]*4.*np.pi*Rstar*Rstar, wave[mask])

	mean_energy = np.trapz(flux[mask]*4.*np.pi*Rstar*Rstar*(12398.0/wave[mask]), wave[mask])/integrated_flux
	
	# Convert to luminosity (erg/s)
	# Note: The factor of 4πR^2 is already included in the flux normalization, 
	# so we don't need to include it again here.
	luminosity = integrated_flux

	return luminosity, mean_energy*eV

def compute_fuv_euv_luminosities(wave, flux, radius):
	"""
	Compute the FUV luminosity and EUV photon counts.

	Parameters:
	- wave: Wavelength array in Angstroms.
	- flux: Flux array (erg/s/cm^2/A).
	- radius: Stellar radius in cm.

	Returns:
	- FUV_luminosity: FUV luminosity integrated over 912-2000 Angstrom.
	- EUV_photon_counts: EUV photon counts integrated over 10-912 Angstrom.
	"""
	# Define wavelength ranges for FUV and EUV (in Angstroms)
	fuv_range = (912, 2000)
	euv_range = (10, 912)

	# Integrate FUV luminosity (erg/s)
	FUV_luminosity, _ = compute_luminosity(wave, flux, radius, wavelength_start=912, wavelength_end = 2000.0)
	# Integrate EUV luminosity (erg/s)
	EUV_luminosity, mean_e = compute_luminosity(wave, flux, radius, wavelength_start=912, wavelength_end = 2000.0)

	# EUV photon counts (photons/s)
	EUV_photon_counts = EUV_luminosity/mean_e

	return FUV_luminosity, EUV_photon_counts

def compute_spectra_for_table(dataframe, metallicity=0.0, Mdot_acc=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
	"""
	Compute the stellar spectrum, FUV luminosity, and EUV photon counts for each entry in the given DataFrame.

	Parameters:
	- dataframe: A pandas DataFrame containing 'mstar' (stellar mass) and 'age' columns.
	- metallicity: Metallicity value to be passed to the get_spectra function.
	- Mdot_acc: Accretion rate to be passed to the get_spectra function.

	Returns:
	- dataframe: The updated DataFrame with FUV and EUV columns added.
	"""
	fuv_luminosities = []
	euv_photon_counts = []
	atm_models = []

	for index, row in dataframe.iterrows():
		mstar = row['mstar']
		age = row['age']

		# Call the get_spectra function with the appropriate parameters
		wave, flux, radius, atm_mod = get_spectra(mstar, age, metallicity, directory=directory)
		
		# Compute FUV luminosity and EUV photon counts
		FUV_luminosity, EUV_photon_counts = compute_fuv_euv_luminosities(wave, flux, radius)
		
		# Append results to the lists
		fuv_luminosities.append(FUV_luminosity)
		euv_photon_counts.append(EUV_photon_counts)
		atm_models.append(atm_mod)

	# Add FUV and EUV columns to the DataFrame
	dataframe['FUV_luminosity'] = fuv_luminosities
	dataframe['EUV_photon_counts'] = euv_photon_counts
	dataframe['Atmosphere model'] = atm_models


	dataframe.to_csv('mass_age_UV.csv', sep=',', index=False)

	return dataframe

def compute_fluxes_at_coordinate(csv_path, ra_deg, dec_deg, distance_pc):
	"""
	Compute the FUV flux (in G0 units) and EUV counts per cm2 per second at a given RA and Dec.

	Parameters:
	- csv_path: Path to the merged CSV file containing star data.
	- ra_deg: Right Ascension (RA) of the target point in degrees.
	- dec_deg: Declination (Dec) of the target point in degrees.
	- distance_pc: Distance to the target point in parsecs.

	Returns:
	- results_df: DataFrame containing the star names, FUV flux (G0), and EUV counts per cm² per second.
	"""
	# Load the merged data from the CSV file
	df = pd.read_csv(csv_path)

	# Convert input RA and Dec to radians
	target_coord = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, distance=distance_pc*u.pc)

	# Prepare lists to store the results
	star_names = []
	fuv_flux_g0_list = []
	euv_counts_list = []

	for index, row in df.iterrows():
		# Convert RA and Dec from J2000 format to degrees
		star_coord = SkyCoord(row['ra'], row['dec'], unit=(u.hourangle, u.deg))
		separation = star_coord.separation(target_coord).arcsec * (distance_pc * u.pc).to(u.cm)
		physical_separation_cm = separation.value
		
		# Compute FUV flux in G0 units
		fuv_flux_g0 = row['FUV_luminosity'] / (4 * np.pi * physical_separation_cm**2) / 1.6e-3
		
		# Compute EUV counts per cm² per second
		euv_counts = row['EUV_photon_counts'] / (4 * np.pi * physical_separation_cm**2)
		
		# Store the results
		star_names.append(row['Object'])
		fuv_flux_g0_list.append(fuv_flux_g0)
		euv_counts_list.append(euv_counts)

	# Create a DataFrame to store the results
	results_df = pd.DataFrame({
		'Star': star_names,
		'FUV_flux_G0': fuv_flux_g0_list,
		'EUV_counts_per_cm2_s': euv_counts_list
	})

	return results_df


if __name__=='__main__':
	# Example usage
	directory = 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'  
	obs_file_path = 'Pis24_Ostars.dat'
	observational_data = load_observational_data(obs_file_path)
	if not os.path.isfile('mass_age.csv'):
		ma_df = plot_all_isochrones(directory, [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0], mlims=[10., 90.], observational_data=observational_data)
	else:
		ma_df = pd.read_csv('mass_age.csv', delimiter=',', header=0)
	
	print(ma_df)
	if not os.path.isfile('mass_age_UV.csv'):
		maL_df = compute_spectra_for_table(ma_df, directory=directory)
	else:
		maL_df = pd.read_csv('mass_age_UV.csv', delimiter=',', header=0)
	print(maL_df)


	# Load the Pis24_Ostars.dat file (assuming it's space-separated or tab-separated)
	ostars_df = pd.read_csv('Pis24_Ostars.dat', delim_whitespace=True)
	# Merge the two DataFrames based on a common key (e.g., 'Object' or 'Star' name)
	# Assuming the key column in both files is named 'Object'
	merged_df = pd.merge(ostars_df, maL_df, on='Object', how='inner')
	print(merged_df)
	# Save the merged DataFrame to a new CSV file 
	merged_df.to_csv('Pis24_Ostars_wUV.csv', sep=',', index=False)
	# Example usage:
	# Replace the RA, Dec, and distance with your specific values
	ra_deg = 260.0  # Example RA in degrees
	dec_deg = -34.0  # Example Dec in degrees
	distance_pc = 2000.0  # Example distance in parsecs
	results = compute_fluxes_at_coordinate('Pis24_Ostars_wUV.csv', ra_deg, dec_deg, distance_pc)
	print(results)