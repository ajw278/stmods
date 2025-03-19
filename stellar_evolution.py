import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pysynphot as S

from astropy.coordinates import SkyCoord
import astropy.units as u



from definitions import *




def find_max_mass_for_age(directory, target_age):
	"""
	This function finds the maximum stellar mass for which the MIST stellar model reaches the given age.

	Args:
		directory (str): The directory where the MIST stellar model files are stored.
		target_age (float): The age in Myr to check the models against.
		
	Returns:
		max_mass (float): The maximum stellar mass for which the model extends to the target age.
	"""
	max_mass = None
	closest_filename = None

	# Iterate over all files in the directory
	for filename in os.listdir(directory):
		if filename.endswith('.track.eep'):
			try:
				# Extract mass from the filename (assuming the format '00037M.track.eep')
				mass_str = filename.split('M')[0]
				mass = float(mass_str) / 100  # Convert mass string to solar masses

				# Extract the data from the file
				filepath = os.path.join(directory, filename)
				data = extract_eep_data(filepath)

				# Check if the stellar track extends to or beyond the target age
				if data['star_age'].max() >= target_age:
					if max_mass is None or mass > max_mass:
						max_mass = mass
						closest_filename = filename

			except ValueError:
				continue

	if max_mass is None:
		raise Warning(f'No stellar model reaches the age of {target_age} Myr.')

	return max_mass, closest_filename

def find_closest_mass_file(directory, target_mass, tol= 0.5, stmodsdir=STMODS_DIR):
	closest_mass = None
	closest_filename = None

	for filename in os.listdir(stmodsdir+directory):
		if filename.endswith('.track.eep'):
			try:
				mass_str = filename.split('/')[-1].split('M')[0]
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
	if target_age>np.amax(data['star_age']):
		print('Warning: target age > maximum in the grid... using maximum')
		target_age = float(np.amax(data['star_age']))
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

def example_plot(directory, target_mass, stmodsdir=STMODS_DIR):
	closest_filename, closest_mass = find_closest_mass_file(directory, target_mass, stmodsdir=stmodsdir)

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

def fetch_stellar_properties(minit, age_years, directory, stmodsdir=STMODS_DIR):
	if minit>1.4:
		closest_filename, closest_mass = find_closest_mass_file(directory, minit, stmodsdir=stmodsdir)
		
		filepath = os.path.join(stmodsdir+directory, closest_filename)
		eep_data = extract_eep_data(filepath)
		print(filepath)

		Teff, log_g, log_L, R, star_mass = interpolate_stellar_properties(eep_data, age_years)

		print('Mass, age, Teff, logL:', minit, age_years, Teff, log_L, R, star_mass)
		

		return Teff, log_g, log_L, R, star_mass
	else:
	
		# Load the provided CSV file and skip initial header rows
		file_path = stmodsdir+'BHAC15_tracks.csv'
		df_clean = pd.read_csv(file_path, comment='!', delim_whitespace=True)

		# Add appropriate column headers based on the description given
		column_names = [
		    "M/Ms", "log_t", "Teff", "L/Ls", "g", "R/Rs", "log(Li/Li0)", "log_Tc", "log_ROc",
		    "Mrad", "Rrad", "k2conv", "k2rad"
		]
		df_clean.columns = column_names

		
		# Convert the mass column to numeric, forcing errors to NaN (which we will drop)
		df_clean["M/Ms"] = pd.to_numeric(df_clean["M/Ms"], errors='coerce')

		# Drop rows where the mass column has NaN (i.e., non-numeric values were present)
		df_clean_numeric = df_clean.dropna(subset=["M/Ms"])

		
		# Filter the DataFrame for relevant columns and convert mass and age to log space
		df_clean_numeric['log_M/Ms'] = np.log10(df_clean_numeric['M/Ms'])

		# Prepare data for interpolation: extract mass, age, and the columns we want to interpolate
		points = np.array([df_clean_numeric['log_M/Ms'], df_clean_numeric['log_t']]).T
		teff_values = df_clean_numeric['Teff']
		logg_values = df_clean_numeric['g']
		logL_values = df_clean_numeric['L/Ls']
		radius_values = df_clean_numeric['R/Rs']

		log_mass_target = np.log10(minit)
		log_age_target = np.log10(age_years)
		# Perform 2D interpolation using griddata in log space
		interpolated_teff = griddata(points, teff_values, (log_mass_target, log_age_target), method='linear')
		interpolated_logg = griddata(points, logg_values, (log_mass_target, log_age_target), method='linear')
		interpolated_logL = griddata(points, logL_values, (log_mass_target, log_age_target), method='linear')
		interpolated_radius = griddata(points, radius_values, (log_mass_target, log_age_target), method='linear')

		return interpolated_teff, interpolated_logg, interpolated_logL, interpolated_radius

def find_all_mass_files(directory, stmodsdir=STMODS_DIR):
    """
    Find all filenames in the directory that correspond to different initial masses.
    Assumes filenames are in the format 'MIST_mass_X.XX.dat', where X.XX is the mass.
    """
    filenames = []
    for filename in os.listdir(stmodsdir+directory):
        if filename.endswith('.track.eep'):  # Assuming .dat files are the ones to be used
            filenames.append(os.path.join(stmodsdir+directory, filename))
    
    return filenames


def find_closest_age(data, target_age):
    """
    Find the closest available age in the data to the target age.
    """
    return data['star_age'][np.argmin(np.abs(data['star_age'] - target_age))]

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_all_isochrones(directory, target_ages, mlims=[20.,100.], observational_data=None, stmodsdir=STMODS_DIR):
	"""
	Plot the HR diagram with isochrones for all available stellar mass files in the directory.

	Parameters:
	- directory: The directory containing the MIST data files.
	- target_ages: List of ages (in log years) for which to plot the isochrones.
	"""
	files = find_all_mass_files(directory, stmodsdir=stmodsdir)
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
			label_text = ("  %s: $%.0f_{%.0f}^{%.0f}$ $M_{\\odot}$, $%.1f_{%.1f}^{%.1f}$ Myr"%(obj, best_fit[0], lower_mass, upper_mass, best_fit[1]/1e6, lower_age/1e6, upper_age/1e6))

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

			print('Best fit list:', best_fit)
			print(best_fit_dict)

			df=pd.DataFrame.from_dict(best_fit_dict,orient='index').transpose()
			df.to_csv('mass_age.csv', sep=',', index=False)
	plt.gca().invert_xaxis()  # Effective temperature decreases to the right
	plt.xlabel(r'$\log T_{eff}$')
	plt.ylabel(r'$\log L/L_{\odot}$')

	plt.tick_params(which='both', top=True, left=True, right=True, bottom=True)

	# Add a colorbar for mass
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = plt.colorbar(sm, ax=plt.gca())
	cbar.set_label('Mass (M_sun)')
	plt.xlim([4.7, 4.3])

	plt.ylim([4.0,6.5])
	plt.savefig('isochrone_fit.pdf', bbox_inches='tight', format='pdf'
	)
	#plt.legend()
	plt.show()

	return df


def find_best_fit_mass_age(directory, logL, logL_err, T_eff, T_eff_err, age_limits=None,  stmodsdir=STMODS_DIR,mass_limits=None):
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
	files = find_all_mass_files(directory, stmodsdir=stmodsdir)

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

# Planck function for blackbody radiation
def planck_wavelength(wavelength_angstrom, T):
	wavelength_cm = np.asarray(wavelength_angstrom) * 1e-8  # Convert Angstrom to cm
	x = h * c / (wavelength_cm * k_B * T)  # Dimensionless argument h nu / k T

	# Use NumPy's where to apply Rayleigh-Jeans for small x, full Planck otherwise
	B_lambda = (2.0 * k_B * T) / (wavelength_cm**4) * (c**2)
	B_lambda[x>1e-10] = (2.0 * h * c**2) / (wavelength_cm[x>1e-10]**5) * (1.0 / (np.exp(x[x>1e-10]) - 1.0)) 
	
	return B_lambda/1e8   # Convert from per cm to per Angstrom

# Disk temperature profile (simplified thin disk model)
def disk_temperature(r, M_star, R_star, accretion_rate):
	return (3 * G * M_star * accretion_rate / (8 * np.pi * sig_SB * r**3) * (1 - (R_star / r)**0.5))**0.25

# Flux from an accretion disk at a given wavelength
def accretion_disk_flux(wavelength_angstrom, M_star, accretion_rate, R_star, R_outstar=10.0):
	M_star_cgs = M_star * Msol  # Convert stellar mass to grams
	R_star_cgs = R_star * Rsol  # Convert stellar radius to cm
	accretion_rate_cgs = accretion_rate * Msol / year # Convert solar masses/year to g/s

	# Create an array of radii from the stellar surface to the outer disk
	r = np.logspace(np.log10(R_star_cgs), np.log10(R_outstar * R_star_cgs), 1000)
	dr = np.diff(r)
	# Initialize the flux
	total_flux = 0

	# Loop over the radii and calculate the contribution from each disk annulus
	for ir, radius in enumerate(r[1:]):
		T = disk_temperature(radius, M_star_cgs, R_star_cgs, accretion_rate_cgs)  # Temperature at radius
		B_lambda =  planck_wavelength(wavelength_angstrom, T)  # Planck function at wavelength and temperature
		
		dA = 2 * np.pi * radius * dr[ir] # Area of the annulus
		total_flux += B_lambda * dA  # Sum the flux contributions

	
	# Convert from specific intensity to flux by multiplying by 2 pi (integration over solid angle)
	return total_flux  # In units of erg s^-1 A^-1

# Free-fall velocity
def free_fall_velocity(M_star, R_star, r_M):
	return np.sqrt(2 * G * M_star * (1/R_star - 1/r_M))

# Shock temperature
def shock_temperature(v_ff):
	return (3./16.0) * (mu * mH * v_ff**2) / k_B


def column_energy_flux(f, R_star, M_star, Mdot_acc, r_M):
	vff = free_fall_velocity(M_star, R_star, r_M)
	F = 0.5*Mdot_acc*vff**2
	F /= f*4.*np.pi*R_star*R_star
	return F

def coll_temp_approx(Tstar, F):
	Tcol = ((sig_SB*Tstar**4 + F)/sig_SB)**(1./4.)
	return Tcol

# Accretion shock luminosity
def accretion_luminosity(M_star, R_star, M_dot, r_M):
	return (G * M_star * M_dot) / R_star * (1 - R_star / r_M)

# Blackbody emission from the shock region
def shock_flux(wavelength_angstrom, T, area):
	B_lambda = planck_wavelength(wavelength_angstrom, T)
	return B_lambda * area

def magnetospheric_radius(M_star, M_dot, R_star, B=1000.0, xi=0.7):
	mdm = B*R_star**3
	return xi*((mdm**4.)/(4.*G*M_star*M_dot**2))**(1./7.)

# Total accretion shock spectrum approximated as by Mendigutia et al. 2011 (actually reference within MDCH04..)
def accretion_shock_spectrum_approx(M_star_sol, R_star_sol, M_dot_Msolyr, Tstar, wavelength_A, f=0.01):

	M_star = M_star_sol*Msol
	R_star = R_star_sol*Rsol
	M_dot = M_dot_Msolyr*Msol/year

	r_M = magnetospheric_radius(M_star, M_dot, R_star, B=1000.0, xi=0.7)

	print('r_m/r_star', r_M/R_star)

	# Free-fall velocity
	v_ff = free_fall_velocity(M_star, R_star, r_M)

	print('v_ff (km/s)', v_ff/1e5)

	# Shock temperature
	#T_shock = shock_temperature(v_ff)
	F = column_energy_flux(f, R_star,  M_star, M_dot, r_M)

	T_coll = coll_temp_approx(Tstar, F)

	# Luminosity from the shock
	L_acc = accretion_luminosity(M_star, R_star, M_dot, r_M)

	# Filling factor (fraction of the surface covered by accretion)
	A_coll = f * 4 * np.pi * R_star**2

	# Spectrum from the shock as a function of wavelength
	spectrum = shock_flux(wavelength_A, T_coll, A_coll)

	print('Normalisation:', L_acc / np.trapz(spectrum, wavelength_A))
	spectrum *= L_acc / np.trapz(spectrum, wavelength_A)

	return spectrum


def get_spectra(mstar, age, metallicity=0.0, Mdot_acc=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS', stmodsdir=STMODS_DIR,return_wavunits=False):
	
	cwd = os.getcwd()

	os.chdir(stmodsdir)
	print('Input to stellar model spectrum retrieval', mstar, age, metallicity, directory, stmodsdir)
	# Get stellar properties
	Teff, log_g, log_L, R, star_mass = fetch_stellar_properties(mstar, age*1e6, directory, stmodsdir=stmodsdir)
	print('Stmod output:', Teff, log_g, log_L, R, star_mass)
	
	# Compute the stellar spectrum using Castelli & Kurucz atmosphere models
	sp = S.Icat('phoenix', Teff, metallicity, log_g)
	atm_mod = 'Phoenix'
	
	"""try:
		try:
			print('Trying phoenix model...')
			sp = S.Icat('phoenix', Teff, metallicity, log_g)
			atm_mod = 'Phoenix'
		except:
			print('Trying CK04 model...')
			atm_mod = 'Castelli & Kurucz 2004'
			sp = S.Icat('ck04models', Teff, metallicity, log_g)
	except:
		try:
			print('Trying K93 model... ')
			sp = S.Icat('k93models', Teff, metallicity, log_g)
			atm_mod = 'Kurucz 1993'
		except:
			print('Warning: using blackbody spectrum because stellar parameters outside of atmosphere model range')
			sp = S.BlackBody(Teff)
			atm_mod = 'Blackbody'"""


	#sp = S.Icat('k93models', Teff, metallicity, log_g)
	
	#Renormalize given the stellar luminosity 
	Ltot = np.trapz(sp.flux*np.pi*4.0*R*R*Rsol*Rsol, sp.wave)
	
	Lnorm = Lsol*10.**log_L / Ltot

	flux = sp.flux
	facc = 0.0
	if Mdot_acc>0.0:
		facc = accretion_shock_spectrum_approx(star_mass, R, Mdot_acc, Teff, sp.wave, f=0.01)/(4.*np.pi*R*R*Rsol*Rsol)
		"""plt.plot(sp.wave, facc)
		plt.plot(sp.wave, sp.flux*Lnorm)

		plt.plot(sp.wave, sp.flux*Lnorm+facc, color='k')
		plt.xscale('log')
		plt.yscale('log')
		plt.ylim([1e2, 1e7])
		plt.show()"""
		
	
	#print('Wavelength units:', sp.waveunits.name)
	#print('Integrated flux:',Lnorm*Ltot)

	os.chdir(cwd)

	if return_wavunits:
		return sp.wave, flux*Lnorm+facc, R*Rsol, atm_mod, sp.waveunits.name 
	else:
		return sp.wave, flux*Lnorm+facc, R*Rsol, atm_mod

def compute_luminosity(wave, flux, Rstar, wavelength_start=0.0, wavelength_end = np.inf):
	"""
	Compute the luminosity of the star between given wavelengths.

	Parameters:
	wave (array): Wavelength array in Anroms.
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


	# Integrate the flux over the selected wavelength range
	integrated_flux_all = np.trapz(flux*4.*np.pi*Rstar*Rstar, wave)

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
	fuv_range = (912.0, 2000.0)
	euv_range = (10, 912.0)

	# Integrate FUV luminosity (erg/s)
	FUV_luminosity, _ = compute_luminosity(wave, flux, radius, wavelength_start=fuv_range[0], wavelength_end = fuv_range[1])
	# Integrate EUV luminosity (erg/s)
	EUV_luminosity, mean_e = compute_luminosity(wave, flux, radius, wavelength_start=euv_range[0], wavelength_end = euv_range[1])

	# EUV photon counts (photons/s)
	EUV_photon_counts = EUV_luminosity/mean_e

	return FUV_luminosity, EUV_photon_counts

def compute_spectra_for_table(dataframe, metallicity=0.0, Mdot_acc=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS', stmodsdir=STMODS_DIR):
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
		wave, flux, radius, atm_mod = get_spectra(mstar, age, metallicity, Mdot_acc=Mdot_acc, directory=directory,stmodsdir=stmodsdir)
		
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
	
def compute_mstar_FUV_EUV(mstars, metallicity=0.0, age=1.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
	"""
	Compute the stellar spectrum, FUV luminosity, and EUV photon counts for each entry in the given DataFrame.

	"""
	fuv_luminosities = np.zeros(mstars.shape)
	euv_photon_counts = np.zeros(mstars.shape)

	for im, mstar in enumerate(mstars):
		# Call the get_spectra function with the appropriate parameters
		wave, flux, radius, atm_mod = get_spectra(mstar, age, metallicity, directory=directory)
		
		# Compute FUV luminosity and EUV photon counts
		FUV_luminosity, EUV_photon_counts = compute_fuv_euv_luminosities(wave, flux, radius)
		
		# Append results to the lists
		fuv_luminosities[im] = FUV_luminosity
		euv_photon_counts[im] = EUV_photon_counts
	
	np.save('FUV_lum', np.array([ mstars, fuv_luminosities]))
	np.save('EUV_counts', np.array([ mstars, euv_photon_counts]))
	
	return fuv_luminosities, euv_photon_counts
		

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
	separations = []

	for index, row in df.iterrows():
		# Convert RA and Dec from J2000 format to degrees
		star_coord = SkyCoord(row['ra'], row['dec'], unit=(u.hourangle, u.deg))
		# Calculate angular separation
		separation = star_coord.separation(target_coord).arcsec  # in arcseconds

		# Convert angular separation to physical separation (in parsecs)
		# 1 arcsec ≈ 4.84814e-6 radians
		separation_rad = separation * u.arcsec.to(u.rad)
		physical_separation_pc = separation_rad * distance_pc
		physical_separation_cm = physical_separation_pc*u.pc.to(u.cm)
		
		# Compute FUV flux in G0 units
		fuv_flux_g0 = row['FUV_luminosity'] / (4 * np.pi * physical_separation_cm**2) / 1.6e-3
		
		# Compute EUV counts per cm² per second
		euv_counts = row['EUV_photon_counts'] / (4 * np.pi * physical_separation_cm**2)
		
		# Store the results
		star_names.append(row['Object'])
		fuv_flux_g0_list.append(fuv_flux_g0)
		euv_counts_list.append(euv_counts)
		separations.append(physical_separation_pc)

	# Create a DataFrame to store the results
	results_df = pd.DataFrame({
		'Star': star_names,
		'FUV_flux_G0': fuv_flux_g0_list,
		'EUV_counts_per_cm2_s': euv_counts_list,
		'separation_pc': separations
	})

	return results_df


# Calculate FUV flux and EUV counts for each coordinate
def compute_fluxes_for_all_stars_filetype1(Ostars_file , discs_file, distance_pc):

	# Load the Pis24_discs.dat file
	discs_df = pd.read_csv(discs_file, delim_whitespace=True)

	# Convert RA and Dec from J2000 format to degrees
	discs_df['ra_deg'] = discs_df['ra'].apply(lambda x: SkyCoord(x, discs_df['dec'][discs_df['ra'] == x].values[0], unit=(u.hourangle, u.deg)).ra.degree)
	discs_df['dec_deg'] = discs_df['dec'].apply(lambda x: SkyCoord(discs_df['ra'][discs_df['dec'] == x].values[0], x, unit=(u.hourangle, u.deg)).dec.degree)

	# Display the first few rows to verify the conversion
	discs_df[['ra', 'dec', 'ra_deg', 'dec_deg']].head()

	discs_df["FUV_flux_G0"] =  np.nan
	discs_df[ "EUV_flux_cts"] =np.nan
	discs_df["dist_top1_pc"] = np.nan
	discs_df["dist_top2_pc"] = np.nan


	for irow, disc_row in discs_df.iterrows():
		# Extract RA and Dec in degrees
		ra_deg = disc_row['ra_deg']
		dec_deg = disc_row['dec_deg']
		
		# Compute fluxes for this coordinate
		results_df = compute_fluxes_at_coordinate(Ostars_file, ra_deg, dec_deg, distance_pc)
		
		discs_df['FUV_flux_G0'][irow] = results_df['FUV_flux_G0'].sum()
		print(np.asarray(results_df['FUV_flux_G0']))
		imax  = np.argsort(np.asarray(results_df['FUV_flux_G0']))[-1]
		imax2  = np.argsort(np.asarray(results_df['FUV_flux_G0']))[-2]
		discs_df['EUV_flux_cts'][irow] = results_df['EUV_counts_per_cm2_s'].sum()
		discs_df['dist_top1_pc'][irow] = float(results_df['separation_pc'].iloc[imax])
		discs_df['dist_top2_pc'][irow] = float(results_df['separation_pc'].iloc[imax2])

	return discs_df


def plot_fuv_flux_vs_distance(discs_df_all, discs_df_sub1=None, discs_df_sub2=None):
	# Separate the data into two subsets based on the 'Name_region' column
	if discs_df_sub1 is None:
		pis24_df = discs_df_all[discs_df_all['Name_region'].str.contains('Pis24')]
		g353_06_df = discs_df_all[discs_df_all['Name_region'].str.contains('G353-06')]
		g353_07_df = discs_df_all[discs_df_all['Name_region'].str.contains('G353-07')]
	else:
		pis24_df = discs_df_all
		g353_06_df = discs_df_sub2
		g353_07_df = discs_df_sub1

	# Define colors and symbols for the different regions
	region_styles = {
		'Pis24': {'color': 'green', 'marker': 'o'},
		'G353.1+0.6': {'color': 'red', 'marker': 's'},
		'G353.1+0.7': {'color': 'blue', 'marker': 'D'}
	}

	plt.figure(figsize=(5, 4))

	for df, region_name in zip([pis24_df, g353_06_df, g353_07_df], ['Pis24', 'G353.1+0.6', 'G353.1+0.7']):
		color = region_styles[region_name]['color']
		marker = region_styles[region_name]['marker']
		
		plt.scatter([],[], color=color, marker=marker, edgecolor='none', label=region_name)
		for i, row in df.iterrows():
			# Plot the distance to the closest contributor with solid marker
			plt.scatter(row['dist_top1_pc'], row['FUV_flux_G0'], color=color, marker=marker, edgecolor='none')
			
			# Plot the distance to the second closest contributor with hollow marker
			plt.scatter(row['dist_top2_pc'], row['FUV_flux_G0'], facecolors='none', edgecolor=color, marker=marker)
			
			# Connect the two points with a faint line
			plt.plot([row['dist_top1_pc'], row['dist_top2_pc']], 
						[row['FUV_flux_G0'], row['FUV_flux_G0']], 
						color=color, alpha=0.3)
			
			# Add labels
			#plt.text(row['dist_top1_pc']+0.02, row['FUV_flux_G0']*1.05, row['Name'].split('E')[-1], fontsize=8)
			plt.text(row['dist_top1_pc']+0.02, row['FUV_flux_G0']*1.05, row['Object'], fontsize=8)

	dspace = np.logspace(-1.8, 0.5)
	Lthet1C, LEUV = compute_mstar_FUV_EUV(np.array([37.0]), metallicity=0.0, age=1.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS')
	ONC_flux =  Lthet1C/(4.*np.pi*1.6e-3*(dspace*3.06e18)**2)
	# Log scale for the FUV flux axis
	plt.plot(dspace, ONC_flux, color='k', linestyle='dashed', linewidth=1, label='ONC')
	#plt.plot(dspace, 3.*ONC_flux, color='red', linestyle='dashed', linewidth=1, label='ONC x 3')
	plt.plot(dspace, 10.*ONC_flux, color='k', linestyle='dotted', linewidth=1, label='ONC x 10')
	#plt.plot(dspace, 50.*ONC_flux, color='blue', linestyle='dashed', linewidth=1, label='ONC x 50')
	plt.yscale('log')
	plt.xlabel('Distance to source [pc]')
	plt.ylabel('Total FUV flux: $F_\mathrm{FUV}$ [$G_0$]')
	plt.xlim([0., 3.0])
	plt.ylim([1e3, 2e6])
	plt.legend(loc='best', ncol=2, fontsize=8)
	plt.tick_params(which='both', top=True, left=True, bottom=True, right=True)

	# Avoid overlapping labels
	plt.tight_layout()
	plt.savefig('solar_mass_flux.pdf', bbox_inches='tight', format='pdf')

	# Show the plot
	plt.show()


# Calculate FUV flux and EUV counts for each coordinate
def compute_fluxes_for_all_stars_filetype2(ostars_file, discs_file, distance_pc, plot=True):

	discs_df = pd.read_csv(discs_file)

	# Ensure RA and Dec columns are in degrees and named appropriately
	discs_df['FUV_flux_G0'] = np.nan
	discs_df['EUV_flux_cts'] = np.nan

	for irow, disc_row in discs_df.iterrows():
		# Extract RA and Dec in degrees
		ra_deg = disc_row['RA_J2000']
		dec_deg = disc_row['Dec_J2000']
		
		# Compute fluxes for this coordinate
		results_df = compute_fluxes_at_coordinate(ostars_file, ra_deg, dec_deg, distance_pc)
		discs_df.at[irow, 'FUV_flux_G0'] = results_df['FUV_flux_G0'].sum()
		discs_df.at[irow, 'EUV_flux_cts'] = results_df['EUV_counts_per_cm2_s'].sum()

		# Sort the results by FUV_flux_G0 in descending order to find top contributors
		sorted_results = results_df.sort_values(by='FUV_flux_G0', ascending=False)

		# Get distances to the top two contributors, if available
		top1_distance = sorted_results.iloc[0]['separation_pc'] if len(sorted_results) > 0 else np.nan
		top2_distance = sorted_results.iloc[1]['separation_pc'] if len(sorted_results) > 1 else np.nan

		# Assign the distances to the discs DataFrame
		discs_df.at[irow, 'dist_top1_pc'] = top1_distance
		discs_df.at[irow, 'dist_top2_pc'] = top2_distance
	
	plot_fuv_flux_vs_distance(discs_df)

	return discs_df

if __name__=='__main__':
	"""m =  np.logspace(0., 2., 30)
	f, e = compute_mstar_FUV_EUV(m, metallicity=0.0, age=3.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS')
	plt.plot(m, f)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()
	exit()"""
	 
	"""
	
	age = 2.0
	
	directory =   'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'
	mstars  = np.linspace(0.1, 3.0, 50)
	Teffs = np.zeros(mstars.shape)
	Rs = np.zeros(mstars.shape)
	for im, mstar in enumerate(mstars):
		Teff, log_g, log_L, R, star_mass = fetch_stellar_properties(mstar, age*1e6, directory)
		Teffs[im] =Teff
		Rs[im]  = R
	
	np.save('mTR_2Myr.npy', np.array([mstars, Teffs, Rs]))
	
	plt.plot(mstars, Rs)
	plt.show()
	
	plt.plot(mstars, Teffs)
	plt.show()
	exit()"""

	"""age=1.0
	mstars  = np.linspace(0.6, 4.0, 20)
	FUV_max  = np.zeros(mstars.shape)
	FUV_min  = np.zeros(mstars.shape)

	for im, mstar in enumerate(mstars):
		wave, flux, radius, atm_mod = get_spectra(mstar, age, 0.0, Mdot_acc=5e-9 * mstar**2)
		fuv, euv = compute_fuv_euv_luminosities(wave, flux, radius)
		print('%.1lf solar mass FUV  (erg/s): %.2e'%(mstar, fuv))
		print('%.1lf solar mass FUV flux at 10 au (G0): %.2e'%(mstar, fuv/(1.6e-3*4.*np.pi*(100.*1.496e14)**2)))
		FUV_min[im] = fuv/(1.6e-3*4.*np.pi*(1.496e14)**2) 
	
	age=2.0
	for im, mstar in enumerate(mstars):
		wave, flux, radius, atm_mod = get_spectra(mstar, age, 0.0, Mdot_acc=5e-9 * mstar**2)
		fuv, euv = compute_fuv_euv_luminosities(wave, flux, radius)
		print('%.1lf solar mass FUV  (erg/s): %.2e'%(mstar, fuv))
		print('%.1lf solar mass FUV flux at 10 au (G0): %.2e'%(mstar, fuv/(1.6e-3*4.*np.pi*(100.*1.496e14)**2)))
		FUV_max[im] = fuv/(1.6e-3*4.*np.pi*(1.496e14)**2) 

	np.save('non_att_FUV_2Myr', np.array([mstars, FUV_max]))
	np.save('non_att_FUV_1Myr', np.array([mstars,FUV_min] ))

	mstars, FUV_max = np.load('non_att_FUV_2Myr.npy')
	mstars, FUV_min = np.load('non_att_FUV_1Myr.npy')

	#How much flux is attenuated?
	f_att = 0.1

	FUV_max *= f_att
	FUV_min *= f_att


	plt.figure(figsize=(5.,4.))
	plt.yscale('log')
	plt.axvspan(0.8, 1.2, color='lightgreen', alpha=0.2)
	plt.axhspan(2e3, 1e6, color='purple', alpha=0.2)
	plt.axvspan(2.0, 5.0, color='red', alpha=0.2)


	# Add a label to the filled region
	plt.text(1.4,  700.0, 'Our sample', horizontalalignment='center', fontsize=12, color='lightgreen')
	plt.text(3.0,  700.0, 'Existing sample', horizontalalignment='center', fontsize=12, color='red')

	plt.text(2.0,  1e6, 'External FUV range', horizontalalignment='center', fontsize=12, color='purple')


	#plt.plot(mstars, FUV, color='k')
	plt.fill_between(mstars, FUV_min, FUV_max, interpolate=True, color='none', edgecolor='k', alpha=1.0,  hatch='//')


	plt.ylim([500.0, 2e6])
	plt.xlim([0.6, 3.5])
	plt.ylabel('Attenuated FUV flux 1-2 Myr at 100 au [$G_0$]')
	plt.xlabel('Stellar mass [$M_\odot$]')
	plt.tick_params(which='both', top=True, left=True, bottom=True, right=True)
	plt.savefig('FUV_mass.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	
	exit()"""

	redo_massage=False
	redo_massage_UV=True
	directory = 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'  
	obs_file_path = 'Pis24_Ostars.dat'
	observational_data = load_observational_data(obs_file_path)
	if not os.path.isfile('mass_age.csv') or redo_massage:
		ma_df = plot_all_isochrones(directory, [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0], mlims=[10., 90.], observational_data=observational_data)
	else:
		ma_df = pd.read_csv('mass_age.csv', delimiter=',', header=0)
	
	if not os.path.isfile('mass_age_UV.csv') or redo_massage_UV:
		maL_df = compute_spectra_for_table(ma_df, directory=directory)
	else:
		maL_df = pd.read_csv('mass_age_UV.csv', delimiter=',', header=0)

	# Load the Pis24_Ostars.dat file (assuming it's space-separated or tab-separated)
	ostars_df = pd.read_csv('Pis24_Ostars.dat', delim_whitespace=True)
	# Merge the two DataFrames based on a common key (e.g., 'Object' or 'Star' name)
	# Assuming the key column in both files is named 'Object'
	merged_df = pd.merge(ostars_df, maL_df, on='Object', how='inner')
	print(merged_df)
	# Save the merged DataFrame to a new CSV file 
	merged_df.to_csv('Pis24_Ostars_wUV.csv', sep=',', index=False)


	distance_pc = 1690.0  # Kuhn et al. 2019 (not now, updated to XUE value)
	disc_fluxes = compute_fluxes_for_all_stars_filetype1('Pis24_Ostars_wUV.csv', 'Pis24_discs.dat', distance_pc)
	#disc_fluxes = compute_fluxes_for_all_stars_filetype2('Pis24_Ostars_wUV.csv', 'M_0.8_1.2_all_regions.csv', distance_pc)
	disc_fluxes.to_csv('disc_fluxes_Pis24.csv', sep=',', index=False)

	disc_fluxes_BN = compute_fluxes_for_all_stars_filetype1('Pis24_Ostars_wUV.csv', 'BN_discs.dat', distance_pc)
	disc_fluxes_BN.to_csv('disc_fluxes_BN.csv', sep=',', index=False)

	disc_fluxes_N78 = compute_fluxes_for_all_stars_filetype1('Pis24_Ostars_wUV.csv', 'N78_discs.dat', distance_pc)
	disc_fluxes_N78.to_csv('disc_fluxes_N78.csv', sep=',', index=False)

	plot_fuv_flux_vs_distance(disc_fluxes, discs_df_sub1=disc_fluxes_BN, discs_df_sub2=disc_fluxes_N78)



	# Replace the RA, Dec, and distance with your specific values
	"""ra_deg = 260.0  # Example RA in degrees
	dec_deg = -34.0  # Example Dec in degrees
	distance_pc = 1770.0  # Example distance in parsecs
	results = compute_fluxes_at_coordinate('Pis24_Ostars_wUV.csv', ra_deg, dec_deg, distance_pc)
	print(results)"""
