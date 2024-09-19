import numpy as np
import pandas as pd
import stellar_evolution as se
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
from scipy.interpolate import interp1d
import imf_funcs as imff


import astropy.units as u

plt.rc('text', usetex=True)

def random_OBstar(massrange=[10., 40.], agerange=[1., 3.], metallicity=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS', mOBfix=None, ageOBfix=None):
	
	if mOBfix is None:
		mass = np.random.uniform() * (massrange[1] - massrange[0]) + massrange[0]
	else:
		mass = mOBfix
	if ageOBfix is None:
		age = np.random.uniform() * (agerange[1] - agerange[0]) + agerange[0]
	else:
		age = ageOBfix
	wave, flux, radius, atm_mod = se.get_spectra(mass, age, metallicity, directory=directory)
	fuv, euv = se.compute_fuv_euv_luminosities(wave, flux, radius)
	return mass, age, fuv, euv

def compute_flux(luminosity, distance_pc):
	"""
	Compute the flux experienced by a star in units of erg cm^-2 s^-1 or counts cm^-2 s^-1.
	Luminosity is given in erg s^-1 or counts s^-1.
	Distance is given in parsecs.
	"""
	distance_cm = distance_pc * 3.086e18  # Convert distance from parsecs to cm
	flux = luminosity / (4 * np.pi * distance_cm**2)
	return flux


def build_composite_spectrum(ages, masses, metallicity=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
	if len(ages)!=len(masses):
		raise Warning('Ages and masses do not have same length')

	total_flux = None
	common_wave = None

	for im, mass in enumerate(masses):
		wave, flux, radius, atm_mod, wavunits= se.get_spectra(mass, ages[im], metallicity, directory=directory, return_wavunits=True)
		
		if common_wave is None:
			common_wave = wave
			total_flux = flux
		else:
			# Interpolate the flux onto the common wavelength grid, assume zero flux outside the star's range
			f_interp_old = interp1d(common_wave, total_flux, bounds_error=False, fill_value=0.0)
			f_interp_new = interp1d(wave, flux, bounds_error=False, fill_value=0.0)
			new_wave = np.append(common_wave, wave)
			new_wave = np.sort(np.unique(common_wave))

			total_flux = f_interp_old(new_wave)
			total_flux += f_interp_new(new_wave)
			common_wave = new_wave
	
	return common_wave, total_flux
	
def build_cluster_spectrum(age, m_min, m_max, N_stars, common_wave=None, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS', nmasses=100, metallicity=0.0):
	"""
	Build the total spectrum for a stellar cluster by summing the spectra of individual stars
	weighted by the number of stars at each mass, based on the IMF.

	Args:
		age (float): The age of the stellar population in Myr.
		metallicity (float): The metallicity of the stellar population.
		m_min (float): Minimum mass of stars in the population.
		m_max (float): Maximum mass of stars in the population.
		N_stars (int): Total number of stars to sample.
		directory (str): Path to the stellar model spectra.

	Returns:
		common_wave (ndarray): Wavelength array for the total spectrum.
		total_flux (ndarray): Total flux for the stellar cluster spectrum.
		bolometric_L (float): Total bolometric luminosity in erg s^-1 .
	"""

	dlogm = 0.05
	mcent = np.linspace(np.log10(m_min), np.log10(m_max), nmasses)
	dlogm = mcent[1]-mcent[0]
	mbound = np.arange(np.log10(m_min)-dlogm/2., np.log10(m_max)+dlogm*1.5, dlogm)
	mcent = 10.**mcent

	dm = np.diff(10.**mbound)

	weight = imff.imf_piecewise(mcent)*dm


	weight /= np.sum(weight)
	weight *= N_stars

	weight = weight[::-1]
	mcent = mcent[::-1]


	bolometric_L = 0.0

	total_flux = None
	common_wave = None

	# Loop through each star
	for im, m_star in enumerate(mcent):
		# Get the spectrum for the star
		wave, flux, radius, atm_mod, wavunits= se.get_spectra(m_star, age, metallicity, directory=directory, return_wavunits=True)
		
		# Weight the flux by the number of stars in this mass bin
		star_weight = weight[im]  # Normalization: each star contributes equally
 
		Lum = np.trapz(4.*np.pi*radius*radius*flux, wave) # Normalise to give the total energy output from the star

		if common_wave is None:
			common_wave = wave
			total_flux = flux*star_weight
			bolometric_L = star_weight * Lum
		else:
			# Interpolate the flux onto the common wavelength grid, assume zero flux outside the star's range
			f_interp_old = interp1d(common_wave, total_flux, bounds_error=False, fill_value=0.0)
			f_interp_new = interp1d(wave, flux, bounds_error=False, fill_value=0.0)
			new_wave = np.append(common_wave, wave)
			new_wave = np.sort(np.unique(common_wave))

			total_flux = f_interp_old(new_wave)
			total_flux += star_weight*f_interp_new(new_wave)
			common_wave = new_wave

			bolometric_L += star_weight * Lum

	return common_wave, total_flux, bolometric_L, wavunits

def assign_galactic_coordinates(rstars, ref_coord):
	"""
	Assigns RA, Dec, and parallax based on a reference coordinate.
	rstars: positions of stars in parsecs relative to the reference position in a Galactocentric frame.
	ref_coord: SkyCoord object with the reference RA, Dec, and distance (parallax).
	"""
	# Define the Galactocentric frame with the standard parameters
	galactocentric_frame = coord.Galactocentric(galcen_distance=8.2 * u.kpc, z_sun=0 * u.pc)

	# Convert the reference ICRS coordinate to the Galactocentric frame
	ref_coord_galactic = ref_coord.transform_to(galactocentric_frame)

	# Get the Cartesian coordinates of the reference point in the Galactocentric frame
	ref_cartesian = ref_coord_galactic.cartesian

	# Ensure rstars are a Quantity with units of parsecs
	rstars_cartesian = rstars * u.pc

	# Add rstars (relative positions) to the reference position in Galactocentric Cartesian coordinates
	x_new = ref_cartesian.x + rstars_cartesian[:, 0]
	y_new = ref_cartesian.y + rstars_cartesian[:, 1]
	z_new = ref_cartesian.z + rstars_cartesian[:, 2]

	# Create a new SkyCoord in the Galactocentric frame from these coordinates
	new_coords_galactic = coord.SkyCoord(x=x_new, y=y_new, z=z_new, representation_type='cartesian', frame=galactocentric_frame)

	# Transform back to the ICRS frame to obtain RA, Dec, and parallax
	new_coords_icrs = new_coords_galactic.transform_to(coord.ICRS())

	# Extract RA, Dec, and parallax
	ra = new_coords_icrs.ra
	dec = new_coords_icrs.dec
	parallax = new_coords_icrs.distance.to(u.pc).to(u.mas, equivalencies=u.parallax())

	return ra.deg, dec.deg, parallax.value

def generate_plummer_sphere(nstar, a):
	"""
	Generates random positions for nstar stars from a Plummer sphere distribution.

	Parameters:
	- nstar: Number of stars to generate.
	- a: Scale parameter of the Plummer sphere.

	Returns:
	- rstars: Array of shape (nstar, 3) containing the x, y, z positions of the stars.
	"""
	# Randomly generate positions in spherical coordinates
	r = a * (np.random.random(nstar) ** (-2/3) - 1) ** (-0.5)
	theta = np.arccos(2 * np.random.random(nstar) - 1)
	phi = 2 * np.pi * np.random.random(nstar)

	# Convert to Cartesian coordinates
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)

	# Combine into a single array of shape (nstar, 3)
	rstars = np.vstack((x, y, z)).T

	return rstars

def calculate_fuv_flux(low_mass_ra, low_mass_dec, low_mass_parallax, ob_ra, ob_dec, ob_parallax, ob_fuv_luminosities):
	"""
	Calculate the FUV flux experienced by low-mass stars based on the positions and luminosities of OB stars.

	Parameters:
	- low_mass_ra, low_mass_dec, low_mass_parallax: RA, Dec, and Parallax of low-mass stars (arrays in degrees and mas).
	- ob_ra, ob_dec, ob_parallax: RA, Dec, and Parallax of OB stars (arrays in degrees and mas).
	- ob_fuv_luminosities: Array of FUV luminosities for OB stars (in erg/s).

	Returns:
	- fuv_flux_low_mass: FUV flux experienced by each low-mass star (array in erg/s/cm^2).
	"""

	print(low_mass_ra)

	# Convert RA, Dec, and Parallax to SkyCoord objects (for low-mass and OB stars)
	low_mass_coords = SkyCoord(ra=low_mass_ra*u.deg, dec=low_mass_dec*u.deg, 
								distance=(1.0 / low_mass_parallax) * u.kpc, frame='icrs')
	

	ob_coords = SkyCoord(ra=ob_ra*u.deg, dec=ob_dec*u.deg, 
							distance=(1.0 / ob_parallax) * u.kpc, frame='icrs')

	# Initialize array to store FUV flux for each low-mass star
	fuv_flux_low_mass = np.zeros(len(low_mass_ra))

	# Loop over each OB star and calculate its contribution to FUV flux for each low-mass star
	for i, (ob_coord, ob_fuv_luminosity) in enumerate(zip(ob_coords, ob_fuv_luminosities)):
		
		print(low_mass_coords)
		# Calculate the distance between the OB star and each low-mass star
		distances = ob_coord.separation_3d(low_mass_coords).to(u.cm)
		print(distances)
		
		# Calculate the FUV flux contribution from this OB star to each low-mass star
		fuv_flux_contribution = (ob_fuv_luminosity / (4 * np.pi * distances.value**2))
		
		# Sum the flux contributions for each low-mass star
		fuv_flux_low_mass += fuv_flux_contribution

	return fuv_flux_low_mass

def plot_density_distributions(rstars, nbins=50):
	"""
	Plots the surface density along the x-axis and density as a function of cylindrical and spherical radius.

	Parameters:
	- rstars: Array of star positions (shape: [nstar, 3]).
	- nbins: Number of bins for the histograms.
	"""
	# Extract x, y, z coordinates
	x, y, z = rstars.T

	# Compute cylindrical radius (R = sqrt(x^2 + y^2)) and spherical radius (r = sqrt(x^2 + y^2 + z^2))
	R = np.sqrt(x**2 + y**2)
	r = np.sqrt(x**2 + y**2 + z**2)

	# 1. Plot the surface density along the x-axis
	plt.figure(figsize=(12, 5))

	# Histogram of star positions along the x-axis
	plt.subplot(1, 3, 1)
	plt.hist(x, bins=nbins, density=True)
	plt.xlabel('x [pc]')
	plt.ylabel('Linear density [stars/pc]')
	plt.title('Density along x-axis')

	# 2. Plot the density as a function of cylindrical radius
	plt.subplot(1, 3, 2)
	R_bins = np.logspace(np.log10(np.amin(R)),  np.log10(np.max(R)), nbins)
	R_hist, _ = np.histogram(R, bins=R_bins)
	R_bin_centers = (R_bins[:-1] + R_bins[1:]) / 2
	annulus_areas = np.pi * (R_bins[1:]**2 - R_bins[:-1]**2)
	R_density = R_hist / annulus_areas

	plt.plot(R_bin_centers, R_density)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim([1e-1, 3.0])
	plt.ylim([1e-1, 300.0])
	plt.xlabel('Cylindrical Radius R [pc]')
	plt.ylabel('Surface density [stars pc$^{-2}$]')
	plt.title('Surface density')

	# 3. Plot the density as a function of spherical radius
	plt.subplot(1, 3, 3)
	r_bins = np.logspace(np.log10(np.amin(r)),  np.log10(np.max(r)), nbins)
	r_hist, _ = np.histogram(r, bins=r_bins)
	r_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
	shell_volumes = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
	r_density = r_hist / shell_volumes

	plt.plot(r_bin_centers, r_density)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim([1e-1, 3.0])
	plt.ylim([1e-1, 300.0])
	plt.xlabel('Spherical Radius r [pc]')
	plt.ylabel('Density [stars pc$^{-3}$]')
	plt.title('Density')

	plt.tight_layout()
	plt.show()

def build_uv_cluster(fname='rsnap_3105.npy', ctype='load', aplum=1.0, nplum=400, nOB=20, mOBfix=None, ageOBfix=None, massrange=[4., 20.], agerange=[1., 3.], metallicity=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
	if ctype == 'load':
		rstars = np.load(fname)
		tag = '_substruct'
	else:
		rstars = generate_plummer_sphere(nplum, aplum)
		rmag = np.linalg.norm(rstars, axis=1)
		rinc = rmag<np.percentile(rmag, 90.0)
		rstars = rstars[rinc]
		tag = '_plummer'
		plot_density_distributions(rstars, nbins=10)

	num_stars, num_dimensions = rstars.shape
	print(f"Loaded {num_stars} stars with {num_dimensions} dimensions.")

	# Randomly select nOB stars to be OB stars and assign them to positions
	selected_indices = np.sort(np.random.choice(num_stars, nOB, replace=False))

	# Create a boolean mask where True indicates an OB star
	is_ob_star = np.zeros(num_stars, dtype=bool)
	is_ob_star[selected_indices] = True

	OB_positions = rstars[is_ob_star]
	non_OB_positions = rstars[~is_ob_star]

	# Initialize arrays to store total fluxes at non-OB stars
	total_fuv_flux = np.zeros(len(non_OB_positions))
	total_euv_flux = np.zeros(len(non_OB_positions))

	# Initialize lists to store the properties of OB stars
	OB_masses = []
	OB_ages = []
	OB_fuv_luminosities = []
	OB_euv_luminosities = []

	for pos in OB_positions:
		mass, age, fuv_luminosity, euv_luminosity = random_OBstar(massrange, agerange, metallicity, directory, mOBfix=mOBfix, ageOBfix=ageOBfix)
		
		# Store OB star properties
		OB_masses.append(mass)
		OB_ages.append(age)
		OB_fuv_luminosities.append(fuv_luminosity)
		OB_euv_luminosities.append(euv_luminosity)
		
		# Compute the distance from this OB star to all non-OB stars
		distances = np.linalg.norm(non_OB_positions - pos, axis=1)
		
		# Compute the FUV and EUV flux from this OB star to all non-OB stars
		fuv_flux = compute_flux(fuv_luminosity, distances)
		euv_flux = compute_flux(euv_luminosity, distances)
		
		# Accumulate the fluxes
		total_fuv_flux += fuv_flux
		total_euv_flux += euv_flux

	# Convert FUV flux to units of G0
	total_fuv_flux_G0 = total_fuv_flux / 1.6e-3

	# Reference coordinates 
	ref_coord = SkyCoord(ra=240.0*u.deg, dec=-40.0*u.deg, distance=140*u.pc, frame='icrs')

	# Assign RA, Dec, and parallax to all stars
	ra, dec, parallax = assign_galactic_coordinates(rstars, ref_coord)

	x, y, z = rstars.T[:]

	# Prepare the non-OB stars data for saving
	# Now we compute the FUV and EUV fluxes using astrometry for non-OB stars
	low_fuv_flux_from_astrometry = calculate_fuv_flux(ra[~is_ob_star], dec[~is_ob_star], parallax[~is_ob_star], ra[is_ob_star], dec[is_ob_star], parallax[is_ob_star], OB_fuv_luminosities)
	low_euv_flux_from_astrometry = calculate_fuv_flux(ra[~is_ob_star], dec[~is_ob_star], parallax[~is_ob_star], ra[is_ob_star], dec[is_ob_star], parallax[is_ob_star], OB_euv_luminosities)
	low_fuv_flux_from_astrometry /= 1.6e-3

	# Prepare the non-OB stars data for saving
	low_mass_data = {
		'RA': ra[~is_ob_star],
		'Dec': dec[~is_ob_star],
		'x': x[~is_ob_star],
		'y': y[~is_ob_star],
		'z': z[~is_ob_star],
		'Parallax': parallax[~is_ob_star],
		'True_FUV': total_fuv_flux_G0,
		'True_EUV': total_euv_flux,
		'FUV_from_astrometry': low_fuv_flux_from_astrometry,
		'EUV_from_astrometry': low_euv_flux_from_astrometry
	}
	low_mass_df = pd.DataFrame(low_mass_data)
	low_mass_df.to_csv('low_mass_stars'+tag+'.csv', index=False)

	# Prepare the OB stars data for saving
	high_mass_data = {
		'RA': ra[selected_indices],
		'Dec': dec[selected_indices],
		'Parallax': parallax[selected_indices],
		'x': x[is_ob_star],
		'y': y[is_ob_star],
		'z': z[is_ob_star],
		'Mass': OB_masses,
		'Age': OB_ages,
		'FUV_Luminosity': OB_fuv_luminosities,
		'EUV_Luminosity': OB_euv_luminosities
	}
	high_mass_df = pd.DataFrame(high_mass_data)
	high_mass_df.to_csv('high_mass_stars'+tag+'.csv', index=False)

	return total_fuv_flux_G0, total_euv_flux, ra, dec, parallax, OB_masses, OB_ages, OB_fuv_luminosities, OB_euv_luminosities, is_ob_star, tag

def plot_star_cluster_wcs(ra, dec, total_fuv_flux_G0, OB_ra, OB_dec, OB_fuv_luminosities, tag=''):
	"""
	Plots a scatter plot of low-mass star locations using a WCS projection, colored by their FUV flux experienced,
	with OB stars marked by star symbols, sized according to the logarithm of their FUV luminosity.

	Parameters:
	- ra, dec: RA and Dec arrays for the low-mass stars (in degrees).
	- total_fuv_flux_G0: Array of FUV flux values experienced by low-mass stars (in G0 units).
	- OB_ra, OB_dec: RA and Dec arrays for the OB stars (in degrees).
	- OB_fuv_luminosities: Array of FUV luminosities for the OB stars.
	"""
	# Initialize a WCS object
	wcs = WCS(naxis=2)
	wcs.wcs.crpix = [0, 0]
	wcs.wcs.cdelt = np.array([-0.1, 0.1])
	wcs.wcs.crval = [np.median(ra), np.median(dec)]
	wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

	# Create a figure with WCS projection
	fig = plt.figure(figsize=(6, 5))
	ax = fig.add_subplot(111, projection=wcs)

	# Scatter plot for low-mass stars, color-coded by log FUV flux
	sc = ax.scatter(ra, dec, c=total_fuv_flux_G0, cmap='viridis', norm=LogNorm(vmin=np.percentile(total_fuv_flux_G0, 10.0), vmax=np.percentile(total_fuv_flux_G0, 90.0)), s=10, alpha=0.7, transform=ax.get_transform('world'))

	# Adding a color bar for the FUV flux in log scale
	cbar = plt.colorbar(sc, ax=ax)
	cbar.set_label('log. FUV flux: $\log F_\mathrm{FUV}$ [$G_0$]')

	# Mark the OB stars with star symbols, sized by log FUV luminosity
	ax.scatter(OB_ra, OB_dec, s=np.log10(OB_fuv_luminosities)*3., c='red', marker='*', edgecolor='black', label='OB Stars', transform=ax.get_transform('world'))

	# Set labels and title
	ax.set_xlabel('RA (J2000)')
	ax.set_ylabel('Dec (J2000)')
	if tag=='_substruct':
		ax.set_title('Sub-structured region')
	else:
		ax.set_title('Plummer sphere')

	# Set ticks for RA in hours, minutes, and seconds
	ax.coords[0].set_major_formatter('hh:mm:ss')
	ax.coords[0].set_axislabel('RA (J2000)')
	ax.coords[1].set_axislabel('Dec (J2000)')

	# Invert the x-axis to follow the standard convention
	ax.invert_xaxis()

	plt.legend()
	plt.savefig('FUV_flux'+tag+'.pdf', format='pdf', bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
	total_fuv_flux_G0, total_euv_flux, ra, dec, parallax, OB_masses, OB_ages, OB_fuv_luminosities, OB_euv_luminosities, is_ob_star, tag = build_uv_cluster(ctype='plummer', nplum=400, nOB=20, mOBfix=15.0, ageOBfix=2.0)

	plot_star_cluster_wcs(ra[~is_ob_star], dec[~is_ob_star], total_fuv_flux_G0, ra[is_ob_star], dec[is_ob_star], OB_fuv_luminosities, tag=tag)

	total_fuv_flux_G0, total_euv_flux, ra, dec, parallax, OB_masses, OB_ages, OB_fuv_luminosities, OB_euv_luminosities, is_ob_star, tag = build_uv_cluster(ctype='load',  nOB=20, mOBfix=15.0, ageOBfix=2.0)

	plot_star_cluster_wcs(ra[~is_ob_star], dec[~is_ob_star], total_fuv_flux_G0, ra[is_ob_star], dec[is_ob_star], OB_fuv_luminosities, tag=tag)