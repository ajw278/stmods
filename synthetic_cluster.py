import numpy as np
import pandas as pd
import stellar_evolution as se
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle

import astropy.units as u

plt.rc('text', usetex=True)

def random_OBstar(massrange=[10., 40.], agerange=[1., 3.], metallicity=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
    mass = np.random.uniform() * (massrange[1] - massrange[0]) + massrange[0]
    age = np.random.uniform() * (agerange[1] - agerange[0]) + agerange[0]
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
	return ra, dec, parallax.value

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

def build_uv_cluster(fname='rsnap_3105.npy', ctype='load', aplum=1.0, nplum=400, nOB=3, massrange=[10., 40.], agerange=[1., 3.], metallicity=0.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):
	if ctype == 'load':
		rstars = np.load(fname)
		tag = '_substruct'
	else:
		rstars = generate_plummer_sphere(nplum, aplum)
		rmag = np.linalg.norm(rstars, axis=1)
		rinc = rmag<np.percentile(rmag, 90.0)
		rstars = rstars[rinc]
		tag = '_plummer'

	num_stars, num_dimensions = rstars.shape
	print(f"Loaded {num_stars} stars with {num_dimensions} dimensions.")

	# Randomly select nOB stars to be OB stars and assign them to positions
	selected_indices = np.random.choice(num_stars, nOB, replace=False)

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
		mass, age, fuv_luminosity, euv_luminosity = random_OBstar(massrange, agerange, metallicity, directory)
		
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
	low_mass_data = {
		'RA': ra[~is_ob_star],
		'Dec': dec[~is_ob_star],
		'x': x[~is_ob_star],
		'y': y[~is_ob_star],
		'z': z[~is_ob_star],
		'Parallax': parallax[~is_ob_star],
		'True_FUV': total_fuv_flux_G0,
		'True_EUV': total_euv_flux
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

	return total_fuv_flux_G0, total_euv_flux, ra.deg, dec.deg, parallax, OB_masses, OB_ages, OB_fuv_luminosities, OB_euv_luminosities, is_ob_star, tag

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
	total_fuv_flux_G0, total_euv_flux, ra, dec, parallax, OB_masses, OB_ages, OB_fuv_luminosities, OB_euv_luminosities, is_ob_star, tag = build_uv_cluster(ctype='load')

	plot_star_cluster_wcs(ra[~is_ob_star], dec[~is_ob_star], total_fuv_flux_G0, ra[is_ob_star], dec[is_ob_star], OB_fuv_luminosities, tag=tag)
