import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


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
		physical_separation_cm = separation.value * 4.84814e-6
		
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


# Calculate FUV flux and EUV counts for each coordinate
def compute_fluxes_for_all_stars(Ostars_file , discs_file, distance_pc):

	# Load the Pis24_discs.dat file
	discs_df = pd.read_csv(discs_file, delim_whitespace=True)

	# Convert RA and Dec from J2000 format to degrees
	discs_df['ra_deg'] = discs_df['ra'].apply(lambda x: SkyCoord(x, discs_df['dec'][discs_df['ra'] == x].values[0], unit=(u.hourangle, u.deg)).ra.degree)
	discs_df['dec_deg'] = discs_df['dec'].apply(lambda x: SkyCoord(discs_df['ra'][discs_df['dec'] == x].values[0], x, unit=(u.hourangle, u.deg)).dec.degree)

	# Display the first few rows to verify the conversion
	discs_df[['ra', 'dec', 'ra_deg', 'dec_deg']].head()

	discs_df["FUV_flux_G0"] =  np.nan
	discs_df[ "EUV_flux_cts"] =np.nan

	for irow, disc_row in discs_df.iterrows():
		# Extract RA and Dec in degrees
		ra_deg = disc_row['ra_deg']
		dec_deg = disc_row['dec_deg']
		
		# Compute fluxes for this coordinate
		results_df = compute_fluxes_at_coordinate(Ostars_file, ra_deg, dec_deg, distance_pc)
		discs_df['FUV_flux_G0'][irow] = results_df['FUV_flux_G0'].sum()
		discs_df['EUV_flux_cts'][irow] = results_df['EUV_counts_per_cm2_s'].sum()

	return discs_df



if __name__=='__main__':

	distance_pc = 2000  #  distance in parsecs
	disc_fluxes = compute_fluxes_for_all_stars('Pis24_Ostars_wUV.csv', 'Pis24_discs.dat', distance_pc)


	disc_fluxes.to_csv('disc_fluxes.csv', sep=',', index=False)


	# Replace the RA, Dec, and distance with your specific values
	"""ra_deg = 260.0  # Example RA in degrees
	dec_deg = -34.0  # Example Dec in degrees
	distance_pc = 2000.0  # Example distance in parsecs
	results = compute_fluxes_at_coordinate('Pis24_Ostars_wUV.csv', ra_deg, dec_deg, distance_pc)
	print(results)"""
