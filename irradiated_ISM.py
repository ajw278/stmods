import numpy as np

import synthetic_cluster as sc
import stellar_evolution as se
import imf_funcs as imff
import os

def load_ISM_grid(datafile='grid-1.npy', infofile='grid-info-1.txt'):

	data = np.load(datafile)

	with open(infofile, 'r') as f:
		info_0 = [float(num) for num in f.readline().split()]

	# Extract information from the info array
	offset_0, box_length_0 = np.array(info_0[:3]), info_0[3]


	# Create meshgrid for physical coordinates
	x_physical = np.linspace(offset_0[0], offset_0[0] + box_length_0, data.shape[0])
	y_physical = np.linspace(offset_0[1], offset_0[1] + box_length_0, data.shape[1])
	z_physical = np.linspace(offset_0[2], offset_0[2] + box_length_0, data.shape[2])


	return x_physical, y_physical, z_physical, data


def centre_and_scale(x, y, z, density, Lside = 3.0, Mtot=1e6):
	x -= np.median(x)
	y -= np.median(y)
	z -= np.median(z)

	Dx = np.amax(x)-np.amin(x)
	x *= Lside/Dx
	Dy = np.amax(y) - np.amin(y)
	Dy *= Lside/Dy
	Dz = np.amax(z) - np.amin(z)
	z *= Lside/Dz

	dx = abs(x[1]-x[0])
	dV = dx*dx*dx

	density *= Mtot/np.sum(density*dV)

	return x, y, z, density



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



pc2cm = 3.086e18
Msol2g = 1.988e33
mH = 1.6738e-24
mu = 2.3



# Helper functions to calculate ionization
def generate_sphere_points(N_res):
    """
    Generate N_res isotropically distributed points on the surface of a unit sphere.
    Uses the Fibonacci lattice method to ensure isotropic distribution.
    """
    indices = np.arange(0, N_res, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/N_res)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.vstack((x, y, z)).T  # N_res x 3 array

def calculate_ionisation_sphere(star_positions, euv_counts, x, y, z, non_irr_ISM_Msolpc3, N_res=10000, alpha_B=2.59e-13, tol=1e-10):
	"""
	Calculate the ionization fraction in the ISM grid based on the EUV photon counts from stars, 
	expanding a sphere around each star.
	"""

	non_irr_ISM = non_irr_ISM_Msolpc3*Msol2g/pc2cm**3


	ionisation_fraction = np.zeros_like(non_irr_ISM)  # Initialize ionization fraction array

	# Sort stars by their EUV photon counts from most to least luminous
	sorted_indices = np.argsort(euv_counts)[::-1]
	star_positions = star_positions[sorted_indices]
	euv_counts = euv_counts[sorted_indices]

	# Generate isotropic points on a unit sphere (this stays constant)
	sphere_points = generate_sphere_points(N_res)

	# Calculate the grid resolution (assumes uniform grid spacing)
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	dz = z[1] - z[0]


	x_edge_min, x_edge_max = x[0] - dx / 2, x[-1] + dx / 2
	y_edge_min, y_edge_max = y[0] - dy / 2, y[-1] + dy / 2
	z_edge_min, z_edge_max = z[0] - dz / 2, z[-1] + dz / 2

	cell_size = np.mean([dx, dy, dz])  # Average cell size for spherical expansion

	# KDTree for efficient neighbor searching
	grid_positions = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T  # (Nx * Ny * Nz, 3)
	tree = cKDTree(grid_positions)  # KDTree for efficient cell finding

	# Loop through each star starting from the most EUV luminous
	for i, (star_pos, euv) in enumerate(zip(star_positions, euv_counts)):
		remaining_photons = np.ones(N_res, dtype=float) * (euv / N_res)  # Initialize photons per point on sphere
		radius = cell_size/2  # Start with a radius equal to one cell size

		while remaining_photons.sum() > 0:
			# Scale sphere points to the current radius
			scaled_sphere_points = sphere_points * radius

			# Calculate positions of the sphere surface points relative to the star position
			sphere_surface_positions = scaled_sphere_points + star_pos

			#Remove all photons outside grid domain
			sphere_indices_outside = np.where(
					(sphere_surface_positions[:, 0] < x_edge_min) | (sphere_surface_positions[:, 0] >= x_edge_max) |
					(sphere_surface_positions[:, 1] < y_edge_min) | (sphere_surface_positions[:, 1] >= y_edge_max) |
					(sphere_surface_positions[:, 2] < z_edge_min) | (sphere_surface_positions[:, 2] >= z_edge_max)
				)[0]
			

			remaining_photons[sphere_indices_outside] = 0.0
			inz = remaining_photons>0.0
			active_sphere = sphere_surface_positions[inz,:]

			# Find the grid cells that contain the surface points
			cell_indices = tree.query(active_sphere, k=1)[1]  # Query nearest grid cell for each sphere points

			# Loop through each unique cell to calculate recombinations and ionization
			unique_cells, counts = np.unique(cell_indices, return_counts=True)
			
			

			"""
			# Plotting in 3D
			# Extract the positions of the unique cells
			cell_positions = np.array([grid_positions[cell_idx] for cell_idx in unique_cells])

			fig = plt.figure(figsize=(10, 8))
			ax = fig.add_subplot(111, projection='3d')

			# Plot cell centers
			ax.scatter(cell_positions[:, 0], cell_positions[:, 1], cell_positions[:, 2], color='blue', label='Cell Centers')

			# Plot sphere surface points
			inz = remaining_photons>0.0
			ax.scatter(sphere_surface_positions[inz, 0], sphere_surface_positions[inz, 1], sphere_surface_positions[inz, 2], 
					color='yellow', alpha=0.6, label='Active Surface Points')
			ax.scatter(sphere_surface_positions[~inz, 0], sphere_surface_positions[~inz, 1], sphere_surface_positions[~inz, 2], 
					color='red', alpha=0.6, label='Passive Surface Points')

			# Add labels and legend
			ax.set_xlabel('X (parsecs)')
			ax.set_ylabel('Y (parsecs)')
			ax.set_zlabel('Z (parsecs)')
			ax.legend()
			plt.title('3D Scatter Plot of Cell Centers and Sphere Surface Points')
			plt.show()"""

			for cell_idx, n_points_in_cell in zip(unique_cells, counts):
				# Extract cell indices
				ix, iy, iz = np.unravel_index(cell_idx, non_irr_ISM.shape)

				# Calculate the boundaries of the cell
				x_min, x_max = x[ix] - dx / 2, x[ix] + dx / 2
				y_min, y_max = y[iy] - dy / 2, y[iy] + dy / 2
				z_min, z_max = z[iz] - dz / 2, z[iz] + dz / 2

				# Find the indices of the sphere points that lie within the cell boundaries
				sphere_indices_in_cell = np.where(
					(sphere_surface_positions[:, 0] >= x_min) & (sphere_surface_positions[:, 0] < x_max) &
					(sphere_surface_positions[:, 1] >= y_min) & (sphere_surface_positions[:, 1] < y_max) &
					(sphere_surface_positions[:, 2] >= z_min) & (sphere_surface_positions[:, 2] < z_max) &
					(remaining_photons > 0.0)
				)[0]

				n_points_in_cell = len(sphere_indices_in_cell)

				sphere_point_photons = remaining_photons[sphere_indices_in_cell]
				total_photons_in_cell = np.sum(sphere_point_photons)

				# Calculate the density and recombination rate in the cell
				density = non_irr_ISM[ix, iy, iz] * (1 - ionisation_fraction[ix, iy, iz])  # Non-ionized gas
				volume = dx * dy * dz * pc2cm**3 # Volume of the cell in pc^3
				n_H = density / (mu * mH)  # Convert density to hydrogen atoms/cm^3


				# Recombination rate in this cell (Case B recombinations)
				recombinations_max = alpha_B * n_H**2 * volume  # Number of recombinations per second
				recombinations = min(recombinations_max, total_photons_in_cell)


				# Initialize the absorbed_photons_per_point array with base_photons
				absorbed_photons_per_point = np.ones(n_points_in_cell, dtype=float) * (recombinations / n_points_in_cell)

				# Check if any points run out of photons before absorbing all recombinations
				iexhaust = (sphere_point_photons-absorbed_photons_per_point)<=0.0
				#How many too few photons are there in the exhausted points?
				n_photons_extra = np.sum(absorbed_photons_per_point[iexhaust] - sphere_point_photons[iexhaust])
				
				while np.sum(iexhaust)>0 and int(np.sum(iexhaust))<n_points_in_cell and n_photons_extra>0.0:

					#How many exhausted points?
					n_points_remain = np.sum(~iexhaust)

					#Add all the excess photons to other points
					add_photons = n_photons_extra*np.ones(n_points_remain, dtype=float)/n_points_remain

					#Set absorbed photons to maximum for sphere points which have been completely absorbed
					absorbed_photons_per_point[iexhaust] =  sphere_point_photons[iexhaust]
					#For all other cells add the remaining photons
					absorbed_photons_per_point[~iexhaust] += add_photons

					#Check again to find out if we still have exhausted points
					iexhaust = (sphere_point_photons-absorbed_photons_per_point)<=0.0

					#How many too few photons are there in the exhausted points?
					n_photons_extra = np.sum(absorbed_photons_per_point[iexhaust] - sphere_point_photons[iexhaust])
				if int(np.sum(iexhaust))==n_points_in_cell:
						absorbed_photons_per_point =  sphere_point_photons

				remaining_photons[sphere_indices_in_cell] -= absorbed_photons_per_point
				remaining_photons[remaining_photons<0.0] = 0.0
				recombinations = np.sum(absorbed_photons_per_point)

				

				# Update the ionization fraction in the cell based on the total absorbed photons
				ionisation_fraction[ix, iy, iz] = max(min(ionisation_fraction[ix, iy, iz] + recombinations / (recombinations_max+1e-10), 1.0), 0.0)

			# Expand the sphere by one cell size
			radius += cell_size

			# Stop if all photons are exhausted
			if remaining_photons.sum() <= 0:
				break

	return ionisation_fraction


def build_irradiated_ISM(ascale=1.0, sfactor=5.0, age=5.0, mOB_min=20., metallicity=0.0, Mtot=1e5, Mtot_gas=1e5, tag ='', directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):

	#Load grid of gas density distribution
	x, y, z, non_irr_ISM = load_ISM_grid()

	#Renormalise to the length/density scale given by parameters
	Lside  = ascale*sfactor*2
	x, y, z, non_irr_ISM = centre_and_scale(x, y, z, non_irr_ISM, Lside=Lside, Mtot=Mtot_gas)

	if not os.path.isfile('OB_stars_irr_ISM_%d_Myr'%age+tag+'.npy'):
		maxms, _ = se.find_max_mass_for_age(directory, age*1e6)
		print('Maximum mass at %.1lf Myr = %.2lf Msol'%(age,maxms))

		#Calculate total number of stars
		mmean = imff.mean_mass(m_min=0.08, m_max=maxms)
		ntot = Mtot/mmean

		#Calculate fraction of IMF in massive stars, and therefore total number
		fOB = imff.imf_fraction(mOB_min, maxms, m_min=0.08, m_max=maxms)

		print('Fraction of OB stars:', fOB)

		NOB = int(fOB*ntot)
		print('Number of OB stars:', NOB)

		rstars = sc.generate_plummer_sphere(NOB, ascale)

		#Cut down to stars just within a spherical region 
		rmag = np.linalg.norm(rstars, axis=1)
		rstars = rstars[rmag<sfactor*ascale]

		NOB = len(rstars)

		#Draw stellar masses in appropriate range
		mstars = imff.sample_imf(mOB_min, maxms, NOB)
		
		Lfuv = np.zeros(mstars.shape)
		Ndeuv = np.zeros(mstars.shape)

		#Loop through stars and calculate FUV luminosity and EUV counts/s from each massive star
		for istar, mstar in enumerate(mstars):
			wave, flux, radius, atm_mod = se.get_spectra(mstar, age, metallicity, directory=directory)
			fuv, euv = se.compute_fuv_euv_luminosities(wave, flux, radius)
			Lfuv[istar] = fuv
			Ndeuv[istar] = euv

		xstars, ystars, zstars = rstars.T

		np.save('OB_stars_irr_ISM_%d_Myr'%age+tag, np.array([xstars, ystars, zstars, mstars, Lfuv, Ndeuv]))
	else:
		xstars, ystars, zstars, mstars, Lfuv, Ndeuv = np.load('OB_stars_irr_ISM_%d_Myr'%age+tag+'.npy')
		rstars = np.array([xstars, ystars, zstars]).T
	
	# Run the ionization calculation
	ionisation_fraction = calculate_ionisation_sphere(rstars, Ndeuv, x, y, z, non_irr_ISM)

	star_z_idx = np.random.choice(np.arange(len(z)))

	plt.figure(figsize=(8, 6))
	plt.imshow(ionisation_fraction[:, :, star_z_idx], extent=[0, Lside, 0, Lside], vmin=0.0, vmax=1.0, origin='lower', cmap='inferno')
	plt.colorbar(label="Ionisation Fraction")
	plt.xlabel("X (parsecs)")
	plt.ylabel("Y (parsecs)")
	plt.show()


		

		



if __name__=='__main__':
	build_irradiated_ISM(ascale=1.0, sfactor=5.0, age=5.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS')


