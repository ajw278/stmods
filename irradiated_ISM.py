import numpy as np

import synthetic_cluster as sc
import stellar_evolution as se
import imf_funcs as imff
import os
from utilities import *

import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

import PDR_calcs as PDR

from definitions import *

def compute_FUV_icell(ix, iy, iz, x, y, z, rstars, Lfuv, density, ionisation_fraction, N_points=400, NHAV=1.8e21, AFUVAV=2.5):

	# Position of the current grid cell center
	grid_shape = density.shape

	cell_pos = np.array([x[ix], y[iy], z[iz]])

	# Initialize FUV flux for this cell
	FUV_flux = 0.

	density_cgs = density*Msol2g/(pc2cm**3)

	# Loop over each OB star
	for i_star, (star_pos, fuv_lum) in enumerate(zip(rstars, Lfuv)):
		# Define the line of sight (LOS) from the cell to the star
		distance_vector = star_pos - cell_pos
		distance = np.linalg.norm(distance_vector)

		# Discretize the line into N_points along the LOS
		t = np.linspace(0, 1, N_points)
		line_points = cell_pos[None, :] + t[:, None] * distance_vector[None, :]

		# Convert line points to indices in the grid
		ix_line = np.clip(np.searchsorted(x, line_points[:, 0]) - 1, 0, grid_shape[0] - 1)
		iy_line = np.clip(np.searchsorted(y, line_points[:, 1]) - 1, 0, grid_shape[1] - 1)
		iz_line = np.clip(np.searchsorted(z, line_points[:, 2]) - 1, 0, grid_shape[2] - 1)

		# Calculate the non-ionized density along the line
		density_along_line = density_cgs[ix_line, iy_line, iz_line] * (1 - ionisation_fraction[ix_line, iy_line, iz_line])

		# Integrate the surface density along the LOS (using Simpson's rule)
		dr = distance * pc2cm / N_points
		surface_density = simps(density_along_line, dx=dr)

		# Compute the extinction factor
		A_FUV = surface_density *AFUVAV  / (mu * NHAV * mH)
		extinction = np.exp(-A_FUV)

		# Compute the FUV flux contribution from this star
		FUV_flux += (fuv_lum / (4 * np.pi * (distance*pc2cm)**2)) * extinction
	
	return FUV_flux/1.6e-3

#Default opacity in FUV taken from Draine 2003 -- 10^-21 cm^2/H ~ 260 cm^2 /g (if mu=2.3)
def compute_FUV_field(rstars, Lfuv, ionisation_fraction, x, y, z, density, NHAV=1.8e21, AFUVAV=2.5, N_points=400):
	"""
	Compute the FUV field in the entire grid, taking into account extinction along the path to each OB star.

	Args:
		rstars (ndarray): Positions of the OB stars, shape (N_stars, 3).
		Lfuv (ndarray): FUV luminosity of each OB star, shape (N_stars,).
		ionisation_fraction (ndarray): Ionization fraction grid, same shape as non_irr_ISM.
		x, y, z (ndarray): 1D arrays of grid cell positions.
		non_irr_ISM (ndarray): Non-ionized gas density grid.
		NHAV (float): Extinction per column density of neutral hydrogen (atoms cm^-2 mag^-1)
		AFUVAV (float): Ratio of FUV extinctioon to visual extinction 
		N_points (int): Number of points along the line of sight for integration.

	Returns:
		FUV_grid (ndarray): 3D array representing the FUV field in each grid cell.
	"""
	grid_shape = non_irr_ISM.shape
	FUV_grid = np.zeros(grid_shape)

	# Create 3D meshgrid of cell center positions
	X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

	ngrid = len(X.flatten())
	igrid = 0

	# Loop over each cell in the grid
	for ix in range(grid_shape[0]):
		for iy in range(grid_shape[1]):
			for iz in range(grid_shape[2]):
				FUV_grid[ix, iy, iz] = compute_FUV_icell(ix, iy, iz, x, y, z, rstars, Lfuv, density, ionisation_fraction, N_points=N_points, NHAV=NHAV, AFUVAV=AFUVAV)
				igrid+=1
				if igrid%10000 == 0:
					print('Completed %d/%d grid cell FUV calculation...'%(igrid, ngrid))


	return FUV_grid

#N_H/A_V = 1.8  * 10^21 atoms cm^-2 mag^-1
def compute_fuv_extinction_maps(x, y, z, rstars, Lfuv, ionisation_fraction, density, eps=1e-4, A_V_max=60, N_z_regrid=60, NHAV=1.8e21):
	"""
	Compute the FUV extinction maps and regrid the density structure based on extinction.

	Args:
		x, y, z (ndarray): 1D arrays of grid cell positions.
		rstars (ndarray): Positions of OB stars, shape (N_stars, 3).
		Lfuv (ndarray): FUV luminosity of each OB star, shape (N_stars,).
		ionisation_fraction (ndarray): Ionization fraction grid, same shape as non_irr_ISM.
		non_irr_ISM (ndarray): Non-ionized gas density grid.
		eps (float): Small extinction threshold for first cell.
		A_V_max (float): Maximum extinction allowed.
		N_z_regrid (int): Number of points in the regridded z-direction.
		NHAV (float): Extinction per column density of neutral hydrogen (atoms cm^-2 mag^-1)

	Returns:
		Tuple of 5 2D maps: (A_V_last, FUV_first, FUV_last, z_first, z_last, regridded_density).
	"""
	# Initialize the 2D maps
	A_V_last_map = np.zeros((len(x), len(y)))
	FUV_first_map = np.zeros((len(x), len(y)))
	FUV_last_map = np.zeros((len(x), len(y)))
	z_first_map = np.zeros((len(x), len(y)))
	z_last_map = np.zeros((len(x), len(y)))
	regridded_density = np.zeros((len(x), len(y), N_z_regrid))
	regridded_AV = np.zeros((len(x), len(y), N_z_regrid))

	density_cgs = density*Msol2g/(pc2cm**3)

	dz = z[1] - z[0]

	ngrid = len(x)*len(y)
	igrid = 0

	# Loop over each (x, y) column
	for ix in range(len(x)):
		for iy in range(len(y)):
			A_V = 0.0
			first_found = False
			z_first, z_last = None, None
			density_along_z = []
			AV_along_z = []
			z_positions = []
			iz_last = -1
			iz_first = 0
			# Go through the z-direction (from negative to positive)
			for iz in range(len(z)):
				if ionisation_fraction[ix, iy, iz] > 0.999:
					# Stop when hitting a fully ionized cell
					break

				# Non-ionized density
				non_ionized_density = density_cgs[ix, iy, iz] * max(1. - ionisation_fraction[ix, iy, iz],0.0)

				# Add contribution to A_V
				A_V += non_ionized_density* dz*pc2cm  / (NHAV * mu * mH)

				# Store data once A_V exceeds eps
				if A_V > eps and not first_found:
					iz_first= iz
					A_V =0.0
					first_found = True

				if first_found:
					# Keep storing density and positions for regridding
					density_along_z.append(density[ix,iy,iz])
					AV_along_z.append(A_V)
					z_positions.append(z[iz])

				# If we exceed A_V_max or reach the last z-index, store the last cell info
				if A_V >= A_V_max or iz == len(z) - 1 or (ionisation_fraction[ix, iy, iz]>0.5 and not first_found):
					iz_last = iz
					break
			
			z_last = z[iz_last]
			z_first = z[iz_first]
			z_last_map[ix, iy] = z_last
			z_first_map[ix, iy] = z_first
			FUV_first_map[ix, iy] = compute_FUV_icell(ix, iy, iz_first, x, y, z, rstars, Lfuv, density, ionisation_fraction)
			FUV_last_map[ix, iy] = max(compute_FUV_icell(ix, iy, iz_last, x, y, z, rstars, Lfuv, density, ionisation_fraction), 1.0)
			A_V_last_map[ix, iy] = A_V

			# Regrid the density structure between z_first and z_last if the first cell was found
			if len(z_positions)>1:
				regrid_z = np.linspace(z_first, z_last, N_z_regrid)
				f_interp = interp1d(z_positions, np.log10(density_along_z), kind='linear', fill_value='extrapolate')
				regridded_density[ix, iy, :] = 10.**f_interp(regrid_z)
				f_interp = interp1d(z_positions,AV_along_z, kind='linear', fill_value='extrapolate')
				regridded_AV[ix, iy, :] = f_interp(regrid_z)
			else:
				regridded_density[ix, iy, :] = density[ix, iy, iz_first]
				regridded_AV[ix, iy, :] = np.linspace(0.,A_V, N_z_regrid)
			
			
			igrid+=1
			if igrid%1000==0:
				print('Completed the density map for %d/%d grid columns'%(igrid, ngrid))

	return A_V_last_map, FUV_first_map, FUV_last_map, z_first_map, z_last_map, regridded_density, regridded_AV

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


	return x_physical, y_physical, z_physical, np.exp(data)


def sigma_lnrho(lmbda, sig0=1.0, pl=0.5,cs=0.2):
	sigv = sig0*lmbda**pl
	siglnr = np.sqrt(np.log(1.+ 0.75*(sigv/cs)**2 ))
	return siglnr

def centre_and_scale(x, y, z, density, Lside = 3.0, Mtot=1e6, sig0=1.0, pl=0.5, cs=0.2):
	x -= np.median(x)
	y -= np.median(y)
	z -= np.median(z)

	Dx = np.amax(x)-np.amin(x)
	x *= Lside/Dx
	Dy = np.amax(y) - np.amin(y)
	Dy *= Lside/Dy
	Dz = np.amax(z) - np.amin(z)
	z *= Lside/Dz

	lndensity = np.log(density)
	lndensity -= np.median(density)
	lndensity *= sigma_lnrho(Lside, sig0=sig0, pl=pl,cs=cs)/np.std(lndensity)

	density_new = np.exp(lndensity)

	dx = abs(x[1]-x[0])
	dV = dx*dx*dx

	Mean_density = Mtot/(Lside**3)

	density_new *= Mean_density/np.mean(density_new)

	return x, y, z, density_new




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
		print('i star %d , euv = %.2E'%(i, euv))
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


def find_ionization_front(ionisation_fraction, x, y, z, plot_3d=False):
	"""
	Identify the grid cells on the ionization front surface.

	Args:
		ionisation_fraction (ndarray): The 3D array of ionization fractions.
		x, y, z (ndarray): The 1D arrays of grid cell positions.
		plot_3d (bool): If True, make a 3D scatter plot of the front surface.

	Returns:
		front_indices (ndarray): The indices of the grid cells on the ionization front.
	"""
	# Define the ionization front condition
	front_condition = (ionisation_fraction > 0.001) & (ionisation_fraction < 0.999)

	# Find the indices of the cells that meet this condition
	front_indices = np.argwhere(front_condition)

	# Optionally plot the 3D scatter plot of the ionization front surface
	if plot_3d:
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		
		# Extract the corresponding grid positions for each index
		front_positions_x = x[front_indices[:, 0]]
		front_positions_y = y[front_indices[:, 1]]
		front_positions_z = z[front_indices[:, 2]]
		
		# Plot the scatter plot
		ax.scatter(front_positions_x, front_positions_y, front_positions_z, color='red', s=1, alpha=0.1)
		ax.set_xlabel('X (parsecs)')
		ax.set_ylabel('Y (parsecs)')
		ax.set_zlabel('Z (parsecs)')
		plt.title('3D Scatter Plot of Ionization Front Surface')
		plt.show()

	return front_indices

def get_profile_along_radial_vector(density_cart, ion_frac_cart, x, y, z, r_array, theta, phi):
	"""
	Function to compute density and ionisation fraction profiles along a radial array
	for constant theta and phi.

	Parameters:
	-----------
	density_cart : 3D array
		The Cartesian grid of density values.
	ion_frac_cart : 3D array
		The Cartesian grid of ionisation fraction values.
	x, y, z : 1D arrays
		Cartesian grid coordinates.
	r_array : 1D array
		Array of radial distances to probe.
	theta : float
		Constant polar angle (in radians).
	phi : float
		Constant azimuthal angle (in radians).

	Returns:
	--------
	r_array : 1D array
		Array of radial distances.
	density_profile : 1D array
		Interpolated density profile along the radial direction.
	ion_frac_profile : 1D array
		Interpolated ionisation fraction profile along the radial direction.
	"""

	# Create interpolators for density and ionization fraction
	density_interpolator = RegularGridInterpolator((x, y, z), density_cart)
	ion_frac_interpolator = RegularGridInterpolator((x, y, z), ion_frac_cart)

	# Convert spherical (r_array, theta, phi) to Cartesian coordinates
	cartesian_coords = spherical_to_cartesian(r_array, theta, phi)

	# Interpolate the density and ionisation fraction at the spherical coordinates
	density_profile = density_interpolator(cartesian_coords)
	ion_frac_profile = ion_frac_interpolator(cartesian_coords)

	return r_array, density_profile, ion_frac_profile

def compute_av_profile(density_profile, ion_frac_profile, r_array, NHAV = 1.8e21):
	"""
	Compute the extinction profile and distance to the star, cutting off ionized cells.

	Parameters:
	-----------
	density_profile : 1D array
		The density profile along the radial direction (in Solar masses / pc^3).
	ion_frac_profile : 1D array
		The ionisation fraction profile along the radial direction.
	r_array : 1D array
		Radial distances corresponding to the profiles (in parsecs).

	Returns:
	--------
	r_non_ionized : 1D array
		Radial distances in the non-ionized region.
	A_V_profile : 1D array
		Visual extinction profile across the non-ionized region.
	density_non_ionized : 1D array
		Density profile in the non-ionized region.
	star_distance : float
		The first radial distance at which the ionization fraction drops below 0.5.
	"""

	solar_mass_per_pc3_to_atoms_cm3 = 1.989e33 / (3.086e18)**3 / mu / 1.6735575e-24  # Conversion factor

	# Convert density from Solar masses / pc^3 to number density (atoms / cm^3)
	number_density_profile = density_profile * solar_mass_per_pc3_to_atoms_cm3

	# Step 1: Cut off ionised cells (ionisation fraction > 0.5)
	i_first_ni = np.where(ion_frac_profile<0.5)[0][0]
	r_non_ionized = r_array[i_first_ni:]
	density_non_ionized = density_profile[i_first_ni:]
	number_density_non_ionized = number_density_profile[i_first_ni:]

	# Step 2: Determine the first non-ionised radius (distance to the star)
	if np.sum(ion_frac_profile<0.5)>0:
		star_distance = r_non_ionized[0]
	else:
		star_distance = np.nan  # If all cells are ionised

	# Step 3: Compute the visual extinction A_V across the non-ionized grid
	# A_V is proportional to the column density, which is the integral of number density
	A_V_profile = np.zeros_like(r_non_ionized)
	for i in range(1, len(r_non_ionized)):
		# Approximate the column density as the integral of n_H * dr
		dr = r_non_ionized[i] - r_non_ionized[i-1]
		N_H = number_density_non_ionized[i-1] * dr * 3.086e18  # Convert dr from parsecs to cm
		A_V_profile[i] = A_V_profile[i-1] + N_H / NHAV

	return r_non_ionized, A_V_profile, number_density_non_ionized, star_distance



def calc_temperature(density, ionisation_fraction, x, y, z, wave, flux, wl_units='angstrom', rho_floor=1e-6):

	rgrid = np.linspace(0., np.amax(x), 1000)
	theta = np.pi/3.
	phi = np.pi/3.

	i_ionised = ionisation_fraction>0.5

	X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

	xcent = np.median(X[i_ionised])
	ycent = np.median(Y[i_ionised])
	zcent = np.median(Z[i_ionised])	
	x -= xcent
	y -= ycent
	z -= zcent


	r_arr, rho_arr, if_arr = get_profile_along_radial_vector(density, ionisation_fraction, x, y, z, rgrid, theta, phi)
	rho_arr *= (1.-if_arr)
	rho_arr[rho_arr<rho_floor] = rho_floor



	r_prof, A_V_prof, nH_prof, star_distance = compute_av_profile(rho_arr, if_arr, r_arr)

	"""print(star_distance)

	plt.plot(r_arr, rho_arr)
	plt.plot(r_prof, nH_prof)
	plt.plot(r_arr, if_arr)
	plt.plot(r_prof, A_V_prof)
	plt.yscale('log')
	plt.show()"""


	if not star_distance is None:
		PDR.run_pdr_model_tcalc(1.0, 1.0, star_distance, nH_prof, A_V_prof, wave, flux, 
					model_name='test_pdr_model', input_filename='pdr.in', wl_units=wl_units)

	exit()
	



def build_irradiated_ISM(ascale=1.0, sfactor=5.0, age=5.0, mOB_min=20., metallicity=0.0, Mtot=1e5, Mtot_gas=1e4, tag_1 ='', tag_2='', N_res=50000, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):


	if not os.path.isfile('xyz'+tag_2+'.npy') or not os.path.isfile('density'+tag_2+'.npy'):
		#Load grid of gas density distribution
		x, y, z, non_irr_ISM = load_ISM_grid()

		#Renormalise to the length/density scale given by parameters
		Lside  = ascale*sfactor*2
		x, y, z, non_irr_ISM = centre_and_scale(x, y, z, non_irr_ISM, Lside=Lside, Mtot=Mtot_gas)
		np.save('xyz'+tag_2, np.array([x,y,z]))
		np.save('density'+tag_2, non_irr_ISM)
	else:
		x, y, z = np.load('xyz'+tag_2+'.npy')
		non_irr_ISM = np.load('density'+tag_2+'.npy')

	dL = x[1]-x[0]
	print('Median density:', np.median(non_irr_ISM)*Msol2g/(pc2cm**3))

	maxms, _ = se.find_max_mass_for_age(directory, age*1e6)
	#Calculate total number of stars
	mmean = imff.mean_mass(m_min=0.08, m_max=maxms)
	ntot = Mtot/mmean
	if not os.path.isfile('OB_stars_irr_ISM_%d_Myr'%age+tag_1+'.npy'):
		
		print('Maximum mass at %.1lf Myr = %.2lf Msol'%(age,maxms))

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

		np.save('OB_stars_irr_ISM_%d_Myr'%age+tag_1, np.array([xstars, ystars, zstars, mstars, Lfuv, Ndeuv]))
	else:
		xstars, ystars, zstars, mstars, Lfuv, Ndeuv = np.load('OB_stars_irr_ISM_%d_Myr'%age+tag_1+'.npy')
		rstars = np.array([xstars, ystars, zstars]).T
	
	# Run the ionization calculation 
	iiter= 0 
	iiter_max = 1
	while iiter<iiter_max:
		if not os.path.isfile('ion_front_%d'%iiter + tag_1+tag_2 +'.npy'):
				ionisation_fraction = calculate_ionisation_sphere(rstars, Ndeuv, x, y, z, non_irr_ISM, N_res=N_res)
				np.save('ion_front_%d'%iiter+tag_1+tag_2, ionisation_fraction)
		else:
				ionisation_fraction = np.load('ion_front_%d'%iiter + tag_1+tag_2 +'.npy')

		iiter+=1

	"""iifront = find_ionization_front(ionisation_fraction, x, y, z, plot_3d=True)

	mid_idz = len(z)//2

	plt.figure(figsize=(8, 6))
	dl = 0.1
	plt.contourf(x, y, ionisation_fraction[:, :, mid_idz],  levels=np.arange(0., 1.+dl, dl), origin='lower', cmap='inferno')
	plt.scatter(xstars, ystars, color='b', marker='*', s=5, edgecolor='k')
	plt.colorbar(label="Ionisation Fraction")
	plt.xlabel("X (parsecs)")
	plt.ylabel("Y (parsecs)")
	plt.show()"""

	"""
	if not os.path.isfile('FUV_map'+tag_1+tag_2+'.npy'):
		FUV_map = compute_FUV_field(rstars, Lfuv, ionisation_fraction, x, y, z, non_irr_ISM)
		np.save('FUV_map'+tag_1+tag_2, FUV_map)
	else:
		FUV_map = np.load('FUV_map'+tag_1+tag_2+'.npy')"""

	"""if not os.path.isfile('ext_maps'+tag_1+tag_2+'.npy') or not os.path.isfile('dense_regrid'+tag_1+tag_2+'.npy'):
		A_V_last_map, FUV_first_map, FUV_last_map, z_first_map, z_last_map, regridded_density, regridded_AV = compute_fuv_extinction_maps(
    x, y, z, rstars, Lfuv, ionisation_fraction, non_irr_ISM)
		np.save('ext_maps'+tag_1+tag_2, np.array([A_V_last_map, FUV_first_map, FUV_last_map, z_first_map, z_last_map]))
		np.save('dense_regrid'+tag_1+tag_2, regridded_density, regridded_AV)
	else:
		A_V_last_map, FUV_first_map, FUV_last_map, z_first_map, z_last_map = np.load('ext_maps'+tag_1+tag_2+'.npy')
		regridded_density, regridded_AV = np.load('dense_regrid'+tag_1+tag_2+'.npy')"""
	
	wavunits = 'angstrom'
	if not os.path.isfile('sed_tot_'+wavunits+'_'+tag_1+tag_2+'.npy') or not os.path.isfile('bol_lum'+tag_1+tag_2+'.npy'):
		wav, flux, Ltot, wavunits_= sc.build_cluster_spectrum(age, 0.1, maxms, int(ntot))
		if wavunits!=wavunits_:
			raise Warning('Wavelength units do not match.')
		#Flux has units erg cm^-2 s^-1 A^-1 sr^-1 
		#wav has units A
		#Ltot has units erg s^-1
		np.save('sed_tot_'+wavunits+'_'+tag_1+tag_2, np.array([wav, flux]))
		np.save('bol_lum'+tag_1+tag_2, np.array([Ltot]))

	else:
		wav, flux = np.load('sed_tot_'+wavunits+'_'+tag_1+tag_2+'.npy')
		Ltot = np.load('bol_lum'+tag_1+tag_2+'.npy')
	
	calc_temperature(non_irr_ISM, ionisation_fraction, x, y, z, wav, flux, wl_units='angstrom')
	
	print("%.2e"%Ltot)


	plt.plot(wav, flux)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()
	


	
	isel_x = 150
	isel_y = 150

	
	plt.figure(figsize=(5, 4))
	dA = 0.2
	FUV_first_map[FUV_first_map<FUV_last_map] =FUV_last_map[FUV_first_map<FUV_last_map] 
	FUV_first_map[FUV_first_map<1.0] =1.0
	FUV_first_map[A_V_last_map<0.01] = -1.0
	ctf = plt.contourf(x, y, np.log10(FUV_first_map),  levels=np.arange(0., 3.6+dA, dA), origin='lower', cmap='inferno')
	plt.scatter(x[isel_x], y[isel_y], color='pink', s=60)
	plt.scatter(xstars, ystars, color='cyan', marker='*', s=30)
	plt.colorbar(ctf,label="log. FUV flux [$G_0$]")
	plt.xlabel("X (parsecs)")
	plt.ylabel("Y (parsecs)")
	plt.show()

	plt.figure(figsize=(5, 4))
	dA = 1.0
	ctf = plt.contourf(x, y, A_V_last_map,  levels=np.arange(0., 20.0+dA, dA), origin='lower', cmap='inferno')
	plt.scatter(xstars, ystars, color='cyan', marker='*', s=30)
	plt.scatter(x[isel_x], y[isel_y], color='pink', s=60)
	plt.colorbar(ctf,label="Final visual extinction")
	plt.xlabel("X (parsecs)")
	plt.ylabel("Y (parsecs)")
	plt.show()



	


	
		



if __name__=='__main__':
	
	fgas = 1.0
	Mstars=3.2e4
	Mgas = Mstars*fgas
	
	build_irradiated_ISM(ascale=0.44, sfactor=2.5/0.44, age=5.0, N_res=50000, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS', Mtot=Mstars,  Mtot_gas=Mgas, tag_1='_Wd1', tag_2='_hr_M_%.1e'%Mgas)


