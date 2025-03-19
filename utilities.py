import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Function to convert spherical to Cartesian
def spherical_to_cartesian(r, theta, phi):
	x_cartesian = r * np.sin(theta) * np.cos(phi)
	y_cartesian = r * np.sin(theta) * np.sin(phi)
	z_cartesian = r * np.cos(theta)
	return np.array([x_cartesian, y_cartesian, z_cartesian]).T


def interpolate_cart2sph(val_cart, x, y, z, r, theta, phi, interpolator=None):

	if interpolator is None:		
		# Create the interpolator function for the 3D density field
		interpolator = RegularGridInterpolator((x, y, z), val_cart)

	# Convert to Cartesian coordinates
	cartesian_coords = spherical_to_cartesian(r, theta, phi)

	# Interpolate the density at the spherical coordinates
	val_sph = interpolator(cartesian_coords)

	return interpolator, val_sph

# Function to convert Cartesian to Spherical
def cartesian_to_spherical(x, y, z):
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arccos(z / r)
	phi = np.arctan2(y, x)
	return r, theta, phi

def interpolate_sph2cart(val_sph, r, theta, phi, x, y, z, interpolator=None):

	if interpolator is None:
		# Create the interpolator function for the 3D spherical field
		interpolator = RegularGridInterpolator((r, theta, phi), val_sph)

	# Convert Cartesian coordinates to Spherical coordinates
	spherical_coords = cartesian_to_spherical(x, y, z)

	# Interpolate the value at the Cartesian coordinates
	val_cart = interpolator(spherical_coords)

	return interpolator, val_cart
