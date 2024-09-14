import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Define the power-law indices
alpha_1 = 1.3  # for m < 0.5 M_sun
alpha_2 = 2.3  # for 0.5 <= m < 1.0 M_sun
alpha_3 = 2.7  # for m >= 1.0 M_sun

# Mass boundaries for the piecewise function
m_boundary1 = 0.5  # Boundary between first and second segment (in solar masses)
m_boundary2 = 1.0  # Boundary between second and third segment (in solar masses)

# Normalization constant (set for first segment and propagate for continuity)
A1 = 1  # We can set A1 arbitrarily as 1 for the first segment

# Continuity at m = 0.5
A2 = A1 * (m_boundary1 ** (alpha_2 - alpha_1))  # Continuity at 0.5 M_sun

# Continuity at m = 1.0
A3 = A2 * (m_boundary2 ** (alpha_3 - alpha_2))  # Continuity at 1.0 M_sun

# Define the piecewise IMF function
def imf_piecewise(m):
	m = np.array(m, ndmin=1)  # Ensure m is treated as an array
	result = np.zeros_like(m)

	# Apply the piecewise conditions
	mask1 = m < m_boundary1
	mask2 = (m >= m_boundary1) & (m < m_boundary2)
	mask3 = m >= m_boundary2

	result[mask1] = A1 * m[mask1] ** -alpha_1
	result[mask2] = A2 * m[mask2] ** -alpha_2
	result[mask3] = A3 * m[mask3] ** -alpha_3

	# If m was a scalar, return a scalar result
	return result.item() if result.size == 1 else result

# Function to calculate the fraction of IMF between two masses
def imf_fraction(m1, m2, m_min=0.08, m_max=50.0):
	# Integrate the IMF from m1 to m2
	numerator, _ = quad(imf_piecewise, m1, m2)

	# Integrate the IMF from the minimum to the maximum range
	denominator, _ = quad(imf_piecewise, m_min, m_max)

	# Return the fraction of stars between m1 and m2
	return numerator / denominator


def mean_mass(m_min=0.08, m_max=50.0):
	# Function to calculate the weighted IMF for the numerator of the average mass calculation
	def weighted_imf(m):
		return m * imf_piecewise(m)

	# Calculate the numerator (integral of m * IMF)
	numerator_avg_mass, _ = quad(weighted_imf, m_min, m_max)

	# Calculate the denominator (integral of IMF, which we've already done as part of normalization)
	denominator_avg_mass, _ = quad(imf_piecewise, m_min, m_max)

	# Calculate the average mass
	return numerator_avg_mass / denominator_avg_mass

# Cumulative distribution function (CDF)
def imf_cdf(m, m_min, m_max):
	# Integrate IMF from m_min to m to get the CDF
	integral_value, _ = quad(imf_piecewise, m_min, m)
	normalization, _ = quad(imf_piecewise, m_min, m_max)
	return integral_value / normalization

# Inverse CDF function (interpolated)
def inverse_cdf(m_min, m_max, num_points=1000):
	masses = np.logspace(np.log10(m_min), np.log10(m_max), num_points)
	cdf_values = np.array([imf_cdf(m, m_min, m_max) for m in masses])
	return interp1d(cdf_values, masses, bounds_error=False, fill_value=(m_min, m_max))

# Sampling function
def sample_imf(m1, m2, N):
	inv_cdf = inverse_cdf(m1, m2)
	# Draw N uniform random numbers between 0 and 1
	random_values = np.random.uniform(0, 1, N)
	# Map uniform random numbers to stellar masses using the inverse CDF
	return inv_cdf(random_values)
