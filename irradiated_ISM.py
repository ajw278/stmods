import numpy as np

import synthetic_cluster as sc
import stellar_evolution as se
import imf_funcs as imff

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




def build_irradiated_ISM(ascale=1.0, sfactor=5.0, age=5.0, mOB_min=20., metallicity=0.0, Mtot=1e5, Mtot_gas=1e6, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'):

	x, y, z, non_irr_ISM = load_ISM_grid()
	x, y, z, non_irr_ISM = centre_and_scale(x, y, z, non_irr_ISM, Lside=ascale*sfactor*2., Mtot=Mtot_gas)

	maxms, _ = se.find_max_mass_for_age(directory, age*1e6)
	print('Maximum mass at %.1lf Myr = %.2lf Msol'%(age,maxms))

	mmean = imff.mean_mass(m_min=0.08, m_max=maxms)

	ntot = Mtot/mmean

	fOB = imff.imf_fraction(mOB_min, maxms, m_min=0.08, m_max=maxms)

	print('Fraction of OB stars:', fOB)

	NOB = int(fOB*ntot)
	print('Number of OB stars:', NOB)

	rstars = sc.generate_plummer_sphere(NOB, ascale)
	rmag = np.linalg.norm(rstars, axis=1)
	rstars = rstars[rmag<sfactor*ascale]

	NOB = len(rstars)

	mstars = imff.sample_imf(mOB_min, maxms, NOB)
	
	Lfuv = np.zeros(mstars.shape)
	Ndeuv = np.zeros(mstars.shape)

	for istar, mstar in enumerate(mstars):
		wave, flux, radius, atm_mod = se.get_spectra(mass, age, metallicity, directory=directory)
		fuv, euv = se.compute_fuv_euv_luminosities(wave, flux, radius)
		Lfuv[istar] = fuv
		Ndeuv[istar] = euv

	



if __name__=='__main__':
	build_irradiated_ISM(ascale=1.0, sfactor=5.0, age=5.0, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS')


