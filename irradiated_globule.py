import numpy as np
import stellar_evolution as se
import PDR_calcs as PDR
import synthetic_cluster as sc
import irradiated_ISM as ISM
from definitions import *


def pdr_globule(diameter, nH, ambientG0, stmasses=None, stages=None, stdist=0.0, FUV_amb=1.0):

	AV =  ISM.NH_to_AV(diameter*au2cm*nH)
	print('Visual extinction:', AV)

	if not stmasses is None:
		wave, flux = sc.build_composite_spectrum(stages, stmasses)
	
	PDR.run_pdr_model_tcalc(FUV_amb, FUV_amb, stdist, density_profile=None, av_profile=None, wavelength=wave, flux=flux, \
					model_name='globule_ne4_1000', input_filename='pdr_globule_ne4_1000.in', wl_units='Angstrom',  nH=nH, avmax=AV)
	

if __name__=='__main__':


	pdr_globule(1000.0, 1e4, 1.0, stmasses=[65., 60.], stages=[2.5, 2.5])
