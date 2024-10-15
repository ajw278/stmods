import numpy as np
import stellar_evolution as se
import PDR_calcs as PDR
import synthetic_cluster as sc
import irradiated_ISM as ISM
import process_PDRlines as PDRplot
from definitions import *
import os
import pickle 
import pandas as pd


import matplotlib.pyplot as plt

def pdr_globule(diameter, nH, ambientG0, stmasses=None, stages=None, stdist=0.0, FUV_amb=1.0, modname='test', pdr_cdir=PDR_CDIR, plot=True, fetch_spect=True):

	AV =  ISM.NH_to_AV(diameter*au2cm*nH)
	print('Visual extinction:', AV)

	cwd = os.getcwd()
	if not os.path.isdir('pdr_'+modname):
		os.makedirs('pdr_'+modname)
	
	os.chdir('pdr_'+modname)
	if not os.path.isfile('pdr_out.out'):
		if not stmasses is None:
			wave, flux = sc.build_composite_spectrum(stages, stmasses)
		
		PDR.run_pdr_model_tcalc(FUV_amb, FUV_amb, stdist, density_profile=None, av_profile=None, wavelength=wave, flux=flux, \
						model_name=modname, input_filename='pdr_'+modname+'.in', wl_units='Angstrom',  nH=nH, avmax=AV, pdr_cdir=pdr_cdir)
	else:
		print('Output file for model %s already found, skipping PDR calcs...'%modname)


	os.chdir(cwd)

	print('FETCH SPECT', fetch_spect)
	if fetch_spect:
		datdir = pdr_cdir+'/out/'+modname

		hdf5_file = modname+ '_s_20.hdf5'
		linefile, linedf = PDRplot.get_line_intensities(0.01, 100.0, hdf5_file, datdir=datdir, reset=True)
		spectfile, specdf = PDRplot.get_spectrum(hdf5_file, datdir=datdir)
	
		if plot:
			PDRplot.plot_spectrum(spectfile, line_file=linefile)
	
		print(specdf)
		return spectfile, specdf

	return None


def run_globgrid(ambientG0=1.0, diameter=1000.0, stdist=-0.0375):
	model_dict = {'globule_ne8_1000_0.0375': 1e8, 'globule_ne7_1000_0.0375': 1e7,'globule_ne6_1000_0.0375': 1e6,  'globule_ne5_1000_0.0375': 1e5,'globule_ne4_1000_0.0375': 1e4}
	model_dict = {'globule_ne7_1000_0.0375': 1e7, 'globule_ne4_1000_0.0375': 1e4}
	#model_dict = {'globule_ne3_1000_0.0375': 1e3, 'globule_ne7_1000_0.0375': 1e7


	
	fig, ax = plt.subplots(figsize=(6, 5))


	df = PDRplot.get_molline_transitions('O', outfile='every_line.dat', reset=False)
	print(df)
	for wv in df['Wavelength_micron']:
		plt.axvline(wv, color='r', linewidth=0.1)

	for mod in model_dict:
		if not os.path.isfile(mod+'_flux.npy') or True:
			print('Running PDR calculation for %s'%mod)
			spf, data = pdr_globule(diameter, model_dict[mod], ambientG0,  stmasses=[65., 60.], stages=[2.5, 2.5], stdist=stdist, FUV_amb=1.0, modname=mod, plot=False, fetch_spect=True)

			# Assign column names based on the header information
			data.columns = ['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)', 'Wavelength (Å)']

			# Converting Wavelength from Ångström to microns and renormalizing Emerging intensity accordingly
			data['Wavelength (µm)'] = data['Wavelength (Å)'] * 1e-4  # 1 Å = 1e-4 µm
			data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)'] = data['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)'] * 1e4  # Conversion factor from per Ångström to per micron

			# Filtering the data to include only the range between 1 and 25 microns
			filtered_data =data # data[(data['Wavelength (µm)'] >= 1) & (data['Wavelength (µm)'] <= 25)]

			wl = filtered_data['Wavelength (µm)']
			flux = filtered_data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)']
			np.save(mod+'_flux', np.array([wl, flux]))
		else:
			wl, flux = np.load(mod+'_flux.npy')
		nH = model_dict[mod]
		mdot = nH*mH*mu*4.*np.pi*3.0*1e5*(diameter*au2cm/2.6)**2
		mdot /= Msol2g/year
		Mtot = nH*mH*mu*4.*np.pi*(diameter*au2cm/2.6)**3 / 3.0
		Mtot /= Msol2g
		tdep = Mtot/mdot
		tdep /= year
		lab = '$n_\mathrm{H} = 10^{%d}$ cm$^{-3}$, $\dot{M}_\mathrm{ext} \sim %.2E \,M_\odot$ yr$^{-1}$'%(np.log10( model_dict[mod]), mdot)
		plt.plot(wl, flux, lw=1, label=lab)

	plt.xscale('log')
	plt.yscale('log')

	#plt.xlim([1., 25.])
	plt.legend(loc=4, fontsize=8)
	ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
	plt.xlabel('Wavelength [$\mu$m]')
	plt.ylabel('Flux [erg cm$^{-2}$ s$^{-1}$ $\mu$m$^{-1}$ sr$^{-1}$]')
	plt.savefig('quick_globule.pdf', bbox_inches='tight', format='pdf')
	plt.show()


def calc_mdot(R_IF_au, dOB, mOB, tOB, metallicity=0.0):
	# Call the get_spectra function with the appropriate parameters
	print('Mas ages 1:', mOB, tOB)
	wave, flux, radius, atm_mod = se.get_spectra(mOB, tOB, metallicity=metallicity)
		
	# Compute FUV luminosity and EUV photon counts
	FUV_luminosity, EUV_photon_counts = se.compute_fuv_euv_luminosities(wave, flux, radius)
	print(EUV_photon_counts)

	Mdot = 1e-8 * (R_IF_au/1200.)**1.5 * (1./dOB) * np.sqrt(EUV_photon_counts/1e45)

	return Mdot

def mdot_density(Mdot, R_IF_au, cs_I=3.0, cs_II=10.0):
	nI = Mdot*Msol2g/year/(4.*np.pi*(R_IF_au*au2cm)**2 * mH *mu * (1e5*cs_I)**2 / (2.*1e5*cs_II))
	return nI


def run_proplydgrid(ambientG0=1.0, diameter=1000.0, stmass=35.0, stage=1.0, plotsave=True):

	model_dict = {'proplyd_163-249': {'dOB': 0.07, 'DIF': 215.}, \
	'proplyd_176-543': {'dOB': 0.27, 'DIF': 525.}, \
	'proplyd_171-340': {'dOB': 0.04, 'DIF': 324.},\
	'proplyd_218-339': {'dOB': 0.16, 'DIF': 189.},\
	'proplyd_170-249': {'dOB': 0.07, 'DIF': 423.},\
	'proplyd_187-314': {'dOB': 0.07, 'DIF': 252.},\
	'proplyd_175-251': {'dOB': 0.07, 'DIF': 198.},\
	'proplyd_170-337': {'dOB': 0.03, 'DIF': 513.},\
	'proplyd_189-329': {'dOB': 0.07, 'DIF': 153.},\
	'proplyd_191-350': {'dOB': 0.09, 'DIF': 414.}}


	if not os.path.isfile('proplyd_dict.pkl'):
		with open('proplyd_dict.pkl', 'wb') as f:
			for mod in model_dict:
				Mdot_wind = calc_mdot(model_dict[mod]['DIF']/2.6, model_dict[mod]['dOB'], stmass, stage, metallicity=0.0)
				nH = mdot_density(Mdot_wind, model_dict[mod]['DIF']/2.6)
				model_dict[mod]['Mdot_wind'] = Mdot_wind
				model_dict[mod]['nH'] = nH
				print(model_dict[mod])
				print('nH: %.2E'%nH)
				print('mdot_wind: %.2E'%Mdot_wind)
			pickle.dump(model_dict, f)
	else:
		with open('proplyd_dict.pkl', 'rb') as f:
			model_dict = pickle.load(f)

	remodel_dict = {}
	remodel_dict['name'] = []
	for key in model_dict:
		for keykey in model_dict[key]:
			remodel_dict[keykey] = []
		break
	for mod in model_dict:
		remodel_dict['name'].append(mod)
		for keykey in model_dict[mod]:
			remodel_dict[keykey].append(model_dict[mod][keykey])
	

	df = pd.DataFrame(remodel_dict)
	df.to_csv('proplyd_valus.csv',index=False)

	#model_dict = {'globule_ne3_1000_0.0375': 1e3, 'globule_ne7_1000_0.0375': 1e7


	
	fig, ax = plt.subplots(figsize=(6, 5))
	imod=0
	for mod in model_dict:
		if not os.path.isfile(mod+'_flux.npy'):
			print('Running PDR calculation for %s'%mod)
			out = pdr_globule(model_dict[mod]['DIF'], model_dict[mod]['nH'], ambientG0,  stmasses=[stmass], stages=[stage], stdist=-1.*model_dict[mod]['dOB'], FUV_amb=1.0, modname=mod, plot=False, fetch_spect=plotsave)
			if plotsave:
				spf, data= out
				# Assign column names based on the header information
				data.columns = ['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)', 'Wavelength (Å)']

				# Converting Wavelength from Ångström to microns and renormalizingf Emerging intensity accordingly
				data['Wavelength (µm)'] = data['Wavelength (Å)'] * 1e-4  # 1 Å = 1e-4 µm
				data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)'] = data['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)'] * 1e4  # Conversion factor from per Ångström to per micron

				# Filtering the data to include only the range between 1 and 25 microns
				filtered_data = data[(data['Wavelength (µm)'] >= 1) & (data['Wavelength (µm)'] <= 25)]

				wl = filtered_data['Wavelength (µm)']
				flux = filtered_data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)']
				np.save(mod+'_flux', np.array([wl, flux]))
		else:
			wl, flux = np.load(mod+'_flux.npy')
		if plotsave:
			nH = model_dict[mod]['nH']
			diameter = model_dict[mod]['DIF']
			mdot = nH*mH*mu*4.*np.pi*3.0*1e5*(diameter*au2cm/2.6)**2
			mdot /= Msol2g/year
			Mtot = nH*mH*mu*4.*np.pi*(diameter*au2cm/2.6)**3 / 3.0
			Mtot /= Msol2g
			tdep = Mtot/mdot
			tdep /= year
			lab = '$n_\mathrm{H} = %.2E$ cm$^{-3}$, $\dot{M}_\mathrm{ext} \sim %.2E \,M_\odot$ yr$^{-1}$'%(nH, mdot)
			plt.plot(wl, flux, lw=1, label=lab)
		


		imod+=1
		if imod>=5:
			break
	


	if plotsave:
		plt.xscale('log')
		plt.yscale('log')

		#plt.xlim([1., 25.])
		plt.legend(loc=4, fontsize=8)
		ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
		plt.xlabel('Wavelength [$\mu$m]')
		plt.ylabel('Flux [erg cm$^{-2}$ s$^{-1}$ $\mu$m$^{-1}$ sr$^{-1}$]')
		plt.savefig('quick_proplyd.pdf', bbox_inches='tight', format='pdf')
		plt.show()

	
def run_proplydgrid_d3(ambientG0=1.0, diameter=1000.0, stmass=35.0, stage=1.0, plotsave=True):

	model_dict = {'proplyd_163-249_d3': {'dOB': 0.07*3, 'DIF': 215.}, \
	'proplyd_176-543_d3': {'dOB': 0.27*3, 'DIF': 525.}, \
	'proplyd_171-340_d3': {'dOB': 0.04*3, 'DIF': 324.},\
	'proplyd_218-339_d3': {'dOB': 0.16*3, 'DIF': 189.},\
	'proplyd_170-249_d3': {'dOB': 0.07*3, 'DIF': 423.},\
	'proplyd_187-314_d3': {'dOB': 0.07*3, 'DIF': 252.},\
	'proplyd_175-251_d3': {'dOB': 0.07*3, 'DIF': 198.},\
	'proplyd_170-337_d3': {'dOB': 0.03*3, 'DIF': 513.},\
	'proplyd_189-329_d3': {'dOB': 0.07*3, 'DIF': 153.},\
	'proplyd_191-350_d3': {'dOB': 0.09*3, 'DIF': 414.}}


	if not os.path.isfile('proplyd_dict_d3.pkl'):
		print(model_dict)
		with open('proplyd_dict_d3.pkl', 'wb') as f:

			for mod in model_dict:
				Mdot_wind = calc_mdot(model_dict[mod]['DIF']/2.6, model_dict[mod]['dOB'], stmass, stage, metallicity=0.0)
				nH = mdot_density(Mdot_wind, model_dict[mod]['DIF']/2.6)
				model_dict[mod]['Mdot_wind'] = Mdot_wind
				model_dict[mod]['nH'] = nH
				print(model_dict[mod])
				print('nH: %.2E'%nH)
				print('mdot_wind: %.2E'%Mdot_wind)
			pickle.dump(model_dict, f)
	else:
		with open('proplyd_dict_d3.pkl', 'rb') as f:
			model_dict = pickle.load(f)

	remodel_dict = {}
	remodel_dict['name'] = []
	for key in model_dict:
		for keykey in model_dict[key]:
			remodel_dict[keykey] = []
		break
	for mod in model_dict:
		remodel_dict['name'].append(mod)
		for keykey in model_dict[mod]:
			remodel_dict[keykey].append(model_dict[mod][keykey])
	

	df = pd.DataFrame(remodel_dict)
	df.to_csv('proplyd_values_d3.csv',index=False)

	#model_dict = {'globule_ne3_1000_0.0375': 1e3, 'globule_ne7_1000_0.0375': 1e7


	
	fig, ax = plt.subplots(figsize=(6, 5))
	imod=0
	for mod in model_dict:
		if not os.path.isfile(mod+'_flux.npy'):
			print('Running PDR calculation for %s'%mod)
			out = pdr_globule(model_dict[mod]['DIF'], model_dict[mod]['nH'], ambientG0,  stmasses=[stmass], stages=[stage], stdist=-1.*model_dict[mod]['dOB'], FUV_amb=1.0, modname=mod, plot=False, fetch_spect=plotsave)
			if plotsave:
				spf, data= out
				# Assign column names based on the header information
				data.columns = ['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)', 'Wavelength (Å)']

				# Converting Wavelength from Ångström to microns and renormalizingf Emerging intensity accordingly
				data['Wavelength (µm)'] = data['Wavelength (Å)'] * 1e-4  # 1 Å = 1e-4 µm
				data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)'] = data['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)'] * 1e4  # Conversion factor from per Ångström to per micron

				# Filtering the data to include only the range between 1 and 25 microns
				filtered_data = data[(data['Wavelength (µm)'] >= 1) & (data['Wavelength (µm)'] <= 25)]

				wl = filtered_data['Wavelength (µm)']
				flux = filtered_data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)']
				np.save(mod+'_flux', np.array([wl, flux]))
		else:
			wl, flux = np.load(mod+'_flux.npy')
		if plotsave:
			nH = model_dict[mod]['nH']
			diameter = model_dict[mod]['DIF']
			mdot = nH*mH*mu*4.*np.pi*3.0*1e5*(diameter*au2cm/2.6)**2
			mdot /= Msol2g/year
			Mtot = nH*mH*mu*4.*np.pi*(diameter*au2cm/2.6)**3 / 3.0
			Mtot /= Msol2g
			tdep = Mtot/mdot
			tdep /= year
			lab = '$n_\mathrm{H} = %.2E$ cm$^{-3}$, $\dot{M}_\mathrm{ext} \sim %.2E \,M_\odot$ yr$^{-1}$'%(nH, mdot)
			plt.plot(wl, flux, lw=1, label=lab)
		


		imod+=1
	


	if plotsave:
		plt.xscale('log')
		plt.yscale('log')

		#plt.xlim([1., 25.])
		plt.legend(loc=4, fontsize=8)
		ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
		plt.xlabel('Wavelength [$\mu$m]')
		plt.ylabel('Flux [erg cm$^{-2}$ s$^{-1}$ $\mu$m$^{-1}$ sr$^{-1}$]')
		plt.savefig('quick_proplyd_d3.pdf', bbox_inches='tight', format='pdf')
		plt.show()

if __name__=='__main__':
	#run_proplydgrid(stmass=37.0, stage=1.0)
	#exit()
	run_proplydgrid_d3(stmass=37.0, stage=1.0)
	exit()
	run_globgrid()
	exit()
	pdr_globule(1000.0, 1e4, 1.0, stmasses=[65., 60.], stages=[2.5, 2.5], stdist = -0.0375)
