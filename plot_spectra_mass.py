"""
Plot stellar spectra for a range of masses at a fixed age, and plot EUV/FUV
luminosity as a function of stellar mass.

Usage:
    python plot_spectra_mass.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings

import stellar_evolution as se

# ── wavelength band definitions (Angstroms) ──────────────────────────────────
EUV_RANGE = (10., 912.)      # extreme UV (ionising)
FUV_RANGE = (912., 2070.)    # far UV (Habing / dissociation)

# Habing field reference for G0 conversion (erg/cm²/s)
G0_ERG = 1.6e-3


def get_fuv_euv(mstar, age, metallicity=0.0):
	"""Return (FUV_erg_s, EUV_phot_s) for a star of given mass and age (Myr)."""
	wave, flux, radius, _ = se.get_spectra(mstar, age, metallicity)
	fuv, euv = se.compute_fuv_euv_luminosities(wave, flux, radius)
	return fuv, euv


def plot_spectra(masses, age=1.0, metallicity=0.0, ax=None, savefig=None):
	"""
	Plot normalised stellar spectra for a list of masses at a given age.

	Parameters
	----------
	masses : list of float
	    Stellar masses in solar masses.
	age : float
	    Age in Myr.
	metallicity : float
	    [Fe/H] metallicity.
	"""
	show = ax is None
	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 5))

	norm = mcolors.LogNorm(vmin=min(masses), vmax=max(masses))
	cmap = cm.get_cmap('plasma')

	for mstar in masses:
		print(f'  Getting spectrum for {mstar:.2f} Msol ...')
		try:
			wave, flux, radius, atm_mod = se.get_spectra(mstar, age, metallicity)
		except Exception as e:
			print(f'  Skipped {mstar} Msol: {e}')
			continue

		Ltot = np.trapezoid(flux * 4. * np.pi * radius**2, wave)
		color = cmap(norm(mstar))
		ax.plot(wave, flux * 4. * np.pi * radius**2 / Ltot,
				color=color, lw=1.0, alpha=0.85,
				label=f'{mstar:.2f} $M_\\odot$')

	# Shade EUV and FUV bands
	ax.axvspan(*EUV_RANGE, alpha=0.12, color='purple', label='EUV')
	ax.axvspan(*FUV_RANGE, alpha=0.10, color='gold',   label='FUV')
	ax.axvline(912.,  color='purple', lw=0.8, ls='--')
	ax.axvline(2070., color='goldenrod', lw=0.8, ls='--')

	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	plt.colorbar(sm, ax=ax, label='Mass [$M_\\odot$]')

	ax.set_xlabel('Wavelength [Å]')
	ax.set_ylabel('Normalised luminosity per Å  [arb.]')
	ax.set_title(f'Stellar spectra at {age:.1f} Myr')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(50, 1e5)
	ax.set_ylim(1e-10, 1.)
	ax.tick_params(which='both', top=True, right=True)
	ax.legend(fontsize=7, ncol=2, loc='upper left')
	plt.tight_layout()

	if savefig:
		plt.savefig(savefig, bbox_inches='tight')
		print(f'Saved {savefig}')
	if show:
		plt.show()


def plot_fuv_euv_vs_mass(masses, age=1.0, metallicity=0.0, savefig=None):
	"""
	Plot EUV photon luminosity and FUV luminosity as a function of stellar mass.

	Parameters
	----------
	masses : array-like of float
	    Stellar masses in solar masses.
	age : float
	    Age in Myr.
	"""
	fuv_arr = np.full(len(masses), np.nan)
	euv_arr = np.full(len(masses), np.nan)

	for im, mstar in enumerate(masses):
		print(f'  {mstar:.2f} Msol ...', end=' ', flush=True)
		try:
			fuv_arr[im], euv_arr[im] = get_fuv_euv(mstar, age, metallicity)
			print(f'FUV={fuv_arr[im]:.2e} EUV={euv_arr[im]:.2e}')
		except Exception as e:
			print(f'failed: {e}')

	ok = np.isfinite(fuv_arr) & np.isfinite(euv_arr)

	fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

	axes[0].plot(masses[ok], fuv_arr[ok], 'o-', color='goldenrod', lw=1.5, ms=5)
	axes[0].set_ylabel('FUV luminosity [erg s$^{-1}$]')
	axes[0].set_yscale('log')
	axes[0].set_title(f'UV luminosities at {age:.1f} Myr')
	axes[0].tick_params(which='both', top=True, right=True)
	axes[0].grid(True, which='both', alpha=0.3)

	axes[1].plot(masses[ok], euv_arr[ok], 'o-', color='purple', lw=1.5, ms=5)
	axes[1].set_ylabel('EUV photon luminosity [s$^{-1}$]')
	axes[1].set_xlabel('Stellar mass [$M_\\odot$]')
	axes[1].set_yscale('log')
	axes[1].tick_params(which='both', top=True, right=True)
	axes[1].grid(True, which='both', alpha=0.3)

	for ax in axes:
		ax.set_xscale('log')

	plt.tight_layout()

	if savefig:
		plt.savefig(savefig, bbox_inches='tight')
		print(f'Saved {savefig}')
	plt.show()

	return masses[ok], fuv_arr[ok], euv_arr[ok]


if __name__ == '__main__':

	AGE = 1.0  # Myr

	# ── spectra plot: representative masses from 0.1 to 100 Msol ─────────────
	spec_masses = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 15.0, 40.0, 100.0]
	print(f'\n=== Plotting spectra for {spec_masses} Msol at {AGE} Myr ===')
	plot_spectra(spec_masses, age=AGE, savefig='spectra_vs_mass.pdf')

	# ── FUV / EUV vs mass grid (0.1–100 Msol) ───────────────────────────────
	mass_grid = np.logspace(np.log10(0.1), np.log10(100.), 40)
	print(f'\n=== Computing FUV/EUV for {len(mass_grid)} masses at {AGE} Myr ===')
	m_ok, fuv_ok, euv_ok = plot_fuv_euv_vs_mass(
		mass_grid, age=AGE, savefig='fuv_euv_vs_mass.pdf'
	)

	# Save results
	np.save('fuv_euv_1Myr', np.array([m_ok, fuv_ok, euv_ok]))
	print('Saved fuv_euv_1Myr.npy  (rows: mass [Msol], FUV [erg/s], EUV [phot/s])')
