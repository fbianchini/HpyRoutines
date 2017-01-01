import numpy as np
import healpy as hp

arcmin2rad = np.pi / 180. / 60. 

def SmoothMask(mask, fwhm, pcut=1e-2, order='ring', verbose=False):
	"""
	mask is binary healpix mask
	fwhm in arcmin
	Based on Kim2011 (astro-ph:)
	"""
	try:
		mask = hp.read_map(mask, verbose=verbose)
	except:
		mask = mask		

	# Generate a widened *binary* mask from original one by:
	# i)  smoothing it w/ Gaussian beam
	# ii) setting to zero pixels according to threshold
	extmask = hp.smoothing(mask, fwhm=fwhm*arcmin2rad, verbose=verbose)
	extmask[extmask <= (1. - pcut)] = 0.
	extmask[extmask >= (1. - pcut)] = 1.

	# Smooth the extended mask
	maskout = hp.smoothing(extmask, fwhm=fwhm*arcmin2rad, verbose=verbose)
	p  = np.where(maskout >= 1. - pcut)
	p2 = np.where(maskout <= 1. - pcut)
	maskout[p] = 1.
	p = np.where(maskout <= pcut)
	maskout[p] = 0.
	extmaskout = maskout.copy()
	extmaskout[p2] = 0.

	return maskout, extmaskout

