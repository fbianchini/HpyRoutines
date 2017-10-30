import numpy as np
import healpy as hp

def Sky2Hpx(sky1, sky2, nside, coord='G', nest=False, rad=False):
	"""
	Converts sky coordinates, i.e. (RA,DEC), to Healpix pixel at a given resolution nside.
	By default, it rotates from EQ -> GAL

	Parameters
	----------
	sky1 : array-like
		First coordinate, it can be RA, LON, ...

	sky2 : array-like
		Second coordinate, it can be DEC, LAT, ...
		
	nside : int
		Resolution of the Healpix pixelization scheme

	coord : str 'C','E' or 'G' [def]
		Coordinate system of the output. If coord='C', *no* rotation is applied

	nest : bool [def=False]
		If True, nested pixelation scheme is considered for the output pixels

	rad : bool [def=False]
		If True, input coordinates are expressed in radians, otherwise in degree

	Returns
	-------
	ipix : array-like
		Pixel indices corresponding to (sky1,sky2) coordinates
	"""
	sky1, sky2 = np.asarray(sky1), np.asarray(sky2)

	if rad == False: # deg -> rad
		theta = np.deg2rad(90.-sky2) 
		phi   = np.deg2rad(sky1) 
	else: 
		theta = np.pi/2. - sky2
		phi   = sky1 	     

	# Apply rotation if needed (default EQ -> GAL)
	r = hp.Rotator(coord=['C',coord], deg=False)
	theta, phi = r(theta, phi)

	npix = hp.nside2npix(nside)

	return hp.ang2pix(nside, theta, phi, nest=nest) # Converting galaxy coordinates -> pixel 

def GetCountsMap(sky1, sky2, nside, coord='G', nest=False, rad=False, sqdeg=False):
	"""
	Creates an Healpix map with sources number counts at resolution nside given two 
	arrays containing SKY1 and SKY2 positions.

	Parameters
	----------
	sky1 : array-like
		First coordinate, it can be RA, LON, ...

	sky2 : array-like
		Second coordinate, it can be DEC, LAT, ...
		
	nside : int
		Resolution of the Healpix pixelization scheme

	coord : str 'C','E' or 'G' [def]
		Coordinate system of the output. If coord='C', *no* rotation is applied

	nest : bool [def=False]
		If True, nested pixelation scheme is considered for the output pixels

	rad : bool [def=False]
		If True, input coordinates are expressed in radians, otherwise in degree

	sqdeg : bool [def=False]
		If True, pixels of the output map contains galaxy density in *square degrees*

	Returns
	-------
	ipix : array-like
		Pixel indices corresponding to (sky1,sky2) coordinates
	"""
	# Get sources position (pixels)
	pix = Sky2Hpx(sky1, sky2, nside, coord=coord, nest=nest, rad=rad)

	# Create counts map
	counts_map = np.bincount(pix, minlength=hp.nside2npix(nside))*1.

	if sqdeg:
		counts_map *= hp.nside2pixarea(nside)

	return counts_map

def Counts2Delta(counts, mask=None):
	"""
	Converts a number counts Healpix map into a density contrast map.

	Parameters
	----------
	counts : array-like
		Input healpix map with number counts

	mask : array-like
		Healpix map containing the mask 

	Note
	----
	* use only binary mask.
	"""
	counts = np.asarray(counts)

	if not hp.isnpixok(counts.size):
		raise ValueError("Not a valid healpix map!")

	if mask is not None:
		mask   = np.asarray(mask)
		counts = hp.ma(counts)
		counts.mask = np.logical_not(mask)
	
	mean  = np.mean(counts)
	delta = (counts - mean) / mean

	if mask is not None:
		delta = delta.filled(counts.fill_value) # To avoid pixels with fill_value = 1e+20
	
	return delta

def GuessMask(sky1, sky2, nside, coord='G', nest=False, rad=False):
	"""
	Returns a binary Healpix mask at resolution nside by assigning 1 to pixels
	containing a source or 0 otherwise. given two arrays containing 
	RA and DEC positions

	Parameters
	----------
	sky1 : array-like
		First coordinate, it can be RA, LON, ...

	sky2 : array-like
		Second coordinate, it can be DEC, LAT, ...
		
	nside : int
		Resolution of the Healpix pixelization scheme

	coord : str 'C','E' or 'G' [def]
		Coordinate system of the output. If coord='C', *no* rotation is applied

	nest : bool [def=False]
		If True, nested pixelation scheme is considered for the output pixels

	rad : bool [def=False]
		If True, input coordinates are expressed in radians, otherwise in degree
	"""
	# Get sources positions
	pix = Sky2Hpx(sky1, sky2, nside, coord=coord, nest=nest, rad=rad)

	# Create Healpix mask
	mask = np.zeros(hp.nside2npix(nside))
	mask[pix] = 1.

	return mask

def GetGalMask(nside, lat=None, fsky=None, nest=False):
	"""
	Returns a symmetric Galactic Mask in Healpix format at nside resolution.
	Pixels with latitude < |lat| deg are set to 0.
	Otherwise you input the fsky and evaluates the required latitude.

	Parameters
	----------
	nside : int
		Resolution of the Healpix pixelization scheme

	lat : float
		Latitude of the galatic cut

	fsky : float
		Sky coverage of the footprint

	nest : bool [def=False]
		If True, nested pixelation scheme is considered for the output pixels
	"""
	if lat is None:
		if fsky is not None:
			lat = np.rad2deg(np.arcsin(1. - fsky))
		else:
			raise ValueError("Missing lat or fsky !")

	mask = np.zeros(hp.nside2npix(nside))
	theta_cut = np.deg2rad(90. - lat)
	mask[np.where((hp.pix2ang(nside, np.arange(mask.size), nest=nest))[0] >= (np.pi - theta_cut))] = 1.
	mask[np.where((hp.pix2ang(nside, np.arange(mask.size), nest=nest))[0] <= theta_cut)] = 1.

	return mask

def GetSourcesMask(sky1, sky2, nside, radius, coord='G', nest=False, rad=False):
	'''
	Generates a Healpix map with the 0 in the pixels within a 
	radius distance from the objects coordinates (sky1,sky2)

	Parameters
	----------
	sky1 : array-like
		First coordinate, it can be RA, LON, ...

	sky2 : array-like
		Second coordinate, it can be DEC, LAT, ...
		
	nside : int
		Resolution of the Healpix pixelization scheme

	radius : array-like
		The radius of the disc in degrees, can be different for each source

	coord : str 'C','E' or 'G' [def]
		Coordinate system of the output. If coord='C', *no* rotation is applied

	nest : bool [def=False]
		If True, nested pixelation scheme is considered for the output pixels

	rad : bool [def=False]
		If True, input coordinates are expressed in radians, otherwise in degree

	Returns
	-------
	psmask : array-like
		A Healpix map with 0 values inside the discs
	'''

	pspix  = Sky2Hpx(sky1, sky2, nside, coord=coord, nest=nest, rad=rad)
	x,y,z  = hp.pix2vec(nside, pspix, nest=nest)
	psmask = np.ones(hp.nside2npix(nside))

	if len(np.atleast_1d(radius)) == 1:
		radius = np.ones(len(x)) * radius

	for i in xrange(len(x)):
		ipix = hp.query_disc(nside, (x[i],y[i],z[i]), np.radians(radius[i]))
		psmask[ipix] = 0.0

	return psmask

def ThrowCircleMasksOnTheSky(radius, mask=None, nside=2048, overlap=0.1, overlap_mask=0.05, fsky=0.1, max_iter=200, return_pix=False):

    full_mask = np.ones(hp.nside2npix(nside))

    if mask is not None:
        full_mask *= mask
    else:
        mask = np.ones(hp.nside2npix(nside))
        
    pix_ones = np.where(full_mask == 1)[0]
    
    i = 0
    xs = []
    ys = []
    zs = []
    while (np.mean(full_mask) > fsky):
        x, y, z = hp.pix2vec(nside,np.random.choice(pix_ones))
        ipix = hp.query_disc(nside, (x, y, z), np.radians(radius))
        cond1 = len(np.where(full_mask[ipix]==0)[0]) < overlap * len(ipix)
        cond2 = len(np.where(mask[ipix]==0)[0]) < overlap_mask * len(ipix)
        if cond1 and cond2:
            full_mask[ipix] = 0.
            pix_ones = np.where(full_mask == 1)[0]
            i = 0 
            xs.append(x)
            ys.append(y)
            zs.append(z)
        else:
            i += 1
            if i > max_iter: break
            
    print 'created %d holes' %len(xs)
    if i !=0:
        print 'exiting with %d iterations'%(i)
    print 'f_sky = %.3f' %full_mask.mean()
    if return_pix:
        return full_mask, (xs, ys, zs)
    else:
        return full_mask
	
def filter_bandpass_1d(reclen, lmin, lmax, dl):
    # central_freq = samp_freq/float(reclen)
    l = np.arange(reclen)#/2+1) * central_freq
    filt = np.ones(reclen)

    filt[l<lmin-dl/2] = 0.0
    window_for_transition_to_zero = (lmin-dl/2 <= l) * (l <= lmin+dl/2)
    ind = np.where(window_for_transition_to_zero==True)[0]
    reclen_transition_window = len(filt[window_for_transition_to_zero])
    filt[window_for_transition_to_zero] = (1. - np.cos( np.pi* np.arange(reclen_transition_window) /(reclen_transition_window-1))) /2.0

    filt[l>lmax+dl/2] = 0.0

    window_for_transition_to_zero = (lmax-dl/2 <= l) * (l <= lmax+dl/2)
    ind = np.where((window_for_transition_to_zero==True))[0]
    reclen_transition_window = len(filt[window_for_transition_to_zero])
    filt[window_for_transition_to_zero] = (1. + np.cos( np.pi* np.arange(reclen_transition_window) /(reclen_transition_window-1))) /2.0

    return filt

def GetCorrAlms(clxy, clxx, clyy, nside, lmax=None):
	"""
	Generates correlated gaussian realizations of two fields X and Y 
	which statistics is described by the sets of three Auto- and 
	Cross-spectra clxx, clyy, and clxy. 

	Based on HealPix routine create_alm.
	
	Parameters
	----------
	clxy :  array-like 
		Input cross-power spectrum C_l^{XY}

	clxx :  array-like 
		Input auto-power  spectrum C_l^{XX} of first field

	clyy :  array-like 
		Input auto-power  spectrum C_l^{YY} of second field

	nside : int 
		Healpix resolution

	lmax : int (optional) 
		Maximum multipole 
	

	Returns
	-------
	alm_xx : array-like
		First field Alm coefficients
	
	alm_yy : array-like
		Second field Alm coefficients 

	Notes
	-----
	*  Realizations are returned as *alms* 

	"""

	if lmax is None:
		lmax = 2*nside 
		lmax = int(lmax)

	tot_alm = hp.sphtfunc.Alm.getsize(lmax)  

	alm_xx = np.zeros(tot_alm, dtype=complex)
	alm_yy = np.zeros(tot_alm, dtype=complex)


	# Loop on l: Let's deal with X field coefficients
	# and first term rms_yy1 of the Y field coefficients
	for l in xrange(0,lmax+1):
		rms_xx = 0.
		rms_yy1 = 0.
		if clxx[l] != 0.: # To avoid division by zero
			rms_xx = np.sqrt(clxx[l])
			rms_yy1 = clxy[l]/rms_xx

		# Treat the m = 0 case
		rand1_r = np.random.normal()
		rand1_i = 0.
		alm_xx.real[l] = rand1_r*rms_xx
		alm_yy.real[l] = rand1_r*rms_yy1

		#Treat the m > 0 cases
		for m in xrange(1,l+1):
			rand1_r = np.random.normal()/np.sqrt(2.0)
			rand1_i = np.random.normal()/np.sqrt(2.0)
			alm_xx.real[(m*(2*lmax+1-m)/2)+l] = rand1_r*rms_xx
			alm_xx.imag[(m*(2*lmax+1-m)/2)+l] = rand1_i*rms_xx
			alm_yy.real[(m*(2*lmax+1-m)/2)+l] = rand1_r*rms_yy1
			alm_yy.imag[(m*(2*lmax+1-m)/2)+l] = rand1_i*rms_yy1

	# Loop on l: second term rms_yy2 of Y field coefficients
	for l in xrange(0,lmax+1):
		rms_yy2 = 0.
		if clxx[l] != 0.: # To avoid division by zero
			rms_yy2 = clyy[l] - (clxy[l]/clxx[l])*clxy[l]
			rms_yy2 = np.sqrt(rms_yy2)

		# Treat the m = 0 case
		rand2_r = np.random.normal()
		rand2_i = 0.
		alm_yy.real[l] = alm_yy.real[l] + rand2_r*rms_yy2

		#Treat the m > 0 cases
		for m in xrange(1,l+1):
			rand2_r = np.random.normal()/np.sqrt(2.0)
			rand2_i = np.random.normal()/np.sqrt(2.0)
			alm_yy.real[(m*(2*lmax+1-m)/2)+l] = alm_yy.real[(m*(2*lmax+1-m)/2)+l] + rand2_r*rms_yy2
			alm_yy.imag[(m*(2*lmax+1-m)/2)+l] = alm_yy.imag[(m*(2*lmax+1-m)/2)+l] + rand2_i*rms_yy2

	return np.nan_to_num(alm_xx), np.nan_to_num(alm_yy)

def GetCorrMaps(clxy, clxx, clyy, nside, lmax=None, pixwin=True):
	"""
	Generates correlated gaussian realizations of two fields X and Y 
	which statistics is described by the sets of three Auto- and 
	Cross-spectra clxx, clyy, and clxy. 
	
	! Realizations are returned as *maps* !

	Parameters
	----------
	clxy :  array-like 
		Input cross-power spectrum C_l^{XY}

	clxx :  array-like 
		Input auto-power  spectrum C_l^{XX} of first field

	clyy :  array-like 
		Input auto-power  spectrum C_l^{YY} of second field

	nside : int 
		Healpix resolution

	lmax : int (optional) 
		Maximum multipole 

	pixwin : bool (optional) [def=True] 
		If True, Convolve with pixel window function
	
	Returns
	-------
	map_xx : array-like
		First field map
	
	map_yy : array-like
		Second field map
	"""
	# if lmax is None:
	#     lmax = clxy.size+1

	alm_xx, alm_yy = GetCorrAlms(clxy, clxx, clyy, nside, lmax=lmax)

	# Creating XX and YY Maps
	map_xx = hp.sphtfunc.alm2map(alm_xx, nside, pixwin=pixwin, lmax=lmax, verbose=False)
	map_yy = hp.sphtfunc.alm2map(alm_yy, nside, pixwin=pixwin, lmax=lmax, verbose=False)

	return map_xx, map_yy


def GetNlgg(counts, mask=None, lmax=None, return_ngal=False):
	"""
	Returns galaxy shot-noise spectra given a number counts Healpix map. 
	If return_ngal is True, it returns also the galaxy density in gal/ster.

	Parameters
	----------
	counts :  array-like 
		Healpix map containing sources counts

	mask :  array-like 
		Healpix map containing binary mask

	lmax : int (optional) 
		Maximum multipole 

	return_ngal : bool (optional) [def=False] 
		If True, returns the galaxy density in gal/ster
	
	Returns
	-------
	nlgg : array-like
		noise power spectrum
	

	Notes
	----
	1. If mask is not binary, the mean 
	2. If mask is not None, yielded spectrum is not pseudo
	"""
	counts = np.asarray(counts)

	if lmax is None: lmax = hp.npix2nside(counts.size) * 2
	if mask is not None: 
		mask = np.asarray(mask)
		fsky = np.mean(mask)
	else: 
		mask = 1.
		fsky = 1.

	N_tot = np.sum(counts * mask)
	ngal  = N_tot / 4. / np.pi / fsky

	if return_ngal:
		return np.ones(lmax+1) / ngal, ngal
	else:
		return np.ones(lmax+1) / ngal
