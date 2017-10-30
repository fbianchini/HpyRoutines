import numpy as np
import healpy as hp

arcmin2rad = np.pi / 180. / 60. 

def correctTheta(theta):
    if theta < 0.:
        return -theta
    if theta > np.pi:
        return 2 * np.pi - theta
    return theta


def cosApodization(sigma, x):
   if x > sigma:
       return 1
   return 1 - np.cos(np.pi * x / (2 * sigma))

def gaussApodization(sigma, x):
   if x > sigma:
       return 1
    # check(sigma > 0, "invalid sigma = " << sigma);
    return 1 - np.exp(- 9 * x * x / (2 * sigma * sigma))

# def apodize(angle, Healpix_Map<double>& result, type) const
#     check(type >= COSINE_APODIZATION && type < APODIZATION_TYPE_MAX, "invalid apodization type");
#     check(angle > 0, "invalid angle = " << angle);


#     result.SetNside(mask_.Nside(), mask_.Scheme());
#     result.Import(mask_);

#     nPix = result.Npix();
# 	pixSize = np.sqrt(4 * np.pi / nPix);

#     Healpix_Base2 base2(result.Nside(), result.Scheme(), SET_NSIDE);
    
#     print("Finding the edge pixels...")

#     std::vector<Math::ThreeVectorDouble> edge;

#     for i in xrange(nPix):
#         if result[i] == 0:
#             continue
#         isOnEdge = False
#         fix_arr<int64, 8> neighbors;
#         base2.neighbors(i, neighbors);
#         const long s = neighbors.size();
#         for j = 0; j < s; ++j)
#         {
#             if(result[neighbors[j]] == 0)
#                 isOnEdge = true;
#         }
#         double theta, phi;
#         result.Scheme() == NEST ? pix2ang_nest(result.Nside(), i, &theta, &phi) : pix2ang_ring(result.Nside(), i, &theta, &phi);
#         long index;
#         /*
#         result.Scheme() == NEST ? ang2pix_nest(result.Nside(), correctTheta(theta + pixSize), phi, &index) : ang2pix_ring(result.Nside(), correctTheta(theta + pixSize), phi, &index);
#         if(result[index] == 0)
#             isOnEdge = true;
#         result.Scheme() == NEST ? ang2pix_nest(result.Nside(), correctTheta(theta - pixSize), phi, &index) : ang2pix_ring(result.Nside(), correctTheta(theta - pixSize), phi, &index);
#         if(result[index] == 0)
#             isOnEdge = true;
#         result.Scheme() == NEST ? ang2pix_nest(result.Nside(), theta, phi + pixSize, &index) : ang2pix_ring(result.Nside(), theta, phi + pixSize, &index);
#         if(result[index] == 0)
#             isOnEdge = true;
#         result.Scheme() == NEST ? ang2pix_nest(result.Nside(), theta, phi - pixSize, &index) : ang2pix_ring(result.Nside(), theta, phi - pixSize, &index);
#         if(result[index] == 0)
#             isOnEdge = true;
#         */
#         if(isOnEdge)
#         {
#             edge.push_back(Math::ThreeVectorDouble(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)));
#             //ang2pix_nest(edgeNSide, theta, phi, &index);
#             //edgeMap[index] = 1;
#             //edgeMap[i] = 1;
#         }
#         meter.advance();
#     }
#     output_screen("OK" << std::endl);

#     output_screen("Found " << edge.size() << " edge pixels." << std::endl);

#     /*
#     fitshandle outh;
#     outh.create("edge.fits");
#     write_Healpix_map_to_fits(outh, edgeMap, PLANCK_FLOAT64);
#     */

#     output_screen("Apodizing mask..." << std::endl);
#     ProgressMeter meter1(nPix);
# #ifdef COSMO_OMP
#     omp_lock_t lock;
#     omp_init_lock(&lock);
# #endif
# #pragma omp parallel for default(shared)
#     for(long i = 0; i < nPix; ++i)
#     {
#         if(result[i] == 0)
#         {
# #ifdef COSMO_OMP
#             omp_set_lock(&lock);
# #endif
#             meter1.advance();
# #ifdef COSMO_OMP
#             omp_unset_lock(&lock);
# #endif
#             continue;
#         }

#         double theta, phi;
#         result.Scheme() == NEST ? pix2ang_nest(result.Nside(), i, &theta, &phi) : pix2ang_ring(result.Nside(), i, &theta, &phi);
#         Math::ThreeVectorDouble current(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
#         double minDistance = 2 * Math::pi;
#         for(int j = 0; j < edge.size(); ++j)
#         {
#             const double distance = std::acos(current * edge[j]);
#             if(distance < minDistance)
#                 minDistance = distance;
#         }
#         result[i] = (type == COSINE_APODIZATION ? cosApodization(angle, minDistance) : gaussApodization(angle, minDistance));

# #ifdef COSMO_OMP
#         omp_set_lock(&lock);
# #endif
#         meter1.advance();
# #ifdef COSMO_OMP
#         omp_unset_lock(&lock);
# #endif
#     }

def angular_distance_from_mask(mask):
    """
    For each pixel of a Healpix map, return the smallest angular distance
    to a set of masked pixels (of value True), in degrees.
    Parameter
    ---------
    maskok : boolean Healpix map
        The Healpix mask that defines the set of masked pixels (of value True)
        whose smallest angular distance to each pixel of a Healpix map of same
        nside is computed.
    """
    nside = hp.npix2nside(len(mask))

    # get the list of pixels on the external border of the mask
    ip = np.arange(12*nside**2)[~mask]
    neigh = hp.get_all_neighbours(nside, ip)
    nn = np.unique(neigh.ravel())
    if nn[0] == -1:
        nn = nn[1:]
    nn = nn[mask[nn]]

    # get unit vectors for border and inner pixels
    vecpix_inner = np.array(hp.pix2vec(nside, ip))
    vecpix_outer = np.array(hp.pix2vec(nside, nn))

    # get angles between the border pixels and inner pixels
    cosang = np.dot(vecpix_inner.T, vecpix_outer)
    mapang = np.zeros(12*nside**2)
    mapang[~mask] = np.degrees(np.arccos(np.max(cosang, axis=1)))
    return mapang


def apodize_mask(maskok, fwhm_deg):
    """
    Apodize a mask.
    Parameters
    ----------
    maskok : boolean Healpix map
        The mask to be apodized (of value True).
    fwhm_deg : float
        The FWHM of the apodization function, counted from the mask edges,
        in degrees.
    """
    sigma_deg = fwhm_deg / np.sqrt(8 * np.log(2))
    mapang = angular_distance_from_mask(~maskok)
    return 1 - np.exp(-0.5 * mapang**2 / sigma_deg**2)

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

