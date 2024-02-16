"""
The following code is an adaptation of Michele Cappellari's PPXF code (Cappellari et al. 2017). 
It performs a stellar continuum fit to a data cube (specifically from the VLT/MUSE instrument) 
and saves the final result: continuum cube, velocity and velocity dispersion, in a fits file.
"""


import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from os import path
from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import display_bins
from plotbin.plot_velfield import plot_velfield
from scipy.constants import c
import sys
import datetime

C = c/1000  # speed of light in km/s

class read_muse_cube(object):
    def __init__(self,filename):
        """
        Read MUSE cube, log rebin it and compute coordinates of each spaxel.
        Median FWHM resolution = 2.62Å. Range: 2.51--2.88 (ESOpPXF Purpose
        instrument manual)
        
        """
        hdu = pyfits.open(filename)
        head = hdu[1].header
        cube = hdu[1].data 
        Nz,Ny,Nx=np.shape(cube)

        spec_orig=cube.copy()
        spectra=cube
        wave = (head['CRVAL3'] + head['CD3_3']*np.arange(Nz))   
        pixsize = abs(head["CD1_1"])*3600
        wave=wave/(1+z)

        # Create coordinates centred on the brightest spectrum
        flux = np.nanmean(spectra, 0)
        jm = np.argmax(flux)
        row, col = map(np.ravel, np.indices(cube.shape[-2:]))
        x = (col - col[jm])*pixsize
        y = (row - row[jm])*pixsize

        # DK  20230825: Had to modify the velscale definition according to the ppxf code.
        #velscale = C*np.diff(np.log(wave[-2:]))
        velscale = C*np.diff(np.log(wave[[0, -1]]))[0]/(wave.size - 1)

        # Keeping the same wavelength range as the original cube
        lam_range_temp = [np.min(wave), np.max(wave)]
        spectra, ln_lam_gal, velscale = util.log_rebin(lam_range_temp, spectra, velscale=velscale)
        
        self.spectra = spectra
        self.spec_orig=spec_orig
        self.ny=Ny
        self.nx=Nx
        self.x = x
        self.y = y
        self.col = col + 1   # start counting from 1
        self.row = row + 1
        self.lam_gal = wave
        self.ln_lam_gal=ln_lam_gal
        self.fwhm_gal = 2.62  # Median FWHM resolution of MUSE

def fit_and_clean(templates, galaxy, velscale, start, goodpixels0, lam, lam_temp):
    """
    This function performs the continuum fitting. Compared to the original Cappellari code
    the masks are fixed. 
    """
    goodpixels = goodpixels0.copy()
    pp = ppxf(templates, galaxy, np.ones(len(galaxy)), velscale, start,
              moments=2, degree=-1, mdegree=4, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels)
    optimal_template = templates @ pp.weights
    return pp, optimal_template

if __name__=="__main__":
    
    input_cube='cube.fits'
    z=0.0
    yyyymmddss = datetime.datetime.now()
    date = str(yyyymmddss.year)+str("{0:0=2d}".format(yyyymmddss.month))+str("{0:0=2d}".format(yyyymmddss.day))
    
    s=read_muse_cube(input_cube)
    
    velscale=C*np.diff(s.ln_lam_gal[[0, -1]])[0]/(s.ln_lam_gal.size - 1)
    regul_err = 0.01 # Desired regularization error
    vel0 =0.  ## Initial estimate of the galaxy velocity in km/s.
    start = [vel0, 100.]  # (km/s), starting guess for [V,sigma]

    # Edit the directory path
    ppxf_dir = '/user/dkakkad/miniconda3/envs/research/lib/python3.9/site-packages/ppxf'
    pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
    FWHM_gal = None   # set this to None to skip convolutions
    miles = lib.miles(pathname, velscale, FWHM_gal, norm_range=[5070, 5950])
    stars_templates, ln_lam_temp = miles.templates, miles.ln_lam_temp
    reg_dim = stars_templates.shape[1:]
    stars_templates = stars_templates.reshape(stars_templates.shape[0], -1)
    stars_templates /= np.median(stars_templates) # Normalizes stellar templates by a scalar
    
    lam_range_temp = np.exp(ln_lam_temp[[0, -1]])
    goodpixels0=np.arange(0,len(s.ln_lam_gal),1)
    
    # Key emission lines and telluric regions to flag
    # In the case of AO data, additional notch filter locations will need to be flagged
    Hbeta = 4861.333
    oiii4959 = 4958.911
    oiii5007 = 5006.843
    HeI5875 = 5875.624
    oi6300 = 6300.304
    oi6363 = 6363.776
    nii6548 = 6548.050
    Halpha = 6562.819
    nii6583 = 6583.460
    sii6716 = 6716.081
    sii6731 = 6730.810
    mask_wid = 15
    # 7020-7135 = telluric
    badpix_ranges = [(Hbeta-mask_wid,Hbeta+mask_wid),(oiii4959-mask_wid,oiii4959+mask_wid),
                    (oiii5007-mask_wid,oiii5007+mask_wid), (HeI5875-mask_wid,HeI5875+mask_wid),
                    (oi6300-mask_wid,oi6300+mask_wid),(nii6548-mask_wid,nii6548+mask_wid),
                    (Halpha-mask_wid,Halpha+mask_wid),(nii6583-mask_wid,nii6583+mask_wid),
                    (sii6716-mask_wid,sii6716+mask_wid),(sii6731-mask_wid,sii6731+mask_wid),(7020,7135)]
    filt_badpix = np.any([np.logical_and(s.ln_lam_gal > np.log(start), s.ln_lam_gal < np.log(end)) 
                        for start, end in badpix_ranges], axis=0)
    goodpixels0=np.delete(goodpixels0,filt_badpix)
    
    velbin = np.zeros((s.ny,s.nx))
    sigbin = np.zeros((s.ny,s.nx))
    stellar_cont=np.zeros((len(s.ln_lam_gal),s.ny,s.nx))
    gal_spec=np.zeros((len(s.ln_lam_gal),s.ny,s.nx))
    
    lam_gal = np.exp(s.ln_lam_gal)
    
    for i in range(0,s.nx-1):
        for j in range(0,s.ny-1):
            galaxy = s.spectra[:,j,i]
            galaxy[np.isnan(galaxy)]=0
            pp, bestfit_template = fit_and_clean(stars_templates, galaxy, velscale, start, goodpixels0, lam_gal, miles.lam_temp)
            #gal_spec[:,j,i]=galaxy
            stellar_cont[:,j,i]=np.interp(s.lam_gal,np.exp(s.ln_lam_gal),pp.bestfit)
            velbin[j,i], sigbin[j,i] = pp.sol

    # Preserving the original header information
    hdu = pyfits.open(input_cube)
    new_hdul = pyfits.HDUList()
    new_hdul.append(pyfits.ImageHDU(np.zeros((1,1)), name="Primary",header=hdu[0].header)) 
    new_hdul.append(pyfits.ImageHDU(stellar_cont, name="CONT",header=hdu[1].header))
    new_hdul.append(pyfits.ImageHDU(velbin, name="Vstar",header=hdu[1].header))
    new_hdul.append(pyfits.ImageHDU(sigbin, name="SIGMAstar",header=hdu[1].header))
    new_hdul.writeto("output_cube_"+date+".fits", overwrite=True)
