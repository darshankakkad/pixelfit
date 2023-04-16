import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.io.fits as pyfits
from tqdm import tqdm
import time
start_time = time.time()

class IFUFitter:
    def __init__(self, filename,z,wave_range):
        """
        Initiate the class with the name of the file, redshift 
        and the wavelength ranges of interest
        
        Inputs 
        filename: string, name of the cube
        z: redshift
        wave_range: [w1,w2], Range of wavelength where the cube should be limited

        Outputs
        wave_dered: Deredshifted wavelength array
        spectra: An array containing spectral information for every pixel
        xsize, ysize: Size of the IFU FoV
        """
        
        hdu=pyfits.open(filename)
        head=hdu[1].header
        sci_data=hdu[1].data

        Nz,Ny,Nx=sci_data.shape
        spectra=sci_data.reshape(Nz,-1)
        wave=head['CRVAL3'] + head['CD3_3']*np.arange(Nz)
        wave_dered=wave/(1+z)
        filt_wave = np.where((wave > wave_range[0]) & (wave < wave_range[1]))
        spectra = spectra[filt_wave,:]
        wave_dered = wave_dered[filt_wave]
        
        self.wave=wave
        self.wave_dered=wave_dered
        self.spectra=spectra
        self.xsize=Nx
        self.ysize=Ny

    def fit_multigauss(self, x,cont0,cont1,*params):
        """
        A function defining a linear continuum model and multi-gaussians
        Inputs
        x: Array, in the case here, wavelength array
        cont0,cont1: Linear function coefficients
        *params: Gaussian parameters in the order mean,sigma and peak

        Output:
        A continuum + Multi-Gaussian model. The number of Gaussians defined by
        len(*params)/3
        """
        cont_func = cont0+x*cont1
        gauss_terms = np.zeros_like(x)
        for i in range(0, len(params), 3):
            mean = params[i]
            sig = params[i+1]
            peak = params[i+2]
            gauss_terms = gauss_terms + peak * np.exp(-(x - mean)**2/(2*sig**2))
        return cont_func+gauss_terms

    def fit_gaussian_cube(self,outfile):
        """
        A function to fit the Gaussian function defined above to the datacube
        The function saves each parameter into different extensions of a fits file
        """
        guess = [1.0,1.0]
        bounds_low = [-np.inf,-np.inf]
        bounds_high = [np.inf,np.inf]
        lines = [4860,4959,5007,6549,6562,6585,6716,6731]
        for i in range(len(lines)):
            guess += [lines[i],1.5,1000]
            bounds_low += [lines[i]-10,1.0,0]
            bounds_high += [lines[i]+10,2.0,np.inf]
        
        fit_params = np.zeros((len(guess),self.xsize*self.ysize))
        self.spectra[np.isnan(self.spectra)] = 0
        
        for i in tqdm(range(41000,41010)):
            try:
                popt, pcov = curve_fit(self.fit_multigauss, self.wave_dered, self.spectra[0,:,i],
                                    p0=guess,bounds=[bounds_low,bounds_high])
                fit_params[:,i] = popt
            except RuntimeError:
                popt = np.zeros(len(guess))
                fit_params[:,i] = popt
        new_hdul = pyfits.HDUList()
        new_hdul.append(pyfits.ImageHDU(fit_params))
        new_hdul.writeto(outfile,overwrite=True)


# Usage
if __name__ == "__main__":
    x = IFUFitter('NGC2992.fits',0.007296,[3850,8000])
    y = x.fit_gaussian_cube("NGC2992_model.fits")
    
    end_time = time.time()
    print("Time taken = ", end_time-start_time)
