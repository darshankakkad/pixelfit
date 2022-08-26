# A python code to fit single Gaussian to each pixel in MUSE.
# The code can be adapted for any IFU cube.

# How to run:
# First edit the input cube, redshifts and output name in the
# "input parameters" section.
# If you want to visualize the plots, enter the pixel number
# at the start of the loop and set show_plot = True (Default=False)

# Run the cube using the following command in the terminal:
# python pixel_fit.py

# Output:
# The output is a multi-extension fits file, each extension containing
# the individual Gaussian parameters for each line.
# Three extensions are named *flag*. These flags indicate the goodness of the fit.
# Flag 1.0 = The fit did not work, 0.0 = Fitting processes withouot errors.

# The user can play around with the way starting values are defined, the bounds
# in the curve_fit lines and add more gaussians (although that would require many
# other changes within the code). 

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import sys
from scipy.optimize import curve_fit
################
# Input parameters

input_cube = "./target_cube.fits"
z = 0.001
output_name = "./target_gauss_models.fits"
show_plot = False
######################
c = 3.0e+5

# Define individual functions for the fit
# continuum = linear, emission lines = Gaussian

# Hbeta kinematic components are tied with OIII.
# NII kinematic components are tied with Halpha.
# Additional Gaussians can be added if interested in BLR emission.

def mygauss(x,mean, sig, peak):
    norm = pow(sig*np.sqrt(2*np.pi),-1)
    exp_in = pow(x - mean,2)/(2*pow(sig,2))
    return peak*np.exp(-exp_in)

def func_fit_oiii(x,*p):
    continuum = p[0] + (p[1]*x)
    vel_narrow = (p[3]*p[2])/(c*2.355)
    oiii_5007 = mygauss(x,p[2],vel_narrow,p[4])
    oiii_4959 = mygauss(x,p[2]-47.932,vel_narrow,p[4]/3)
    hbeta = mygauss(x,p[2]-145.48,vel_narrow,p[5])
    return continuum+oiii_5007+oiii_4959+hbeta

def func_fit_halpha(x,*p):
    continuum = p[0] + (p[1]*x)
    vel_narrow = (p[3]*p[2])/(c*2.355)
    halpha = mygauss(x,p[2],vel_narrow,p[4])
    nii6585 = mygauss(x,p[2]+20.641,vel_narrow,p[5])
    nii6549 = mygauss(x,p[2]-14.769,vel_narrow,p[5]/3) 
    return continuum+halpha+nii6585+nii6549

def func_fit_sii(x,*p):
    continuum = p[0] + (p[1]*x)
    vel_narrow = (p[3]*p[2])/(c*2.355)
    sii_6716 = mygauss(x,p[2],vel_narrow,p[4])
    sii_6731 = mygauss(x,p[2]+14.48,vel_narrow,p[5])
    return continuum+sii_6716+sii_6731

# Read in the input cube
cube_data1 = pyfits.open(input_cube)
cube_data = cube_data1[1].data
error_data = cube_data1[2].data

# Get the wavelength data from the cube
# Depending on which IFU data you use, the headers keywords and the
# respective extensions might be different. 
crpix = cube_data1[0].header["CRPIX3"]
crval = cube_data1[0].header["CRVAL3"]
cdel  = cube_data1[0].header["CD3_3"]
num = cube_data1[0].header["NAXIS3"]
wl = (np.linspace(1,num,num) - crpix)*cdel + crval
wl = wl/(1+z)

# Define filters for the OIII + H-beta fits
filt_oiii = np.where((wl>4800)&(wl<5100))
cube_data_oiii = cube_data[filt_oiii,:,:]
cube_data_oiii =  cube_data_oiii[0,:,:,:]
error_data_oiii = error_data[filt_oiii,:,:]
error_data_oiii = error_data_oiii[0,:,:,:]
wl_oiii = wl[filt_oiii]

# Define filters for the Halpha + NII fits
filt_halpha = np.where((wl>6450)&(wl<6650))
cube_data_halpha = cube_data[filt_halpha,:,:]
cube_data_halpha =  cube_data_halpha[0,:,:,:]
error_data_halpha = error_data[filt_halpha,:,:]
error_data_halpha = error_data_halpha[0,:,:,:]
wl_halpha = wl[filt_halpha]

# Define filters for the SII fits
filt_sii = np.where((wl>6630)&(wl<6830))
cube_data_sii = cube_data[filt_sii,:,:]
cube_data_sii =  cube_data_sii[0,:,:,:]
error_data_sii = error_data[filt_sii,:,:]
error_data_sii = error_data_sii[0,:,:,:]
wl_sii = wl[filt_sii]

Nz,Ny,Nx = np.shape(cube_data)

# Define the free Gaussian parameters for each set of lines.

oiii_wavelength = np.zeros((Ny,Nx))
oiii_width = np.zeros((Ny,Nx))
oiii_peak = np.zeros((Ny,Nx))
hb_peak = np.zeros((Ny,Nx))
fail_fit_oiii_flag = np.zeros((Ny,Nx))

halpha_wavelength = np.zeros((Ny,Nx))
halpha_width = np.zeros((Ny,Nx))
halpha_peak = np.zeros((Ny,Nx))
nii6585_peak = np.zeros((Ny,Nx))
fail_fit_halpha_flag = np.zeros((Ny,Nx))

sii6716_wavelength = np.zeros((Ny,Nx))
sii6716_width = np.zeros((Ny,Nx))
sii6716_peak = np.zeros((Ny,Nx))
sii6731_peak = np.zeros((Ny, Nx))
fail_fit_sii_flag = np.zeros((Ny,Nx))

# A loop to run the fitting routine for each pixel.

for i in range(0,Nx):
    for j in range(0,Ny):
        fl_oiii = cube_data_oiii[:,j,i]
        err_oiii = error_data_oiii[:,j,i]
        fl_oiii[np.isnan(fl_oiii)] = 0.0
        err_oiii[np.isnan(err_oiii)] = 0.0

        fl_halpha = cube_data_halpha[:,j,i]
        err_halpha = error_data_halpha[:,j,i]
        fl_halpha[np.isnan(fl_halpha)] = 0.0
        err_halpha[np.isnan(err_halpha)] = 0.0
        
        fl_sii = cube_data_sii[:,j,i]
        err_sii = error_data_sii[:,j,i]
        fl_sii[np.isnan(fl_sii)] = 0.0
        err_sii[np.isnan(err_sii)] = 0.0

        # Initial guesses for OIII & Hbeta fit
        OIII_guess_fl = fl_oiii[np.where((wl_oiii>4990)&(wl_oiii<5015))]
        OIII_guess_wl = wl_oiii[np.where((wl_oiii>4990)&(wl_oiii<5015))]
        centroid_guess_oiii = OIII_guess_wl[np.where(OIII_guess_fl == np.max(OIII_guess_fl))][0]
        width_guess_oiii = 350
        peak_guess_oiii = np.max(OIII_guess_fl)
        peak_guess_hb = np.max(fl_oiii[np.where((wl_oiii>4840)&(wl_oiii<4880))])
        start_oiii = [0.0,0.0,centroid_guess_oiii,width_guess_oiii,peak_guess_oiii,peak_guess_hb]
        
        try:
            p_oiii,pcov_oiii = curve_fit(func_fit_oiii, wl_oiii, fl_oiii, sigma=np.sqrt(abs(fl_oiii)),\
                                         p0=start_oiii, \
                                         bounds=([-np.inf,-np.inf,4990.,150.0,0.0,0.0],\
                                                 [np.inf,np.inf,5020.,2000.0,np.inf,np.inf]))
        except RuntimeError:
            print("OIII fit failed at", i, j)
            p_oiii = np.zeros(len(start_oiii))
            fail_fit_oiii_flag[j,i] = 1.0
        except ValueError:
            print("OIII fit failed at", i, j)
            p_oiii = np.zeros(len(start_oiii))
            fail_fit_oiii_flag[j,i] = 1.0

        if show_plot == True:
            fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(10,4))
            ax[0].plot(wl_oiii, fl_oiii)
            ax[0].plot(wl_oiii, func_fit_oiii(wl_oiii, *p_oiii))
        oiii_wavelength[j,i] = p_oiii[2] 
        oiii_width[j,i] = p_oiii[3]
        oiii_peak[j,i] = p_oiii[4]
        hb_peak[j,i] = p_oiii[5]

        # Initial guesses for NII & Halpha fit
        halpha_guess_fl = fl_halpha[np.where((wl_halpha>6552)&(wl_halpha<6575))]
        halpha_guess_wl = wl_halpha[np.where((wl_halpha>6552)&(wl_halpha<6575))]
        centroid_guess_halpha = halpha_guess_wl[np.where(halpha_guess_fl == np.max(halpha_guess_fl))][0]
        width_guess_halpha = 350
        peak_guess_halpha = np.max(halpha_guess_fl)
        peak_guess_nii = np.max(fl_halpha[np.where((wl_halpha>6576)&(wl_halpha<6590))])
        start_halpha = [0.0,0.0,centroid_guess_halpha,width_guess_halpha,peak_guess_halpha,peak_guess_nii]

        try:
            p_halpha,pcov_halpha = curve_fit(func_fit_halpha, wl_halpha, fl_halpha, sigma=err_halpha,\
                                             p0=start_halpha, \
                                             bounds=([-np.inf,-np.inf,6552.,150.0,0.0,0.0],\
                                                     [np.inf,np.inf,6572.,2000.0,np.inf,np.inf]))
        except RuntimeError:
            print("Halpha fit failed at", i, j)
            p_halpha = np.zeros(len(start_halpha))
            fail_fit_halpha_flag[j,i] = 1.0
        except ValueError:
            print("Halpha fit failed at", i, j)
            p_halpha = np.zeros(len(start_halpha))
            fail_fit_halpha_flag[j,i] = 1.0

        if show_plot == True:
            ax[1].plot(wl_halpha, fl_halpha)
            ax[1].plot(wl_halpha, func_fit_halpha(wl_halpha, *p_halpha))
        halpha_wavelength[j,i] = p_halpha[2]
        halpha_width[j,i] = p_halpha[3]
        halpha_peak[j,i] = p_halpha[4]
        nii6585_peak[j,i] = p_halpha[5]

        # Initial guesses for SII
        sii_guess_fl = fl_sii[np.where((wl_sii>6705)&(wl_sii<6722))]
        sii_guess_wl = wl_sii[np.where((wl_sii>6705)&(wl_sii<6722))]
        centroid_guess_sii = sii_guess_wl[np.where(sii_guess_fl == np.max(sii_guess_fl))][0]
        width_guess_sii = 350
        peak_guess_sii = np.max(sii_guess_fl)
        peak_guess_sii6731 = np.max(fl_sii[np.where((wl_sii>6724)&(wl_sii<6743))])
        start_sii = [0.0,0.0,centroid_guess_sii,width_guess_sii,peak_guess_sii,peak_guess_sii6731]
    
        try:
            p_sii,pcov_sii = curve_fit(func_fit_sii, wl_sii, fl_sii, sigma=err_sii,\
                                       p0=start_sii, \
                                       bounds=([-np.inf,-np.inf,6700.,150.0,0.0,0.0],\
                                               [np.inf,np.inf,6724.,2000.0,np.inf,np.inf]))
        except RuntimeError:
            print("SII fit failed at", i, j)
            p_sii = np.zeros(len(start_sii))
            fail_fit_sii_flag[j,i] = 1.0
        except ValueError:
            print("SII fit failed at", i, j)
            p_sii = np.zeros(len(start_sii))
            fail_fit_sii_flag[j,i] = 1.0

        if show_plot == True:
            ax[2].plot(wl_sii, fl_sii)
            ax[2].plot(wl_sii, func_fit_sii(wl_sii, *p_sii))
            plt.show()
            sys.exit()
        sii6716_wavelength[j,i] = p_sii[2]
        sii6716_width[j,i] = p_sii[3]
        sii6716_peak[j,i] = p_sii[4]
        sii6731_peak[j,i] = p_sii[5]
        print("Finished row "+str(i)+"/"+str(Nx))

new_hdul = pyfits.HDUList()
new_hdul.append(pyfits.ImageHDU(np.zeros((1,1)), name="Primary"))
new_hdul.append(pyfits.ImageHDU(oiii_wavelength, name="OIII_wave"))
new_hdul.append(pyfits.ImageHDU(oiii_width, name="OIII_width"))
new_hdul.append(pyfits.ImageHDU(oiii_peak, name="OIII_peak"))
new_hdul.append(pyfits.ImageHDU(hb_peak, name="HB_peak"))
new_hdul.append(pyfits.ImageHDU(fail_fit_oiii_flag, name="OIII_fit_flag"))
new_hdul.append(pyfits.ImageHDU(halpha_wavelength, name="HA_wave"))
new_hdul.append(pyfits.ImageHDU(halpha_width, name="HA_width"))
new_hdul.append(pyfits.ImageHDU(halpha_peak, name="HA_peak"))
new_hdul.append(pyfits.ImageHDU(nii6585_peak, name="NII6585_peak"))
new_hdul.append(pyfits.ImageHDU(fail_fit_halpha_flag, name="HA_fit_flag"))
new_hdul.append(pyfits.ImageHDU(sii6716_wavelength, name="SII6716_wave"))
new_hdul.append(pyfits.ImageHDU(sii6716_width, name="SII6716_width"))
new_hdul.append(pyfits.ImageHDU(sii6716_peak, name="SII6716_peak"))
new_hdul.append(pyfits.ImageHDU(sii6731_peak, name="SII6731_peak"))
new_hdul.append(pyfits.ImageHDU(fail_fit_sii_flag, name="SII_fit_flag"))
new_hdul.writeto(output_name, overwrite=True)
