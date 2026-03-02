# -*- coding: utf-8 -*-
"""
This code takes the corrected file from *sso_freeze* (hardwired by user) and 
peforms a corrdinate transformation on the X-ray emission to wrap the PSF 
around Jupiter

@history: 
    Adapted from 'gochandra' IDL script. (???)
    Translated into a function and generalized by MJR (2025)

@authors: 
    Dale Weigt (D.M.Weigt@soton.ac.uk)
    Randy Gladstone
    Hunter Waite
    Kurt Franke
    Peter Ford
    Seán McEntee
    Caitríona Jackman
    Will Dunn
    Brad Snios
    Ron Elsner
    Ralph Kraft
    Graziella Branduardi-Raymont 
    Matthew J. Rutala
"""
# Import packages
import go_chandra_analysis_tools as gca_tools # import the defined functions to analysis Chandra data nad perfrom coordinate transformations
import sso_freeze

import numpy as np
import pandas as pd
import scipy
from scipy import interpolate
# from astropy.io import ascii
# from astropy.io import fits as pyfits
import astropy
from astropy import units as u
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
import os
from astropy.time import Time
import configparser
from matplotlib import patches
from pathlib import Path
import glob
import tqdm

from astropy.time import Time                   #convert between different time coordinates
from astropy.time import TimeDelta              #add/subtract time intervals 
from astroquery.jplhorizons import Horizons     #automatically download ephemeris 

plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')


# =============================================================================
#     # Assumptions 
# =============================================================================
j_rotrate = np.rad2deg(1.758533641E-4) # Jupiter's rotation period
# scale = 0.13175 # scale used when observing Jupiter using Chandra - in units of arcsec/pixel
fwhm = 0.8 # FWHM of the HRC-I point spread function (PSF) - in units of arcsec
psfsize = 25 # size of PSF used - in units of arcsec
alt = 400 # altitude where X-ray emission is assumed to occur in Jupiter's ionosphere - in units of km
# dtor = np.pi/180
# rtod = 180/np.pi

# CONSTANTS
# BETTER IF THESE WERE READ FROM SOMEWHERE
rad_eq_0 = R_eq_0 = 71492.0 # radius of equator in km
rad_pole_0 = R_rot_0 = 66854.0 # radius of poles in km
flattening = (rad_eq_0 - rad_pole_0)/rad_eq_0
ecc = np.sqrt(1.0-(rad_pole_0/rad_eq_0)**2) # oblateness of Jupiter 

    
    # Frames:
    # SKY: Full RA & DEC 
    # TAR(GET): Target-centered RA & DEC
    # JUP(ITER): Target-centered and rotated arcseconds

def go_chandra(acis=None, obs_id=None, obs_dir=None, config=None,
               default_psf=False):

    
    # Pull out AU -> m conversion factor
    au_to_m = u.au.to(u.m)
    
    # If acis, obs_id, and obs_dir are specified, they take precedence
    if (acis is not None) & (obs_id is not None) & (obs_dir is not None):
        pass
    else:
        # Parse config file
        cfg = configparser.ConfigParser()
        cfg.read(config)
        
        acis = str(cfg['inputs']['ACIS'])
        obs_id = cfg['inputs']['obsID']
        obs_dir = str(cfg['inputs']['folder_path'])
    
    # Search given dir for sso_freeze-corrected event file
    corrected_event_filepath = sso_freeze.find_event_filepath(acis, obs_id, obs_dir, suffix="ssofreeze_evt2.fits")
    
    # Read the sso_freeze-corrected event file into a dataframe
    corrected_events = cev = pd.DataFrame()
    
    with astropy.io.fits.open(corrected_event_filepath, dtype=float) as hdulist:
        
        # Which variables to read, and what to assign them to
        cev_relex = {'t': 'time', 
                     'x': 'X', 
                     'y': 'Y', 
                     'channel':'pha',
                     'sumamps': 'sumamps',
                     'samp': 'samp',
                     'pi_cal': 'pi',
                     'av1': 'av1', 'av2': 'av2', 'av3': 'av3',
                     'au1': 'au1', 'au2': 'au2', 'au3': 'au3',
                     'amp_sf': 'amp_sf'
                     }
        
        # The .astype() call is needed to convert from big-endian to native little-endian
        for output_key, input_key in cev_relex.items():
            corrected_events.loc[:,output_key] = hdulist['EVENTS'].data[input_key].astype(np.float64)
            
       
        # img_events  = hdulist['EVENTS'].data # event file data
        
        # bigtime     = hdulist['EVENTS'].data['time'] # time
        # bigxarr     = hdulist['EVENTS'].data['X'] # x position of photons
        # bigyarr     = hdulist['EVENTS'].data['Y'] # y position of photons
        # bigchannel  = hdulist['EVENTS'].data['pha'] # pha channel the photons were found in
        # sumamps     = hdulist['EVENTS'].data['sumamps'] # reading in sumamps figure
        # samp        = hdulist['EVENTS'].data['samp'] # reading in samp figure
        # pi_cal      = hdulist['EVENTS'].data['pi']

        # # reading in amplifier signals 
        # av1 = hdulist['EVENTS'].data['av1']
        # av2 = hdulist['EVENTS'].data['av2']
        # av3 = hdulist['EVENTS'].data['av3']
        
        # au1 = hdulist['EVENTS'].data['au1']
        # au2 = hdulist['EVENTS'].data['au2']
        # au3 = hdulist['EVENTS'].data['au3']
        # amp_sf = hdulist['EVENTS'].data['amp_sf'] # reading in amplifier scaling factor

        img_head    = hdulist[1].header # header
        obs_id      = img_head['OBS_ID'] # observation id of the event
        tstart      = img_head['TSTART'] # the start and...
        tend        = img_head['TSTOP'] #... end time of the observation
    
        # The date of the observation is read in...
        datestart = img_head['DATE-OBS']
        evt_date = pd.to_datetime(datestart) #... and coverted to datetiem format to allow the relevant information to be read to...
        evt_hour = evt_date.hour
        evt_doy = evt_date.strftime('%j')
        evt_mins = evt_date.minute
        evt_secs = evt_date.second
        evt_DOYFRAC = gca_tools.doy_frac(float(evt_doy), float(evt_hour), float(evt_mins), float(evt_secs)) #... calculated a fractional Day of 
        # Year (DOY) of the observation
        
        # !!! I want to keep these for now, in case I can use them in centering later on...
        ra_centre, ra_centre_rad = img_head['RA_NOM'], np.deg2rad(img_head['RA_NOM']) # the RA of Jupiter at the centre of the chip is read in as...
        dec_centre, dec_centre_rad = img_head['DEC_NOM'], np.deg2rad(img_head['DEC_NOM']) #... well as Jupitr's DEC
    
   
    # # !!!!! Untouched
    # # Extracts relevent information needed from ephermeris file
    # cml_spline_jup = scipy.interpolate.UnivariateSpline(eph_jup['datetime_jd'], eph_jup['PDObsLon'],k=1)
    # lt_jup = eph_jup['lighttime']
    # sub_obs_lon_jup = eph_jup['PDObsLon']
    # sub_obs_lat_jup = eph_jup['PDObsLat']
    
    # # Adding angular diameter from JPL Horizons to use later to define radius of circular region within which photons are kept
    # ang_diam = max(eph_jup['ang_width'])
    
    # # Also adding tilt angle of Jupiter with respect to true North Pole
    # tilt_ang = np.mean(eph_jup['NPole_ang'])
    
    # # Do we really need to write this to file?
    # # saving angular diameter and tilt angle in text file in order to plot ellipse in post-processing
    # np.savetxt(str(obs_dir) + f'/{obs_id}_JPL_ellipse_vals.txt', np.c_[ang_diam, tilt_ang], delimiter=',', header='angular diameter (arcsec),tilt angle (deg)', fmt='%s')
    
    # eph_dates = pd.to_datetime(eph_jup['datetime_str'])
    # eph_dates = pd.DatetimeIndex(eph_dates)
    # eph_doy = np.array(eph_dates.strftime('%j')).astype(int)
    # eph_hours = eph_dates.hour
    # eph_minutes = eph_dates.minute
    # eph_seconds = eph_dates.second
    
    # eph_DOYFRAC_jup = gca_tools.doy_frac(eph_doy, eph_hours, eph_minutes, eph_seconds) # DOY fraction from ephermeris data
    
    # jup_time = (eph_DOYFRAC_jup - evt_DOYFRAC)*86400.0 + tstart # local tiem of Jupiter
    
    
    # %%===========================================================================
    # Locate the planet on the chip
    # =============================================================================
    # Select Region for analysis
    
    # Plots the photons (x,y) position on a grid of defined size in arcseconds 
    # (defualted at [-50,50] in both x and y). Jupiter is centred on the HRC 
    # instrument. The photon information form the defined 
    
    # The centering values were previsouly hardcoded, but fail for non-standard observations
    # Instead, search header keywords for the x, y values to get the correct center
    for key, val in img_head['TTYPE??'].items():
        if val == 'x':
            keyx = key
        if val == 'y':
            keyy = key
    detx_center = img_head['TCRPX'+keyx[5:]]
    dety_center = img_head['TCRPX'+keyy[5:]]
    detx_scaling = img_head['TCDLT'+keyx[5:]] * 3600 # in "/pixel
    dety_scaling = img_head['TCDLT'+keyy[5:]] * 3600 # in "/pixel
    
    corrected_events.loc[:, 'tar_x'] = (cev['x'] - detx_center) * detx_scaling
    corrected_events.loc[:, 'tar_y'] = (cev['y'] - dety_center) * dety_scaling
    # bigxarr_region = (bigxarr - skyx_center) * skyx_scaling
    # bigyarr_region = (bigyarr - skyy_center) * skyy_scaling
    
    # %% PSF Calculations =====================================================
    # If we're more than ~1 arcminute off-axis, model the PSF with marx
    # =========================================================================
    ΔRA = img_head['RA_TARG'] - img_head['RA_NOM']
    ΔDEC = img_head['DEC_TARG'] - img_head['DEC_NOM']
    off_axis = np.sqrt(ΔRA **2 + ΔDEC**2) * u.degree
    
    # Only use the default if we're on-axis and the user asked for it
    if (default_psf == True) & (off_axis < 1*u.arcminute):
        default_psf = True
    else:
        default_psf = False
    
    # Get the appropriate covariance matrix for the observing ellipse
    if default_psf == False:
        psf_cov = psf_from_header(img_head, obs_dir)
    
    else:
        psf_sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        psf_cov = np.array([[psf_sigma, 0], [0, psf_sigma]])
        
    
    # %%===========================================================================
    #     
    # =============================================================================
    # storing all photon data in text file - need this to calculate area for samp distributions later on
    # np.savetxt(str(obs_dir) + r"/%s_all_photons.txt" % obs_id, np.c_[bigxarr_region, bigyarr_region, bigtime, bigchannel, samp, sumamps, pi_cal, amp_sf, av1, av2, av3, au1, au2, au3])
        

    
    tar_xmin, tar_xmax = -50, 50
    tar_ymin, tar_ymax = -50, 50
    cmin, cmax = 0, cev['channel'].max()
    region_query = "(@tar_xmin <= tar_x <= @tar_xmax) & " + \
                   "(@tar_ymin <= tar_y <= @tar_ymax) & " + \
                   "(@cmin <= channel <= @cmax)"
    region_events = rev = corrected_events.query(region_query).copy()
    
    # Horizons search code courtesy of Brad Snios
    # The start and end times are taken from the header
    start_time = Time(tstart, format='cxcsec')
    stop_time = Time(tend, format='cxcsec')
    delta_time = '1m'
    eph = gca_tools.fetch_ephemerides_fromCXO(start_time, stop_time, delta_time)
    
    # Get ephemeris data for each photon event
    eph_t = Time(eph['datetime_jd'], format='jd').cxcsec
    
    rev['sublon'] = np.interp(rev['t'], eph_t, eph['PDObsLon'])
    rev['sublat'] = np.interp(rev['t'], eph_t, eph['PDObsLat'])
    rev['NPPA'] = np.interp(rev['t'], eph_t, eph['NPole_ang'])
    rev['NPDist'] = np.interp(rev['t'], eph_t, eph['NPole_dist'])
    
    rev['R_eq'] = np.interp(rev['t'], eph_t, eph['ang_width']/2)
    rev['R_rot'] = rev['R_eq'] * (1 - flattening)
    rev['R_NS'] = apparent_size(rev['R_eq'], rev['R_rot'], rev['sublat'])/2
    
    rev['R_eq_emiss'] = (R_eq_0 + alt) * (rev['R_eq'] / R_eq_0)
    rev['R_rot_emiss'] = rev['R_eq_emiss'] * (1 - flattening)
    rev['R_NS_emiss'] = apparent_size(rev['R_eq_emiss'], rev['R_rot_emiss'], rev['sublat'])/2
    
    # Get pointing and dither data for each event
    dither_filepath = Path(corrected_event_filepath).parent / 'pcadf{0}_???N???_asol1.fits'.format(obs_id)
   
    # Read the dither information
    with astropy.io.fits.open(glob.glob(str(dither_filepath))[0]) as hdul:
        dither_header =  hdul[1].header
        dither_data = hdul[1].data
    
    rev['RA_target'] = np.interp(rev['t'], eph_t, eph['RA'])
    rev['DEC_target'] = np.interp(rev['t'], eph_t, eph['DEC'])
    rev['RA_pointing'] = np.interp(rev['t'], dither_data['time'], dither_data['ra'])
    rev['DEC_pointing'] =  np.interp(rev['t'], dither_data['time'], dither_data['dec'])
    rev['RA_offset'] = - (rev['RA_pointing'] - img_head['RA_NOM'])
    rev['DEC_offset'] = - (rev['DEC_pointing'] - img_head['DEC_NOM'])
    
    # =============================================================================
    # Plot the selected region, with references for Jupiter and PSF
    # =============================================================================
    jupiter_facecolor = (*colors.to_rgba('xkcd:peach')[0:3], 0.5)
    # Get an ellipse for Jupiter's limb
    limb_ellipse = get_JupiterPatch(r_eq = rev['R_eq_emiss'].max(),
                                    nppa = rev['NPPA'].mean(),
                                    sublat = rev['sublat'].mean(),
                                    edgecolor='xkcd:peach', facecolor=jupiter_facecolor, linewidth=1)
   
    # Also plot a line representing the north rotation pole
    # Get the vertex position of the ellipse (end of minor axis)
    v = limb_ellipse.get_co_vertices()
    # Get a vector going from the north pole position to 1.25
    npole_vec = np.array([rev['NPDist'].mean()/np.linalg.norm(v[0]), 1.25])
    
    
    fig, ax = plt.subplots(figsize=(4,4))

    # Add Jupiter and rotational axis
    ax.plot(npole_vec * v[0][0], npole_vec  * v[0][1],
            color='black', lw=1)
    ax.add_patch(limb_ellipse)
    
    # Add photons
    ax.scatter(rev['tar_x'], rev['tar_y'], 
               marker='.', s=3, lw=0, alpha=1, color='xkcd:navy blue')
    
    # Add a patch to show the PSF
    ax_inset = ax.inset_axes([-40, -40, 10, 10], transform=ax.transData)
    psf_eigenvals, psf_eigenvecs = np.linalg.eig(psf_cov)
    psf_patch = patches.Ellipse(
        (0, 0), width=max(psf_eigenvals), height=min(psf_eigenvals), 
        angle = np.rad2deg(np.arctan2(max(psf_eigenvals) - psf_cov[0,0], psf_cov[0,1])),
        color='xkcd:blue', lw=0, alpha=0.75)
    ax_inset.add_patch(psf_patch)
    ax_inset.annotate('PSF:', (0,1), (0.1,-0.1), 
                      xycoords = 'axes fraction', textcoords='offset fontsize', 
                      ha='left', va='top', fontsize='small')
    ax_inset.set(aspect=1, xlim=[5, -5], ylim=[-5,5])
    ax_inset.get_xaxis().set_visible(False)
    ax_inset.get_yaxis().set_visible(False)
    
    title_string = 'Selected Region (ObsID {})\n{}--{}'.format(
        obs_id, 
        Time(img_head['TSTART'], format='cxcsec').to_datetime().strftime('%Y-%m-%d %H:%M:%S'), 
        Time(img_head['TSTOP'], format='cxcsec').to_datetime().strftime('%Y-%m-%d %H:%M:%S'))
    ax.set(title = title_string,
           xlabel = r'Planet-centered $\alpha$ ["]', xlim = [tar_xmax, tar_xmin], 
           ylabel = r'Planet-centered $\delta$ ["]', ylim = [tar_ymin, tar_ymax])

    plt.show()
    
    # =========================================================================
    # Save our progress so far
    # =========================================================================
    filepath = str(obs_dir)+ "/{}_selected_region_ellipse.csv".format(obs_id)
    with open(filepath, 'w') as f:
        f.write('#UNITS:  t(s), x(arcsec), y(arcsec), PHA, samp, sumamp, pi, amp_sf, av1, av2, av3, au1, au2, au3\n')
        region_events.to_csv(f, header = True, index = False)
    
    # =============================================================================
    # SIII Coordinate Transformation
    # Redeveloped from scratch by MJR Jan. 2026
    # =============================================================================
    
    dsin = lambda x: np.sin(np.deg2rad(x))
    dcos = lambda x: np.cos(np.deg2rad(x))
    
    # Rotate sky coords into Jupiter frame
    rev['jup_x'] = rev['tar_x']*dcos(rev['NPPA']) - rev['tar_y']*dsin(rev['NPPA'])
    rev['jup_y'] = rev['tar_x']*dsin(rev['NPPA']) + rev['tar_y']*dcos(rev['NPPA'])
    
    # We have the x and y coordinates of each photon, but the inverse eqns
    # require simultaneously solivng multiple equations
    
    # Instead, interpolate from (lon, lat) -> (x, y)
    λ_edge_arr = np.linspace(0, 360, 361) * u.degree
    φ_edge_arr = np.linspace(-90, 90, 181) * u.degree
    λ_edge_g, φ_edge_g = np.meshgrid(λ_edge_arr, φ_edge_arr, indexing='ij')
    
    λ_mid_arr = λ_edge_arr[:-1]/2 + λ_edge_arr[1:]/2
    φ_mid_arr = φ_edge_arr[:-1]/2 + φ_edge_arr[1:]/2
    λ_mid_g, φ_mid_g = λg, φg = np.meshgrid(λ_mid_arr, φ_mid_arr, indexing='ij')
        
    uvis_xy, chip2_xy = get_UVISPolygon()
    nice_plots = False
    
    # Use the jupiter-frame photons to determine which events are on disk (or close enough to disk) to count
    inclusion_factor = 1.2
    on_disk_bool = (rev['jup_x']/rev['R_eq_emiss'].max())**2 + (rev['jup_y']/rev['R_NS_emiss'].max())**2 <= inclusion_factor**2
    
    emission_map_in_UVIS = np.zeros(λg.shape)
    emission_map_out_UVIS = np.zeros(λg.shape)
    visibility_map_in_UVIS = np.zeros(λg.shape)
    visibility_map_out_UVIS = np.zeros(λg.shape)
    time_of_previous_jupiter_photon = rev['t'].iloc[0]
    
    if nice_plots:
        fig_anim, axs_anim = plt.subplots(figsize=(8,4), ncols=2)
        frames_anim = []
        axs_anim[0].set(
            aspect=1, 
            xlim=rev['RA_target'].mean() + np.array([0.1, -0.1]), xlabel = r'$\alpha$ [deg.]', 
            ylim=rev['DEC_target'].mean() + np.array([-0.1, 0.1]), ylabel = r'$\delta$ [deg.]'
            )
        axs_anim[1].set(
            aspect=1, 
            xlim=[25,-25], xlabel = r'Jupiter-Centered $\alpha$ ["]', 
            ylim=[-25,25], ylabel=r'Jupiter-Centered $\delta$ ["]'
            )
    
    for index, event in tqdm.tqdm(rev.iterrows(), total=len(rev)):
        
        if on_disk_bool.loc[index] == True:
        
            # Sub-observer lon & lat
            λ0 = event['sublon'] * u.degree
            φ0 = event['sublat'] * u.degree
            
            # Visibility
            cos_c = dsin(φ0) * dsin(φg) + dcos(φ0) * dcos(φg) * dcos(λg - λ0)

            # Forward model (λ, φ) -> (x, y)
            jup_xg_edge = event['R_eq_emiss']  * dcos(φ_edge_g) * dsin(λ_edge_g - λ0)
            jup_yg_edge = event['R_NS_emiss'] * (dcos(φ0) * dsin(φ_edge_g) - dsin(φ0) * dcos(φ_edge_g) * dcos(λ_edge_g - λ0))
            jup_xg = event['R_eq_emiss'] * dcos(φg) * dsin(λg - λ0)
            jup_yg = event['R_NS_emiss'] * (dcos(φ0) * dsin(φg) - dsin(φ0) * dcos(φg) * dcos(λg - λ0))
            
            # Estimate apparent areas of each grid cell
            d1 = np.sqrt((jup_xg_edge[0:-1, 0:-1] - jup_xg_edge[1:, 0:-1])**2 + (jup_yg_edge[0:-1, 0:-1] - jup_yg_edge[1:, 0:-1])**2)
            d2 = np.sqrt((jup_xg_edge[0:-1, 0:-1] - jup_xg_edge[0:-1, 1:])**2 + (jup_yg_edge[0:-1, 0:-1] - jup_yg_edge[0:-1, 1:])**2)
            areas = d1*d2 * u.arcsec**2
            
            # Rotate the xy grid into the TARGET frame
            tar_xg = jup_xg*dcos(-event['NPPA']) - jup_yg*dsin(-event['NPPA'])
            tar_yg = jup_xg*dsin(-event['NPPA']) + jup_yg*dcos(-event['NPPA'])
            tar_xg_edge = jup_xg_edge*dcos(-event['NPPA']) - jup_yg_edge*dsin(-event['NPPA'])
            tar_yg_edge = jup_xg_edge*dsin(-event['NPPA']) + jup_yg_edge*dcos(-event['NPPA'])
            
            # Scale the xy grid to the SKY frame
            sky_xg_edge = tar_xg_edge/3600 + event['RA_target']
            sky_yg_edge = tar_yg_edge/3600 + event['DEC_target']
            
            # Remove xy grid cells on the farside of the planet
            visible_index = cos_c > 0
            λg_vis = λg[visible_index]
            φg_vis = φg[visible_index]
            tar_xg_vis = tar_xg[visible_index]
            tar_yg_vis = tar_yg[visible_index]
            
            # Generate a probability density function for photon origin
            event_pdf = scipy.stats.multivariate_normal.pdf(
                np.stack((tar_xg, tar_yg), axis=-1), 
                mean = [event['tar_x'], event['tar_y']], cov = psf_cov)
            event_pdf_vis = event_pdf[cos_c > 0]
            
            # Roll the UVIS polygon to the correct angle, then offset
            sky_uvis_x = event['RA_pointing'] + uvis_xy[0]*dcos(img_head['ROLL_NOM']) - uvis_xy[1]*dsin(img_head['ROLL_NOM'])
            sky_uvis_y = event['DEC_pointing'] + uvis_xy[0]*dsin(img_head['ROLL_NOM']) + uvis_xy[1]*dcos(img_head['ROLL_NOM'])
            sky_uvis_xy = np.array([sky_uvis_x, sky_uvis_y])
            
            tar_uvis_x = (sky_uvis_x - event['RA_target']) * 3600
            tar_uvis_y = (sky_uvis_y - event['DEC_target']) * 3600
            tar_uvis_xy = np.array([tar_uvis_x, tar_uvis_y])
            
            sky_chip2_x = event['RA_pointing'] + chip2_xy[0]*dcos(img_head['ROLL_NOM']) - chip2_xy[1]*dsin(img_head['ROLL_NOM'])
            sky_chip2_y = event['DEC_pointing'] + chip2_xy[0]*dsin(img_head['ROLL_NOM']) + chip2_xy[1]*dcos(img_head['ROLL_NOM'])
            sky_chip2_xy = np.array([sky_chip2_x, sky_chip2_y])

            tar_chip2_x = (sky_chip2_x - event['RA_target']) * 3600
            tar_chip2_y = (sky_chip2_y - event['DEC_target']) * 3600
            tar_chip2_xy = np.array([tar_chip2_x, tar_chip2_y])

            if nice_plots == True:
                
                # To display, we need gridded data; set color to NaN on farside
                event_pdf_fordisplay = event_pdf.copy()
                event_pdf_fordisplay[cos_c <= 0] = np.nan
                
                uvis_fc = (*colors.to_rgba('xkcd:red')[0:3], 0.25)
                
                a1 = axs_anim[0].pcolormesh(sky_xg_edge.value, sky_yg_edge.value, event_pdf_fordisplay,
                                        cmap='plasma')
                
                chip2_patch = patches.Polygon(sky_chip2_xy.T, ec='black', alpha=1, lw=1, fc='None')
                a2 = axs_anim[0].add_patch(chip2_patch)
                uvis_patch = patches.Polygon(sky_uvis_xy.T, fc=uvis_fc, ec='xkcd:red', lw=1)
                a3 = axs_anim[0].add_patch(uvis_patch)
                
                a5 = axs_anim[1].pcolormesh(tar_xg_edge.value, tar_yg_edge.value, event_pdf_fordisplay,
                                        cmap='plasma')
                
                chip2_patch = patches.Polygon(tar_chip2_xy.T, ec='black', alpha=1, lw=1, fc='None')
                a6 = axs_anim[1].add_patch(chip2_patch)
                uvis_patch = patches.Polygon(tar_uvis_xy.T, fc=uvis_fc, ec='xkcd:red', lw=1)
                a7 = axs_anim[1].add_patch(uvis_patch)
                
                artist_list = [a1, a2, a3, a5, a6, a7]
                
                frames_anim.append(artist_list)
            
            # Assign photons (& probabilities) to a map
            import matplotlib.path as mpltPath
            tar_UVIS_path = mpltPath.Path(tar_uvis_xy.T)
            tar_xyg = np.array([tar_xg.flatten(), tar_yg.flatten()]).T
            in_UVIS_index = tar_UVIS_path.contains_points(tar_xyg)
            in_UVIS_index = in_UVIS_index.reshape(tar_xg.shape)
            # tar_xyg_vis = np.array([tar_xg_vis, tar_yg_vis]).T
            # in_UVIS_vis_index = tar_UVIS_path.contains_points(tar_xyg_vis)
            
            # The maximum of the PSF is the most likely photon location
            # And determines the UVIS filter
            pdf_max_indx = event_pdf_vis.argmax()
            rev.loc[index, 'lon'] = λg_vis[pdf_max_indx].value
            rev.loc[index, 'lat'] = φg_vis[pdf_max_indx].value
            rev.loc[index, 'in_UVIS'] = in_UVIS_index[visible_index][pdf_max_indx]
            
            # Also create a map based on the full PDF
            emap_in_UVIS = np.zeros(emission_map_in_UVIS.shape)
            emap_out_UVIS = np.zeros(emission_map_out_UVIS.shape)
            
            emap_in_UVIS[visible_index & in_UVIS_index] = event_pdf_vis[in_UVIS_index[visible_index]]
            emap_out_UVIS[visible_index & ~in_UVIS_index] = event_pdf_vis[~in_UVIS_index[visible_index]]
            
            # Emission map has units of counts/arcsec^2
            emission_map_in_UVIS += emap_in_UVIS
            emission_map_out_UVIS += emap_out_UVIS
            
            # Estimate the time each grid cell has been visible in/out UVIS
            elapsed_time = event['t'] - time_of_previous_jupiter_photon
            visibility_map_in_UVIS[visible_index & in_UVIS_index] += elapsed_time
            visibility_map_out_UVIS[visible_index & ~in_UVIS_index] += elapsed_time
            time_of_previous_jupiter_photon = event['t']
        else:
            
            # event_lon.append(np.nan * u.degree)
            # event_lat.append(np.nan * u.degree)
            rev.loc[index, 'lon'] = np.nan
            rev.loc[index, 'lat'] = np.nan
            
    if nice_plots:
        # You MUST let this run after the loop for collection of plots...
        import matplotlib.animation as animation
        import time
        t0 = time.time()
        anim = animation.ArtistAnimation(fig_anim, frames_anim, interval=100, blit=True)
        anim.save('animation.mp4', writer='ffmpeg', fps=30)
        print(time.time() - t0)
    
    breakpoint()
    # Show the emission maps, visibility maps, and emission density maps
    fig, axs = plt.subplots(figsize=(6,6), nrows=2, sharex=True, sharey=True)
    norm = colors.Normalize(
        vmin=np.percentile([*emission_map_in_UVIS, *emission_map_out_UVIS], 25), 
        vmax=np.percentile([*emission_map_in_UVIS, *emission_map_out_UVIS], 99))
    m = axs[0].pcolormesh(λ_edge_g, φ_edge_g, emission_map_in_UVIS, 
                          cmap='plasma', norm=norm)
    m = axs[1].pcolormesh(λ_edge_g, φ_edge_g, emission_map_out_UVIS,
                          cmap='plasma', norm=norm)
    for ax in axs:
        ax.set(
            aspect = 1, 
            xlim=[360, 0], xlabel = 'SIII Longitude [deg]', 
            ylim=[-90,90], ylabel = 'Latitude [deg]')
    axs[0].set(title = 'Inside UVIS Filter')
    axs[1].set(title = 'Outside UVIS Filter')
    fig.colorbar(m, ax=axs, orientation='horizontal', fraction=.1)
    plt.show()
    
    # visibility maps
    fig, axs = plt.subplots(figsize=(6,6), nrows=2, sharex=True, sharey=True)
    norm = colors.Normalize(
        vmin=np.percentile([*visibility_map_in_UVIS, *visibility_map_out_UVIS], 25), 
        vmax=np.percentile([*visibility_map_in_UVIS, *visibility_map_out_UVIS], 99))
    m = axs[0].pcolormesh(λ_edge_g, φ_edge_g, visibility_map_in_UVIS, 
                          cmap='plasma', norm=norm)
    m = axs[1].pcolormesh(λ_edge_g, φ_edge_g, visibility_map_out_UVIS,
                          cmap='plasma', norm=norm)
    for ax in axs:
        ax.set(
            aspect = 1, 
            xlim=[360, 0], xlabel = 'SIII Longitude [deg]', 
            ylim=[-90,90], ylabel = 'Latitude [deg]')
    axs[0].set(title = 'Inside UVIS Filter')
    axs[1].set(title = 'Outside UVIS Filter')
    fig.colorbar(m, ax=axs, orientation='horizontal', fraction=.1)
    plt.show()
    
    # emission density maps
    emissdensity_in_UVIS = emission_map_in_UVIS / visibility_map_in_UVIS
    emissdensity_out_UVIS = emission_map_out_UVIS / visibility_map_out_UVIS
    fig, axs = plt.subplots(figsize=(6,6), nrows=2, sharex=True, sharey=True)
    norm = colors.Normalize(
        vmin=np.nanpercentile([*emissdensity_in_UVIS, *emissdensity_out_UVIS], 25), 
        vmax=np.nanpercentile([*emissdensity_in_UVIS, *emissdensity_out_UVIS], 99))
    m = axs[0].pcolormesh(λ_edge_g, φ_edge_g, emissdensity_in_UVIS, 
                          cmap='plasma', norm=norm)
    m = axs[1].pcolormesh(λ_edge_g, φ_edge_g, emissdensity_out_UVIS,
                          cmap='plasma', norm=norm)
    for ax in axs:
        ax.set(
            aspect = 1, 
            xlim=[360, 0], xlabel = 'SIII Longitude [deg]', 
            ylim=[-90,90], ylabel = 'Latitude [deg]')
    axs[0].set(title = 'Inside UVIS Filter')
    axs[1].set(title = 'Outside UVIS Filter')
    fig.colorbar(m, ax=axs, orientation='horizontal', fraction=.1, label='Counts / arcesec2 / s')
    plt.show()
    
    
    rough_me_lon = np.arange(140, 290+10, 10)
    rough_me_lat = np.array([84.94, 73.46, 56.32, 54.65, 55.43, 57.44, 60.11, 
                             63.19, 66.21, 69.24, 72.47, 75.27, 77.66, 80.00, 
                             82.64, 86.78])
    
    fig, ax = plt.subplots(figsize=(6,6), nrows=1, sharex=True, sharey=True)
    m = ax.pcolormesh(λ_edge_g, φ_edge_g, emissdensity_out_UVIS - emissdensity_in_UVIS, 
                          cmap='plasma', vmin=-0.0001, vmax=0.0001)
    ax.set(
        aspect = 1, 
        xlim=[360, 0], xlabel = 'SIII Longitude [deg]', 
        ylim=[-90,90], ylabel = 'Latitude [deg]')
    ax.plot(rough_me_lon, rough_me_lat, color='black', lw=2)
    fig.colorbar(m, ax=ax, orientation='horizontal', fraction=.1, label='Counts / arcesec2 / s')
    plt.show()
    
    breakpoint()
    
    # # Compute visibility over time
    # visibility_map_in_UVIS = np.zeros(λg.shape)
    # visibility_map_out_UVIS = np.zeros(λg.shape)
    # time_range = np.linspace(0, rev['t'].iloc[-1] - rev['t'].iloc[0], int(1e3))
    # for delta_t in time_range[:-1]:
    #     # Compute which lon/lat grid cells are visible at this time
    #     # Sub-observer point on the planet, and (equatorial) radius
    #     λ0 = np.interp(rev['t'].iloc[0] + delta_t, eph_t, eph['PDObsLon']) * u.degree
    #     # λ0_ = 360 * u.degree - λ0
    #     φ0 =  np.interp(rev['t'].iloc[0] + delta_t, eph_t, eph['PDObsLat']) * u.degree
            
    #     # Visibility
    #     cos_c = dsin(φ0) * dsin(φg) + dcos(φ0) * dcos(φg) * dcos(λg - λ0)
        
    #     # If visible, add delta_t to visibility_map
    #     visibility_map[cos_c > 0] += time_range[1]
    
    # Emission density in counts/s/arcsec^2
    visibility_map_for_denominator = vmfd = visibility_map.copy()
    vmfd[vmfd == 0] = 1
    emission_density_map = emission_map / vmfd
    
    total_area = (sky_xmax - sky_xmin) * (sky_ymax - sky_ymin)
    ellipse_exlusion_area = np.pi * (rev['R_eq_emiss'].max() * rev['R_NS_emiss'].max()) * (inclusion_factor**2)
    background_area = total_area - ellipse_exlusion_area
    background = rev[~on_disk_bool]['t'].count() / background_area
    background_emission_density = background / (rev['t'].iloc[-1] - rev['t'].iloc[0])
    
    breakpoint()
    print('HELP!')
    
    
    # Load file for comaprison
    path = '/Users/mrutala/projects/Jupiter_XUV_Comparison/data/CXO/{0}/primary/{0}_photonlist_full_obs_ellipse.csv'.format(obs_id)
    comparison_df = pd.read_csv(path, comment='#')
    
    rough_me_lon = np.arange(140, 290+10, 10)
    rough_me_lat = np.array([84.94, 73.46, 56.32, 54.65, 55.43, 57.44, 60.11, 
                             63.19, 66.21, 69.24, 72.47, 75.27, 77.66, 80.00, 
                             82.64, 86.78])
    
    fig, axs = plt.subplots(figsize=(4,4), nrows=2)
    
    axs[0].scatter(comparison_df['lon'], comparison_df['lat'], 
                   color='black', marker='o', s=2, lw=0, label='Earlier Versions')
    
    axs[1].scatter(rev['lon'], rev['lat'], 
                   color='xkcd:red', marker='o', s=2, lw=0, label='New Version')
    
    for ax in axs:
        ax.plot(rough_me_lon, rough_me_lat, color='xkcd:blue', lw=1)
    
    for ax in axs:
        ax.set(aspect=1, xlim=[360, 0], ylim=[-90, 90])
    plt.show()
    
    breakpoint()
    fig, ax = plt.subplots()
    ax.scatter(comparison_df['x'], comparison_df['y'], color='black', marker='.', s=1)
    ax.scatter(comparison_df['x'], comparison_df['y'], 
               color='red', marker='o', s=6, facecolor='none')
    ax.set(aspect=1, xlim=[20, -20], ylim=[-20, 20],
           xlabel='RA ["]', ylabel='Dec ["]')
    plt.show()
    
    
    
    fig, ax = plt.subplots()
    ax.scatter(comparison_df['lon'], comparison_df['lat'], color='red', marker='o', s=2)
    ax.set(aspect=1, xlim=[0, 360], ylim=[-90, 90], 
           xlabel='Longitude [deg]', ylabel='Latitude [deg]')
    plt.show()
    
    
    
    
    # rev['lon'] = 360 - np.array(event_lon)
    # rev['lat'] = event_lat
    
    fig, ax = plt.subplots()
    ax.scatter(rev['sky_x'], rev['sky_y'], color='black', marker='.', s=1)
    ax.scatter(rev.dropna()['sky_x'], rev.dropna()['sky_y'], 
               color='red', marker='o', s=6, facecolor='none')
    ax.set(aspect=1, xlim=[20, -20], ylim=[-20, 20],
           xlabel='RA ["]', ylabel='Dec ["]')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(rev['lon'], rev['lat'], color='red', marker='o', s=2)
    ax.set(aspect=1, xlim=[0, 360], ylim=[-90, 90], 
           xlabel='Longitude [deg]', ylabel='Latitude [deg]')
    plt.show()
    
    breakpoint()
    # Super events tracks interpolated quantities that we don't need to save
    # sev = super_events = events.copy(deep=True)
    # sev['t_jd'] = Time(sev['t'], format='cxcsec').jd
    # sev['t_mjd'] = Time(sev['t'], format='cxcsec').mjd
    # sev['NPPA'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['NPole_ang'])
    # # sev['NPPD'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['NPole_dist'])
    # sev['ang_width'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['ang_width'])
    # sev['theta_pd'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['PDObsLat'])
    
    
    # # Only keep photons that originate within Jupiter's disk
    # emission_altitude_scaling = (rad_eq_0 + alt)/rad_eq_0
    # sev['sky_R_eq'] = emission_altitude_scaling * sev['ang_width'] / 2
    # sev['sky_R_ro'] = emission_altitude_scaling * np.sqrt(1 - ecc**2) * sev['ang_width'] / 2
    # sev['theta_pc'] = np.arctan2((sev['sky_R_ro']**2) * np.tan(sev['theta_pd']*dtor), 
    #                              (sev['sky_R_eq']**2)) * rtod
    
    # # # define the local time and central meridian latitude (CML) during the observation  
    # # jup_time = (eph_DOYFRAC_jup - evt_DOYFRAC)*86400.0 + tstart
    # # jup_cml_0 = float(eph_jup['PDObsLon'][0]) + j_rotrate * (jup_time - jup_time[0])
    # # interpfunc_cml = interpolate.interp1d(jup_time, jup_cml_0)
    
    # # jup_cml = interpfunc_cml(events['t'])
    # # jup_cml = np.deg2rad(jup_cml % 360)
    # sev['CML'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], np.unwrap(eph_jup['PDObsLon'], period=360)) % 360
    
    
    # find sublat of Jupiter during each Chandra time interval
    # interpfunc_sublat = interpolate.interp1d(jup_time, (sub_obs_lat_jup.astype(float)))
    # jup_sublat = interpfunc_sublat(events['t'])
    
    # find the distance between Jupiter and Chandra throughout the observation, convert to km
    # interpfunc_dist = interpolate.interp1d(jup_time, (eph_jup['delta'].astype(float))*au_to_m*1e-3)
    # jup_dist = interpfunc_dist(events['t'])
    # sev['dist'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['delta']) * au_to_m * 1e-3
    
    # # dist = sum(jup_dist)/len(jup_dist)
    # # kmtoarc = np.rad2deg(1.0/dist)*3.6E3 # convert from km to arc
    # # kmtopixels = kmtoarc/skyx_scaling # convert from km to pixels using defined scale
    # sev['pix_R_eq'] = sev['sky_R_eq'] / np.abs(skyx_scaling)
    # sev['pix_R_ro'] = sev['sky_R_ro'] / np.abs(skyx_scaling)

    # rad_eq = rad_eq_0 * kmtopixels
    # rad_pole = rad_pole_0 * kmtopixels # convert both radii form km -> pixels
    # alt0 = alt * kmtopixels # altitude at which we think emission occurs - agreed in Southampton Nov 15th 2017
    
    
        
    # sev['sky_R_NS'] = apparent_size(sev['sky_R_eq'], sev['sky_R_ro'], sev['theta_pc'])/2
  
    jup_sublat = sev['theta_pd']
    rad_eq = sev['pix_R_eq'].mean()
    dist = sev['dist'].mean()
    alt0 = alt * (sev['pix_R_eq'] / rad_eq_0).mean()
    
    # define the planetocentric S3 coordinates of Jupiter 
    phi1 = np.deg2rad(sum(jup_sublat)/len(jup_sublat)) # The mean sub-observer latitude [radians]
    nn1 = rad_eq/np.sqrt(1.0 - (ecc*np.sin(phi1))**2) # No clue what this is
    p = dist/rad_eq # distance to Jupiter by pixel radius, [km/pix]
    phig = phi1 - np.arcsin(nn1 * ecc**2 * np.sin(phi1)*np.cos(phi1)/p/rad_eq)
    h = p * rad_eq *np.cos(phig)/np.cos(phi1) - nn1
    interpfunc_nppa = interpolate.interp1d(jup_time, (eph_jup['NPole_ang'].astype(float)))
    jup_nppa = interpfunc_nppa(events['t'])
    gamma = np.deg2rad(sum(jup_nppa)/len(jup_nppa))
    omega = 0.0 # !!! 20260126 What's this?
    Del = 1.0 # !!! 20260126 What's this?
    
    
    # =============================================================================
    #     # Do the reverse orthographic projection yourself...
    # =============================================================================
    
    
    
    
 
    
    
    # Compare to original method
    path = '/Users/mrutala/projects/Jupiter_XUV_Comparison/data/CXO/29673/primary/29673_photonlist_full_obs_ellipse.csv'
    comparison_df = pd.read_csv(path, comment='#')
    
    fig, ax = plt.subplots()
    ax.scatter(comparison_df['x'], comparison_df['y'], color='black', marker='.', s=1)
    ax.scatter(comparison_df['x'], comparison_df['y'], 
               color='red', marker='o', s=6, facecolor='none')
    ax.set(aspect=1, xlim=[20, -20], ylim=[-20, 20],
           xlabel='RA ["]', ylabel='Dec ["]')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(comparison_df['lon'], comparison_df['lat'], color='red', marker='o', s=2)
    ax.set(aspect=1, xlim=[0, 360], ylim=[-90, 90], 
           xlabel='Longitude [deg]', ylabel='Latitude [deg]')
    plt.show()
    
    breakpoint()
    
    
    
    
    # Define the planetocentric SIII coordinates
    sublat_average = sev['theta_pc'].mean()
    
    # breakpoint()
    
    #define latitude and longitude grid for entire surface
    lat = np.zeros((int(360) // int(Del))*(int(180) // int(Del) + int(1)))
    lng = np.zeros((int(360) // int(Del))*(int(180) // int(Del) + int(1)))
    j = np.arange(int(180) // int(Del) + int(1)) * int(Del)
    
    for i in range (int(0), int(360)):# // int(Del) - int(1)):
        lat[j * int(360) // int(Del) + i] = (j* int(Del) - int(90))
        lng[j * int(360) // int(Del) + i] = (i* int(Del) - int(0))
    
    # perform coordinate transfromation from plentocentric -> planteographic (taking into account the oblateness of Jupiter
    # when defining the surface features)
    coord_transfo = gca_tools.ltln2xy(alt=alt0, re0=rad_eq_0, rp0=rad_pole_0, 
                                      r=rad_eq, e=ecc, h=h, phi1=phi1, phig=phig, 
                                      lambda0=0.0, p=p, d=dist, gamma=gamma, 
                                      omega=omega, latc=np.deg2rad(lat), lon=np.deg2rad(lng))
    
    # Assign the corrected transformed position of the X-ray emission
    xt = coord_transfo[0]
    yt = coord_transfo[1]
    cosc = coord_transfo[2]
    condition = coord_transfo[3]
    count = coord_transfo[4]
    
    # Find latiutde and lonfitude of the surface features
    laton = lat[condition] + 90
    lngon = lng[condition]
    
    # Define the limb of Jupiter, to ensure only auroral photons are selected for analysis
    # breakpoint()
    # cosmu = gca_tools.findcosmu(rad_eq, rad_pole, phi1, np.deg2rad(lat), np.deg2rad(lng))
    cosmu = gca_tools.findcosmu(sev['sky_R_eq'].mean(), sev['sky_R_ro'].mean(), phi1, np.deg2rad(lat), np.deg2rad(lng))
    limb = np.where(abs(cosmu) < 0.05)
    
    # This next step creates the parameters used to plot what is measured on Jupiter. In the code, I define this as "props" (properties)
    # which has untis of counts/m^2. "timeprops" has units of seconds
    
    # Creating 2D array of the properties and time properties
    props = np.zeros((int(360) // int(Del), int(180) // int(Del) + int(1)))
    timeprops = np.zeros((int(360) // int(Del), int(180) // int(Del) + int(1)))
    n_events = len(events)
    # define a Gaussian PSF for the instrument
    psfn = np.pi*(fwhm / (2.0 * np.sqrt(np.log(2.0))))**2
    
    breakpoint()
    # create a grid for the position of the properties
    latx = np.zeros(n_events)
    lonx = np.zeros(n_events)
    
    # Equations for defining ellipse region
    tilt_ang_rad = np.deg2rad(tilt_ang)
    R_eq_as = (ang_diam/2.)/np.cos(tilt_ang_rad) # equatorial radius of Jupiter in arcsecs
    R_pol_as = R_eq_as * np.sqrt(1 - ecc**2) # polar radius of Jupiter in arcsecs
    
    
    
    # Modernizing...
    # n_events = num
    
    # cxo_ints = []
    sup_props_list = []
    sup_time_props_list = []
    # sup_lat_list = []
    # sup_lon_list = []
    lonj_max = []
    latj_max = []
    # sup_psf_max = []
    # ph_tevts = []
    # ph_xevts = []
    # ph_yevts = []
    # ph_chavts = []
    # ph_sampvts = []; ph_sumampvts = []; ph_pivts = []; ph_ampsfvts = []
    # ph_av1vts = []; ph_av2vts = []; ph_av3vts = []
    # ph_au1vts = []; ph_au2vts = []; ph_au3vts = []
    emiss_evts = []
    ph_cmlevts = []
    psfmax =[]
    
    # =============================================================================
    # NEW STUFF
    # =============================================================================
    # breakpoint()
    
    dtor = np.pi/180
    rtod = 180/np.pi
    
    # Super events tracks interpolated quantities that we don't need to save
    sev = super_events = events.copy(deep=True)
    sev['t_jd'] = Time(sev['t'], format='cxcsec').jd
    sev['t_mjd'] = Time(sev['t'], format='cxcsec').mjd
    sev['NPPA'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['NPole_ang'])
    # sev['NPPD'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['NPole_dist'])
    sev['ang_width'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['ang_width'])
    sev['theta_pd'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['PDObsLat'])
    sev['lambda_pd'] = np.interp(sev['t_jd'], eph_jup['datetime_jd'], eph_jup['PDObsLon'])
    
    # Only keep photons that originate within Jupiter's disk
    emission_altitude_scaling = (rad_eq_0 + alt)/rad_eq_0
    sev['sky_R_eq'] = emission_altitude_scaling * sev['ang_width'] / 2 + (0 * 0.33)
    sev['sky_R_ro'] = emission_altitude_scaling * np.sqrt(1 - ecc**2) * sev['ang_width'] / 2 + (0 * 0.33)
    sev['theta_pc'] = np.arctan2((sev['sky_R_ro']**2) * np.tan(sev['theta_pd']*dtor), 
                                 (sev['sky_R_eq']**2)) * rtod
    
    # # The apparent size along sky North-South is the max. of the rotated ellipse
    # def apparent_size(a, b, theta):
    #     a, b, theta = np.array(a), np.array(b), np.array(theta)
    #     θ = theta * dtor
    #     n = 1000
    #     # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    #     A = np.tile((a**2 * np.sin(θ)**2) + (b**2 * np.cos(θ)**2), (n,1))
    #     B = np.tile(2 * (b**2 - a**2) * np.sin(θ) * np.cos(θ), (n,1))
    #     C = np.tile((a**2 * np.cos(θ)**2) + (b**2 * np.sin(θ)**2), (n,1))
    #     D = 0
    #     E = 0
    #     F = np.tile(- a**2 * b**2, (n,1))
        
    #     # Rather than optimizing, just try a load of test points
    #     x = np.linspace(-a, a, 1000)
    #     if len(x.shape) == 1: x = x[:, np.newaxis]
    #     y0 = (-B*x + np.sqrt(B**2 * x**2 - 4*A*C*x**2 - 4*C*F))/(2 * C)
    #     # y1 = (-B*x - np.sqrt(B**2 * x**2 - 4*A*C*x**2 - 4*C*F))/(2 * C)
        
    #     return 2*np.nanmax(y0, axis=0)
        
    sev['sky_R_NS'] = apparent_size(sev['sky_R_eq'], sev['sky_R_ro'], sev['theta_pc'])/2
    
    # Rotate sky coords into Jupiter frame (so negative angle)
    sev['xr'] = sev['x']*np.cos(-sev['NPPA']*dtor) - sev['y']*np.sin(-sev['NPPA']*dtor)
    sev['yr'] = sev['x']*np.sin(-sev['NPPA']*dtor) + sev['y']*np.cos(-sev['NPPA']*dtor)
    
    # Plot to check photon inclusion
    def plot_rotatedphotons():
        
        # Assuming 100"x100"
        l = np.sqrt(2)*100/2
        tsev = tight_sev = sev.query("(-@l/2 <= xr <= @l/2) & (-@l/2 <= yr <= @l/2)").copy()
        
        # ellipse_width_at_y = 2 * tsev['sky_R_eq'] * np.sqrt(1 - (tsev['yr']/tsev['sky_R_NS'])**2)
        # sky_width_at_y = l - ellipse_width_at_y
        
        # background = np.sum((tsev['xr']/tsev['sky_R_eq'])**2 + (tsev['yr']/tsev['sky_R_NS'])**2 > 1.25)
        disk_area = np.pi * tsev['sky_R_eq'] * tsev['sky_R_NS']
        sky_area = l**2 - disk_area
        
        tsev_on_disk = (tsev['xr']/tsev['sky_R_eq'])**2 + (tsev['yr']/tsev['sky_R_NS'])**2 <= 1
        tsev.loc[tsev_on_disk, 'count_density'] = 1 / disk_area[tsev_on_disk]
        tsev.loc[~tsev_on_disk, 'count_density'] = 1 / sky_area[~tsev_on_disk]
        
        in_bg_bool = (tsev['xr']/tsev['sky_R_eq'])**2 + (tsev['yr']/tsev['sky_R_NS'])**2 > 1.25**2
        bg_area = (l**2 - np.pi * 1.25 * tsev['sky_R_eq'] * 1.25 * tsev['sky_R_NS'])
        tsev.loc[in_bg_bool, 'bg_count_density'] = 1 / bg_area[in_bg_bool]
        
        # Instead of a histogram, 
        # sum the densities in one bin and multiple by bin width
        pixel_bins = np.arange(-50, 50+5, 5)
        
        h_x_total = tsev.groupby(pd.cut(tsev['xr'], bins=pixel_bins), observed=False)['count_density'].sum()*5
        h_x_disk = tsev.groupby(pd.cut(tsev['xr'].loc[tsev_on_disk], bins=pixel_bins), observed=False)['count_density'].sum()*5
        h_x_sky = tsev.groupby(pd.cut(tsev['xr'].loc[~tsev_on_disk], bins=pixel_bins), observed=False)['count_density'].sum()*5
        h_x_bg = tsev.groupby(pd.cut(tsev['xr'].loc[in_bg_bool], bins=pixel_bins), observed=False)['bg_count_density'].sum()*5
        
        h_y_total = tsev.groupby(pd.cut(tsev['yr'], bins=pixel_bins), observed=False)['count_density'].sum()*5
        h_y_disk = tsev.groupby(pd.cut(tsev['yr'].loc[tsev_on_disk], bins=pixel_bins), observed=False)['count_density'].sum()*5
        h_y_sky = tsev.groupby(pd.cut(tsev['yr'].loc[~tsev_on_disk], bins=pixel_bins), observed=False)['count_density'].sum()*5
        h_y_bg = tsev.groupby(pd.cut(tsev['yr'].loc[in_bg_bool], bins=pixel_bins), observed=False)['bg_count_density'].sum()*5
        
        # h_x_total, _ = np.histogram(x_rotated, pixel_bins)
        # h_x_in, _ = np.histogram(x_rotated[in_ellipse_index], pixel_bins)
        # h_x_out, _ = np.histogram(x_rotated[~in_ellipse_index], pixel_bins)
        
        # h_y_total, _ = np.histogram(y_rotated, pixel_bins)
        # h_y_in, _ = np.histogram(y_rotated[in_ellipse_index], pixel_bins)
        # h_y_out, _ = np.histogram(y_rotated[~in_ellipse_index], pixel_bins)
        
        # fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row',
        #                         width_ratios=[4, 1], height_ratios=[4, 1],
        #                         figsize=(4,4))
        fig = plt.figure(figsize=(6,6))
        ax00 = fig.add_axes([0.1, 0.3, 0.6, 0.6])  #[0.1, 0.3, 0.7, 0.9]
        ax10 = fig.add_axes([0.1, 0.1, 0.6, 0.15]) #[0.1, 0.1, 0.7, 0.25]
        ax01 = fig.add_axes([0.75, 0.3, 0.15, 0.6]) #[0.75, 0.3, 0.9, 0.9]
        axs = np.array([[ax00, ax01], [ax10, None]])
        
        # Scatter plot
        axs[0,0].scatter(tsev.loc[tsev_on_disk,'xr'], tsev.loc[tsev_on_disk,'yr'], 
                         marker='.', s=0.5, color='black')
        axs[0,0].scatter(tsev.loc[~tsev_on_disk,'xr'], tsev.loc[~tsev_on_disk,'yr'], 
                         marker='.', s=0.5, color='red')
        axs[0,0].scatter(*tsev.loc[in_bg_bool,['xr', 'yr']].to_numpy().T, 
                         marker='o', s=10, lw=0.5, color='xkcd:orange', facecolor='none')
       
        # X histogram
        axs[1,0].stairs(h_x_total, pixel_bins, color='black', label = 'Total', ls=':')
        axs[1,0].stairs(h_x_disk, pixel_bins, color='C0', label='Jovian')
        axs[1,0].stairs(h_x_sky, pixel_bins, color='C1', label='Sky')
        axs[1,0].stairs(h_x_bg, pixel_bins, color='C4', label='Background')
        
        # Y histogram
        axs[0,1].stairs(h_y_total, pixel_bins, color='black', label = 'Total', ls=':', orientation='horizontal')
        axs[0,1].stairs(h_y_disk, pixel_bins, color='C0', label='Jovian', orientation='horizontal')
        axs[0,1].stairs(h_y_sky, pixel_bins, color='C1', label='Sky', orientation='horizontal')
        axs[0,1].stairs(h_y_bg, pixel_bins, color='C4', label='Background', orientation='horizontal')
        
        
        # Fix x-axes
        for ax in axs[:,0]:
            ax.set(xlim=[40, -40], xlabel='Jupiter-Centered Right Ascension ["]')
        for ax in axs[0,:]:
            ax.set(ylim=[-40, 40], ylabel='Jupiter-Centered Declination ["]')
        
        axs[0,0].xaxis.tick_top()
        axs[0,0].xaxis.set_label_position('top')
        
        axs[0,1].yaxis.tick_right()
        axs[0,1].yaxis.set_label_position('right')
        
        
        
        plt.show()
        
        # breakpoint()
        
    
    plot_rotatedphotons()
    
    
    # After rotating sky, check for photons within Jupiter's disk
    on_disk_bool = (sev['xr']/sev['sky_R_eq'])**2 + (sev['yr']/sev['sky_R_NS'])**2 <= 1
    
    # ellipse_cond = (events['x'] * np.cos(tilt_ang_rad) + events['y'] * np.sin(tilt_ang_rad)) ** 2./(R_eq_as ** 2) + (events['x'] * np.sin(tilt_ang_rad) - events['y'] * np.cos(tilt_ang_rad)) ** 2./(R_pol_as ** 2.) < 1.0
    
    # # Find max PSF 
    # count_cond = pd.Series(index = ellipse_cond.index, data = False)
    # xpi = events['x'] / skyx_scaling
    # ypi = events['y'] / skyx_scaling
    # for k in range(n_events):
    #     cmlpi = (np.rad2deg(jup_cml[k]))#.astype(int)

    #     xtj = xt[condition]
    #     ytj = yt[condition]
    #     latj = (laton.astype(int)) % 180
    #     lonj = ((lngon + cmlpi.astype(int) + 360.0).astype(int)) % 360
    #     dd = np.sqrt((xpi.iloc[k]-xtj)**2 + (ypi.iloc[k]-ytj)**2) * skyx_scaling
    #     psfdd = np.exp(-(dd/ (fwhm / (2.0 * np.sqrt(np.log(2.0)))))**2) / psfn # define PSF of instrument

    #     psf_max_cond = np.where(psfdd == max(psfdd))[0] # finds the max PSF over each point in the grid
    #     count_mx = np.count_nonzero(psf_max_cond)
        
    #     if (count_mx == 1) & (ellipse_cond.iloc[k] == True):
            
    #         # These four need (?) to be assigned in the loop
    #         props[lonj,latj] = props[lonj,latj] + psfdd # assign the 2D PSF to the each point in the grid
    #         emiss = np.rad2deg(np.cos(cosc[condition[psf_max_cond]])) # find the emission angle from each max PSF
                     
    #         emiss_evts.append(emiss[0])
    #         ph_cmlevts.append(cmlpi)
            
    #         psfmax.append(psfdd[psf_max_cond][0])
    #         latj_max.append(latj[psf_max_cond][0])
    #         lonj_max.append(lonj[psf_max_cond][0])

    #         count_cond.iloc[k] = True   
    
    # Find max PSF 
    count_cond = pd.Series(index = on_disk_bool.index, data = False)
    xpi = events['x'] / skyx_scaling
    ypi = events['y'] / skyx_scaling
    for k in range(n_events):
        
        # breakpoint()
        # cmlpi = (np.rad2deg(jup_cml[k]))#.astype(int)
        cmlpi = np.interp(events['t'].iloc[k], sev['t'], sev['lambda_pd'])

        xtj = xt[condition]
        ytj = yt[condition]
        latj = (laton.astype(int)) % 180
        lonj = ((lngon + cmlpi.astype(int) + 360.0).astype(int)) % 360
        dd = np.sqrt((xpi.iloc[k]-xtj)**2 + (ypi.iloc[k]-ytj)**2) * np.abs(skyx_scaling)
        psfdd = np.exp(-(dd/ (fwhm / (2.0 * np.sqrt(np.log(2.0)))))**2) / psfn # define PSF of instrument

        psf_max_cond = np.where(psfdd == max(psfdd))[0] # finds the max PSF over each point in the grid
        count_mx = np.count_nonzero(psf_max_cond)
        # breakpoint()
        if (count_mx == 1) & (on_disk_bool.iloc[k] == True):
            breakpoint()
            # These four need (?) to be assigned in the loop
            props[lonj,latj] = props[lonj,latj] + psfdd # assign the 2D PSF to the each point in the grid
            emiss = np.rad2deg(np.cos(cosc[condition[psf_max_cond]])) # find the emission angle from each max PSF
                     
            emiss_evts.append(emiss[0])
            ph_cmlevts.append(cmlpi)
            
            psfmax.append(psfdd[psf_max_cond][0])
            latj_max.append(latj[psf_max_cond][0])
            lonj_max.append(lonj[psf_max_cond][0])

            count_cond.iloc[k] = True   
            
            
            
            
    
    # Take the subset
    planet_events = events[count_cond].copy()
    
    # Add lat, lon, emiss, cml, and psf
    planet_events.loc[:, 'lat'] = [l - 90. for l in latj_max]
    planet_events.loc[:, 'lon'] = lonj_max
    planet_events.loc[:, 'cml'] = ph_cmlevts
    planet_events.loc[:, 'emiss'] = emiss_evts
    planet_events.loc[:, 'psf'] = psfmax
    
    # Add true time?
    mjd_events = start_time.mjd + (planet_events['t'] - tstart)/(24*60*60)
    planet_events.loc[:, 'mjd'] = mjd_events
    
    filepath = str(obs_dir)+ "/%s_photonlist_full_obs_ellipse.csv" % obs_id
    with open(filepath, 'w') as f:
        f.write('#UNITS:  t(s), x(arcsec), y(arcsec), PHA, samp, sumamps, pi, amp_sf, av1, av2, av3, au1, au2, au3, lat (deg), SIII_lon (deg), CML (deg), emiss (deg), Max PSF, MJD (days) \n')
        planet_events.to_csv(f, header = True, index = False)
        
    # # In principle, the below does not need to be a loop...s
    # for k in range(n_events):
    # # for k in range(0,num-1):
    
    #     # convert (x,y) position to pixels
    #     xpi = events['x'].iloc[k] / skyx_scaling
    #     ypi = events['y'].iloc[k] / skyx_scaling
    
    #     # only select photons that lie within ellipse of Jupiter defined using JPL Horizons data
    #     if (events['x'].iloc[k] * np.cos(tilt_ang_rad) + events['y'].iloc[k] * np.sin(tilt_ang_rad)) ** 2./(R_eq_as ** 2) + (events['x'].iloc[k] * np.sin(tilt_ang_rad) - events['y'].iloc[k] * np.cos(tilt_ang_rad)) ** 2./(R_pol_as ** 2.) < 1.0:
    
    #         cmlpi = (np.rad2deg(jup_cml[k]))#.astype(int)
    
    #         xtj = xt[condition]
    #         ytj = yt[condition]
    #         latj = (laton.astype(int)) % 180
    #         lonj = ((lngon + cmlpi.astype(int) + 360.0).astype(int)) % 360
    #         dd = np.sqrt((xpi-xtj)**2 + (ypi-ytj)**2) * skyx_scaling
    #         psfdd = np.exp(-(dd/ (fwhm / (2.0 * np.sqrt(np.log(2.0)))))**2) / psfn # define PSF of instrument
    
    #         psf_max_cond = np.where(psfdd == max(psfdd))[0] # finds the max PSF over each point in the grid
    #         count_mx = np.count_nonzero(psf_max_cond)
    #         if count_mx != 1: # ignore points where there are 2 cases of the same max PSF
    #             continue
    #         else:
    
    #             props[lonj,latj] = props[lonj,latj] + psfdd # assign the 2D PSF to the each point in the grid
    #             emiss = np.array(np.rad2deg(np.cos(cosc[condition[psf_max_cond]]))) # find the emission angle from each max PSF
                
    #             # record the corresponding photon data at each peak in the grid...
    #             emiss_evts.append(emiss)
    #             ph_cmlevts.append(cmlpi)
    #             ph_tevts.append(events['t'].iloc[k])
    #             ph_xevts.append(events['x'].iloc[k])
    #             ph_yevts.append(events['y'].iloc[k])
    #             ph_chavts.append(events['channel'].iloc[k])
    #             ph_sampvts.append(events['samp'].iloc[k])
    #             ph_sumampvts.append(events['sumamp'].iloc[k])
    #             ph_pivts.append(events['pi'].iloc[k])
    #             ph_ampsfvts.append(events['amp_sf'].iloc[k])
    #             ph_av1vts.append(events['av1'].iloc[k]); ph_av2vts.append(events['av2'].iloc[k]); ph_av3vts.append(events['av3'].iloc[k])
    #             ph_au1vts.append(events['au1'].iloc[k]); ph_au2vts.append(events['au2'].iloc[k]); ph_au3vts.append(events['au3'].iloc[k])
    #             psfmax.append(psfdd[psf_max_cond][0])
    #             latj_max.append(latj[psf_max_cond][0])
    #             lonj_max.append(lonj[psf_max_cond][0])
    #             # ph_tevts_arr = np.array(ph_tevts, dtype=float)
    #             # ph_xevts_arr = np.array(ph_xevts, dtype=float)
    #             # ph_yevts_arr = np.array(ph_yevts, dtype=float)
    #             # ph_chavts_arr = np.array(ph_chavts, dtype=float)
    #             # ph_sampvts_arr = np.array(ph_sampvts, dtype=float); ph_sumampvts_arr = np.array(ph_sumampvts, dtype=float); ph_pivts_arr = np.array(ph_pivts, dtype=float); ph_ampsfvts_arr = np.array(ph_ampsfvts, dtype=float)
    #             # ph_av1vts_arr = np.array(ph_av1vts, dtype=float); ph_av2vts_arr = np.array(ph_av2vts, dtype=float); ph_av3vts_arr = np.array(ph_av3vts, dtype=float)
    #             # ph_au1vts_arr = np.array(ph_au1vts, dtype=float); ph_au2vts_arr = np.array(ph_au2vts, dtype=float); ph_au3vts_arr = np.array(ph_au3vts, dtype=float)
    #             #... and save as text file
    #             # np.savetxt(str(obs_dir)+ "/%s_photonlist_full_obs_ellipse.txt" % obs_id, np.c_[ph_tevts_arr, ph_xevts_arr, ph_yevts_arr, ph_chavts_arr, latj_max, lonj_max, ph_cmlevts, emiss_evts, psfmax, ph_sampvts_arr, ph_sumampvts_arr, ph_pivts_arr, ph_ampsfvts_arr, ph_av1vts_arr, ph_av2vts_arr, ph_av3vts_arr, ph_au1vts_arr, ph_au2vts_arr, ph_au3vts_arr], delimiter=',', header="t(s),x(arcsec),y(arcsec),PHA,lat (deg),SIII_lon (deg),CML (deg),emiss (deg),Max PSF,samp,sumamps,pi,amp_sf,av1,av2,av3,au1,au2,au3", fmt='%s')
                
    
    # breakpoint()
    # np.savetxt(str(obs_dir)+ "/%s_photonlist_full_obs_ellipse.txt" % obs_id, 
    #            np.c_[ph_tevts_arr, ph_xevts_arr, ph_yevts_arr, ph_chavts_arr, latj_max, lonj_max, ph_cmlevts, emiss_evts, psfmax, ph_sampvts_arr, ph_sumampvts_arr, ph_pivts_arr, ph_ampsfvts_arr, ph_av1vts_arr, ph_av2vts_arr, ph_av3vts_arr, ph_au1vts_arr, ph_au2vts_arr, ph_au3vts_arr], delimiter=',', header="t(s),x(arcsec),y(arcsec),PHA,lat (deg),SIII_lon (deg),CML (deg),emiss (deg),Max PSF,samp,sumamps,pi,amp_sf,av1,av2,av3,au1,au2,au3", fmt='%s')
    
    # Add comment to CSV with units
    
    # header="t(s),x(arcsec),y(arcsec),PHA,lat (deg),
    # SIII_lon (deg),CML (deg),emiss (deg),Max PSF,samp,sumamps,pi,amp_sf,av1,
    # av2,av3,au1,au2,au3", fmt='%s')

    # breakpoint() # Are we saving line by line? WORSE! resaving each line hahahahaha
                
    # effectively, do the same idea except for exposure time
    obs_start_times = events['t'].min()
    obs_end_times = events['t'].max()
    
    interval = obs_end_times - obs_start_times
    
    if interval > 1000.0:
        step = interval/100.0
    elif interval > 100.0:
        step = interval/10.0
    else:
        step = interval/2.0
    
    time_vals = np.arange(round(int(interval/step)))*step + step/2 + obs_start_times
    
    interpfunc_time_cml = interpolate.interp1d(jup_time,jup_cml_0)
    time_cml = interpfunc_time_cml(time_vals)
    
    
    
    for j in range(0, len(time_vals)):
        timeprops[((lngon + time_cml[j].astype(int))%360).astype(int),laton.astype(int)] = timeprops[((lngon + time_cml[j].astype(int))%360).astype(int),laton.astype(int)] + step
    
    
    # record the fluxes and position of the max PSFs
    sup_props_list = props
    sup_time_props_list = timeprops
    
    np.save(str(obs_dir) + f'/{obs_id}_sup_props_list.npy', np.array(sup_props_list))
    np.save(str(obs_dir) + f'/{obs_id}_sup_time_props_list.npy', np.array(sup_time_props_list))
    
    return planet_events

def get_JupiterPatch(r_eq, nppa, sublat, **kwargs):
    import matplotlib.patches as patches

    # Equations for defining ellipse region
    # tilt_ang_rad = np.deg2rad(nppa)
    
    # The rotational axis of Jupiter is flattened
    r_rot = r_eq * (1 - flattening)
    
    # The *apparent* North-South axis is slightly larger than the rotation,
    # if the planet is view from above/below the equator
    r_NS_apparent = apparent_size(r_eq, r_rot, sublat)/2
    
    limb_ellipse = patches.Ellipse(
        xy = (0,0), width = r_eq*2, height = r_NS_apparent*2, angle = -nppa, 
        **kwargs)
    
    return limb_ellipse

# The apparent size along sky North-South is the max. of the rotated ellipse
def apparent_size(a, b, theta):
    a, b, theta = np.array(a), np.array(b), np.array(theta)
    θ = np.deg2rad(theta)
    n = 1000
    # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    A = np.tile((a**2 * np.sin(θ)**2) + (b**2 * np.cos(θ)**2), (n,1))
    B = np.tile(2 * (b**2 - a**2) * np.sin(θ) * np.cos(θ), (n,1))
    C = np.tile((a**2 * np.cos(θ)**2) + (b**2 * np.sin(θ)**2), (n,1))
    D = 0
    E = 0
    F = np.tile(- a**2 * b**2, (n,1))
    
    # Rather than optimizing, just try a load of test points
    x = np.linspace(-a, a, 1000)
    if len(x.shape) == 1: x = x[:, np.newaxis]
    y0 = (-B*x + np.sqrt(B**2 * x**2 - 4*A*C*x**2 - 4*C*F))/(2 * C)
    # y1 = (-B*x - np.sqrt(B**2 * x**2 - 4*A*C*x**2 - 4*C*F))/(2 * C)
    
    return 2*np.nanmax(y0, axis=0)

def psf_from_header(header, obs_dir):
    import subprocess 
    
    # !!!! Hardcoded marx profile
    default_pfile = '/Users/mrutala/miniconda3/envs/ciao-4.17/share/marx/pfiles/marx.par'
    
    output_dir = obs_dir + '/marx_psf'
    
    print("Beginning MARX simulation...")
    
    # Default SIM X/Z values from https://cxc.harvard.edu/ciao/threads/marx_sim/
    # NB: These do not include spectral defaults
    sim_x_defaults = {'ACIS-I': -0.78234819833843,
                      'ACIS-S': -0.68426746699586,
                      'HRC-I': -1.0402925884,
                      'HRC-S': -1.5333365632,}
    sim_z_defaults = {'ACIS-I': -233.5924630914,
                      'ACIS-S': -190.1325231040,
                      'HRC-I': 126.9854943053,
                      'HRC-S': 250.4559758190}
    param_dict = {
        'NumRays': -100000,
        'dNumRays': 100000,
        'TStart': header['TSTART'],
        'ExposureTime': 0,
        'OutputDir': output_dir,
        
        # Science Instrument set up and control
        'MirrorType': "HRMA",
        'GratingType': "NONE",
        'DetectorType': "HRC-S",
        'DetOffsetX': header['SIM_X'] - sim_x_defaults[header['DETNAM']],
        'DetOffsetZ': header['SIM_Z'] - sim_z_defaults[header['DETNAM']],
        
        'SourceFlux': 0.01, #incoming ray flux (photons/sec/cm^2)
        'SpectrumType': "FLAT",
        
        #  Energy limits  (for flat spectrum model)
        'MinEnergy': 0.03,  #,0.03,12.0,"MIN ray energy (keV)"
        'MaxEnergy': 4.0,  # ,0.03,12.0,"MAX ray energy (keV)"
    
        'SourceRA': header['RA_TARG'],
        'SourceDEC': header['DEC_TARG'],
        
        'RA_Nom': header['RA_NOM'],     # "RA_NOM for dither (degrees)"
        'Dec_Nom': header['DEC_NOM'],   # "DEC_NOM for dither (degrees)"
        'Roll_Nom': header['ROLL_NOM'], # "ROLL_NOM for dither (degrees)"
    
        'DitherModel': 'INTERNAL',
        }
    
    marxcall = ['marx'] + ['@@' + default_pfile] 
    marxcall += ['{0}={1}'.format(k, v) for k, v in param_dict.items()]
    marxcall = ' '.join(marxcall)
    out = subprocess.run(marxcall, shell=True, capture_output=True)
    
    marx2fitscall = ['marx2fits'] + [output_dir, obs_dir + '/marx_psf.fits']
    marx2fitscall = ' '.join(marx2fitscall)
    out = subprocess.run(marx2fitscall, shell=True, capture_output=True)
    
    with astropy.io.fits.open(obs_dir + '/marx_psf.fits') as hdul:
        psf_data = hdul[1].data
        psf_hdr =  hdul[1].header
        
    # Get the PSF centering and scaling terms
    psfx_center, psfx_scaling = psf_hdr['TCRPX9'], psf_hdr['TCDLT9']
    psfy_center, psfy_scaling = psf_hdr['TCRPX10'], psf_hdr['TCDLT10']
    
    # Center and scale the PSF
    # Shift the PSF to zero, as this has already been done to the obs.
    bigxpsf = (psf_data.X - psfx_center) * psfx_scaling * 3600
    bigypsf = (psf_data.Y - psfy_center) * psfy_scaling * 3600
    bigxpsf, bigypsf = bigxpsf - bigxpsf.mean(), bigypsf - bigypsf.mean()
    
    # The PSF has very distant outliers. Before fitting, trim it to 99%
    cutoffs_arcsecs = np.arange(0.1, 10, 0.1)
    percent_enclosed = [((bigxpsf**2 + bigypsf**2) < c**2).sum()/len(bigxpsf) for c in cutoffs_arcsecs]
    cutoff_99p = np.interp(0.99, percent_enclosed, cutoffs_arcsecs)
    
    cutoff_indx = (bigxpsf**2 + bigypsf**2) < cutoff_99p**2
    bigxpsf = bigxpsf[cutoff_indx]
    bigypsf = bigypsf[cutoff_indx]
        
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.975, top=0.975)
    ax.scatter(bigxpsf, bigypsf, s=0.5, alpha=0.1, lw=0)
    ax.scatter([0], [0], color='black', marker='+', s=50)
    
    ax.set(aspect=1, 
           xlim=[5, -5], xlabel='Target-centered Sky X ["]', 
           ylim=[-5, 5], ylabel='Target-centered Sky Y ["]')
    
    
    # Try to make a confidence ellipse
    psf_cov = np.cov(bigxpsf, bigypsf)
    eigenval, eigenvec = np.linalg.eig(psf_cov)
    angle = np.rad2deg(np.arctan2(max(eigenval) - psf_cov[0,0], psf_cov[0,1]))
    
    p = patches.Ellipse((0,0), max(eigenval), min(eigenval), angle=angle)
    ax.add_patch(p)
    
    ax.plot([0, eigenvec.T[0][0]*eigenval[0]], [0, eigenvec.T[0][1]*eigenval[0]], color='xkcd:red', lw=2)
    ax.plot([0, eigenvec.T[1][0]*eigenval[1]], [0, eigenvec.T[1][1]*eigenval[1]], color='xkcd:green', lw=2)
    
    # theta = np.arctan2(*eigenvec.T[eigenval.argmax()])
    
    # ax.scatter(
              
    #            s=0.5, alpha=0.9, lw=0, color='xkcd:cyan')
    
    # x_r = bigxpsf*np.cos(theta) + bigypsf*np.sin(theta)
    # y_r = - bigxpsf*np.sin(theta) + bigypsf*np.cos(theta)
    # (x_r**2 / )
    
    # p = patches.Ellipse((0,0), width = max(eigenval), height = min(eigenval), angle = np.rad2deg(theta))
    # ax.add_patch(p)
    # plt.show()
    # breakpoint()
    return psf_cov

def get_UVISPolygon():
    # Copied-and-pasted definitions from OBSVIS
    # With target coordinates = (180,0) & roll = 0 & in RA/DEC [deg]
    polygon = (179.6937151,0.0587638,179.6938477,-0.0483477,180.2506278,-0.0475864,180.2504940,0.0595286)
    lines = [(179.6938092,-0.0170641,179.9044965,-0.0167775), 
             (179.9044965,-0.0167775,179.9044021,0.0590508), 
             (180.0844517,0.0592984,180.0845463,-0.0165308), 
             (180.0845463,-0.0165308,180.2505888,-0.0163018)
             ]
    
    polygon_xy = np.array([polygon[::2] + (polygon[0],), polygon[1::2] + (polygon[1],)])
    lines_xy = np.array([[l[::2], l[1::2]] for l in lines])
    
    polygon_xy[0] -= 180
    lines_xy[:,0,:] -= 180
    
    chip2_polygon_xy = polygon_xy
    
    # Convert to a UVIS polygon
    uvis_polygon_x = np.array([l[0] for l in lines_xy]).flatten()
    uvis_polygon_y = np.array([l[1] for l in lines_xy]).flatten()
    
    missing_vert_index = polygon_xy[1] < uvis_polygon_y.min()
    
    uvis_polygon_x = np.append(uvis_polygon_x, [*polygon_xy[0][missing_vert_index][::-1], uvis_polygon_x[0]])
    uvis_polygon_y = np.append(uvis_polygon_y, [*polygon_xy[1][missing_vert_index][::-1], uvis_polygon_y[0]])
    uvis_polygon_xy = np.array([uvis_polygon_x, uvis_polygon_y])
    
    return uvis_polygon_xy, chip2_polygon_xy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file name.')
    
    args = parser.parse_args()
    config = 'config.ini' if args.config is None else args.config
    
    _ = go_chandra(config=config)