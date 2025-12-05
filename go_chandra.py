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

from astropy.time import Time                   #convert between different time coordinates
from astropy.time import TimeDelta              #add/subtract time intervals 
from astroquery.jplhorizons import Horizons     #automatically download ephemeris 

def go_chandra(acis=None, obs_id=None, obs_dir=None, config=None):
    
    # Assumptions 
    j_rotrate = np.rad2deg(1.758533641E-4) # Jupiter's rotation period
    # scale = 0.13175 # scale used when observing Jupiter using Chandra - in units of arcsec/pixel
    fwhm = 0.8 # FWHM of the HRC-I point spread function (PSF) - in units of arcsec
    psfsize = 25 # size of PSF used - in units of arcsec
    alt = 400 # altitude where X-ray emission is assumed to occur in Jupiter's ionosphere - in units of km
    
    rad_eq_0 = 71492.0 # radius of equator in km
    rad_pole_0 = 66854.0 # radius of poles in km
    ecc = np.sqrt(1.0-(rad_pole_0/rad_eq_0)**2) # oblateness of Jupiter 
    
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
    
    # File is then read in with relevant header information extracted:
    hdulist = astropy.io.fits.open(corrected_event_filepath, dtype=float)
    # matplotlib.rcParams['agg.path.chunksize'] = 10000 # !!!! No idea why this was here
    
    img_events  = hdulist['EVENTS'].data # event file data
    
    bigtime     = hdulist['EVENTS'].data['time'] # time
    bigxarr     = hdulist['EVENTS'].data['X'] # x position of photons
    bigyarr     = hdulist['EVENTS'].data['Y'] # y position of photons
    bigchannel  = hdulist['EVENTS'].data['pha'] # pha channel the photons were found in
    sumamps     = hdulist['EVENTS'].data['sumamps'] # reading in sumamps figure
    samp        = hdulist['EVENTS'].data['samp'] # reading in samp figure
    pi_cal      = hdulist['EVENTS'].data['pi']
    
    # reading in amplifier signals 
    av1 = hdulist['EVENTS'].data['av1']
    av2 = hdulist['EVENTS'].data['av2']
    av3 = hdulist['EVENTS'].data['av3']
    
    au1 = hdulist['EVENTS'].data['au1']
    au2 = hdulist['EVENTS'].data['au2']
    au3 = hdulist['EVENTS'].data['au3']
    amp_sf = hdulist['EVENTS'].data['amp_sf'] # reading in amplifier scaling factor
    
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
    
    # Close the fits file, freeing memory
    hdulist.close()
    
    # Horizons search code courtesy of Brad Snios
    # The start and end times are taken from the header
    start_time = Time(tstart, format='cxcsec')
    stop_time = Time(tend, format='cxcsec')
    delta_time = '1m'
    eph_jup = gca_tools.fetch_ephemerides_fromCXO(start_time, stop_time, delta_time)
    
    
    # !!!!! Untouched
    # Extracts relevent information needed from ephermeris file
    cml_spline_jup = scipy.interpolate.UnivariateSpline(eph_jup['datetime_jd'], eph_jup['PDObsLon'],k=1)
    lt_jup = eph_jup['lighttime']
    sub_obs_lon_jup = eph_jup['PDObsLon']
    sub_obs_lat_jup = eph_jup['PDObsLat']
    
    # Adding angular diameter from JPL Horizons to use later to define radius of circular region within which photons are kept
    ang_diam = max(eph_jup['ang_width'])
    
    # Also adding tilt angle of Jupiter with respect to true North Pole
    tilt_ang = np.mean(eph_jup['NPole_ang'])
    
    # saving angular diameter and tilt angle in text file in order to plot ellipse in post-processing
    np.savetxt(str(obs_dir) + f'/{obs_id}_JPL_ellipse_vals.txt', np.c_[ang_diam, tilt_ang], delimiter=',', header='angular diameter (arcsec),tilt angle (deg)', fmt='%s')
    
    eph_dates = pd.to_datetime(eph_jup['datetime_str'])
    eph_dates = pd.DatetimeIndex(eph_dates)
    eph_doy = np.array(eph_dates.strftime('%j')).astype(int)
    eph_hours = eph_dates.hour
    eph_minutes = eph_dates.minute
    eph_seconds = eph_dates.second
    
    eph_DOYFRAC_jup = gca_tools.doy_frac(eph_doy, eph_hours, eph_minutes, eph_seconds) # DOY fraction from ephermeris data
    
    jup_time = (eph_DOYFRAC_jup - evt_DOYFRAC)*86400.0 + tstart # local tiem of Jupiter
    
    # !!!!! END Untouched
    
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
    skyx_center = img_head['TCRPX'+keyx[5:]]
    skyy_center = img_head['TCRPX'+keyy[5:]]
    skyx_scaling = np.abs(img_head['TCDLT'+keyx[5:]] * 3600) # in "/pixel
    skyy_scaling = np.abs(img_head['TCDLT'+keyy[5:]] * 3600) # in "/pixel
    
    # !!!! NB: skyx_scaling is negative, introducing a flip in the image...
    # !!!! I don't know if this is potentially desirable or not...
    
    bigxarr_region = (bigxarr - skyx_center) * skyx_scaling
    bigyarr_region = (bigyarr - skyy_center) * skyy_scaling
    
    # storing all photon data in text file - need this to calculate area for samp distributions later on
    # np.savetxt(str(obs_dir) + r"/%s_all_photons.txt" % obs_id, np.c_[bigxarr_region, bigyarr_region, bigtime, bigchannel, samp, sumamps, pi_cal, amp_sf, av1, av2, av3, au1, au2, au3])
    
    # Equations for defining ellipse region
    tilt_ang_rad = np.deg2rad(tilt_ang)
    R_eq_as = (ang_diam/2.)/np.cos(tilt_ang_rad) # equatorial radius of Jupiter in arcsecs
    R_pol_as = R_eq_as * np.sqrt(1 - ecc**2) # polar radius of Jupiter in arcsecs
    
    
    # define the x, y, and pha channel limits 
    xlimits, ylimits = [-50,50], [-50,50]
    cha_min = 0
    cha_max = max(bigchannel)
    
    # the photon data is stored in a pandas dataframe 
    evt_df = pd.DataFrame({'time': bigtime, 'x': bigxarr, 'y': bigyarr, 'pha': bigchannel})
    
    # defines the region the photons will be selected from
    indx = gca_tools.select_region(xlimits[0], xlimits[1],ylimits[0], ylimits[1],bigxarr_region,bigyarr_region,bigchannel,cha_min,cha_max)
    
    # find the x and y position of the photons
    x_ph = bigxarr_region[indx]
    y_ph = bigyarr_region[indx]
    
    # plots the selected region (sanity check: Jupiter should be in the centre)
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Plot (circular) region for extraction
    import matplotlib.patches as patches
    # circle = patches.Circle((0, 0), ang_diam/2, color='xkcd:peach', alpha=0.66)
    limb_ellipse = patches.Ellipse((0,0), R_eq_as*2, R_pol_as*2, angle=tilt_ang, 
                           edgecolor='red', facecolor='xkcd:peach', alpha=0.50, linewidth=3)
    ax.add_patch(limb_ellipse)
    
    ax.scatter(x_ph, y_ph, marker='.', s=1, linestyle='None', color='xkcd:navy blue')
    ax.set(title='Selected Region (ObsID %s)' % obs_id, 
            xlim = xlimits, ylim = ylimits)
    print('')
    print('')
    print('Once you are happy with the selected region, close the figure window to continue analysis')
    print('')
    print('')
    plt.show()
    
    # saves the selected region as a text file
    np.savetxt(str(obs_dir) + r"/%s_selected_region_ellipse.txt" % obs_id, np.c_[x_ph, y_ph, bigtime[indx], bigchannel[indx], samp[indx], sumamps[indx], pi_cal[indx], amp_sf[indx], av1[indx], av2[indx], av3[indx], au1[indx], au2[indx], au3[indx]])
    
    ph_data = astropy.io.ascii.read(str(obs_dir) + r"/%s_selected_region_ellipse.txt" % obs_id) # read in the selected region data and...
    ph_time = ph_data['col3'] #... define the time column
    
    # photon times are turned into an array and converted to datetime format
    # np_times = np.array(ph_time)
    # timeincxo = Time(np_times, format='cxcsec')
    # chandra_evt_time = timeincxo.iso
    # Chandra time then converted to a plotable format
    # chandra_evt_time = Time(chandra_evt_time, format='iso', out_subfmt='date_hm')
    # plot_time = Time.to_datetime(chandra_evt_time)
    # print('')
    # print('All observation will be analysed')
    
    # =========================================================================
    # Coordinate Transform for the whole Observation
    # =========================================================================
    
    # 
    
    # # perform the coordinate transformation for entire observation
    # tevents = ph_data['col3']
    # xevents = ph_data['col1']
    # yevents = ph_data['col2']
    # chaevents = ph_data['col4']
    # sampevents = ph_data['col5']; sumampsevents = ph_data['col6']; pievents = ph_data['col7']; ampsfevents = ph_data['col8']
    # av1events = ph_data['col9']; av2events = ph_data['col10']; av3events = ph_data['col11']
    # au1events = ph_data['col12']; au2events = ph_data['col13']; au3events = ph_data['col14']
    
    # All of the above simply writes the subset to disk, reads it back in, then reassigns all the columns to variables...
    # tevents = bigtime[indx]
    # xevents = x_ph
    # yevents = y_ph
    # chaevents = bigchannel[indx]
    # sampevents = samp[indx]
    # sumampevents = sumamps[indx]
    # pievents = pi_cal[indx]
    # ampsfevents = amp_sf[indx]
    # av1events = av1[indx]
    # av2events = av2[indx]
    # av3events = av3[indx]
    # au1events = au1[indx]
    # au2events = au2[indx]
    # au3events = au3[indx]
    
    events = pd.DataFrame({'t': bigtime[indx].byteswap().newbyteorder(), 
                           'x': x_ph.byteswap().newbyteorder(), 
                           'y': y_ph.byteswap().newbyteorder(),
                           'channel': bigchannel[indx].byteswap().newbyteorder(),
                           'samp': samp[indx].byteswap().newbyteorder(),
                           'sumamp': sumamps[indx].byteswap().newbyteorder(),
                           'pi': pi_cal[indx].byteswap().newbyteorder(),
                           'amp_sf': amp_sf[indx].byteswap().newbyteorder(),
                           'av1': av1[indx].byteswap().newbyteorder(), 
                           'av2': av2[indx].byteswap().newbyteorder(), 
                           'av3': av3[indx].byteswap().newbyteorder(),
                           'au1': au1[indx].byteswap().newbyteorder(), 
                           'au2': au2[indx].byteswap().newbyteorder(), 
                           'au3': au3[indx].byteswap().newbyteorder()})
    
    # =============================================================================
    # SIII Coordinate Transformation
    # =============================================================================
    
    # define the local time and central meridian latitude (CML) during the observation  
    jup_time = (eph_DOYFRAC_jup - evt_DOYFRAC)*86400.0 + tstart
    jup_cml_0 = float(eph_jup['PDObsLon'][0]) + j_rotrate * (jup_time - jup_time[0])
    interpfunc_cml = interpolate.interp1d(jup_time, jup_cml_0)
    
    jup_cml = interpfunc_cml(events['t'])
    jup_cml = np.deg2rad(jup_cml % 360)
    
    # find the distance between Jupiter and Chandra throughout the observation, convert to km
    interpfunc_dist = interpolate.interp1d(jup_time, (eph_jup['delta'].astype(float))*au_to_m*1e-3)
    jup_dist = interpfunc_dist(events['t'])
    dist = sum(jup_dist)/len(jup_dist)
    kmtoarc = np.rad2deg(1.0/dist)*3.6E3 # convert from km to arc
    kmtopixels = kmtoarc/skyx_scaling # convert from km to pixels using defined scale

    rad_eq = rad_eq_0 * kmtopixels
    rad_pole = rad_pole_0 * kmtopixels # convert both radii form km -> pixels
    alt0 = alt * kmtopixels # altitude at which we think emission occurs - agreed in Southampton Nov 15th 2017
    
    # find sublat of Jupiter during each Chandra time interval
    interpfunc_sublat = interpolate.interp1d(jup_time, (sub_obs_lat_jup.astype(float)))
    jup_sublat = interpfunc_sublat(events['t'])
    
    # define the planetocentric S3 coordinates of Jupiter 
    phi1 = np.deg2rad(sum(jup_sublat)/len(jup_sublat))
    nn1 = rad_eq/np.sqrt(1.0 - (ecc*np.sin(phi1))**2)
    p = dist/rad_eq
    phig = phi1 - np.arcsin(nn1 * ecc**2 * np.sin(phi1)*np.cos(phi1)/p/rad_eq)
    h = p * rad_eq *np.cos(phig)/np.cos(phi1) - nn1
    interpfunc_nppa = interpolate.interp1d(jup_time, (eph_jup['NPole_ang'].astype(float)))
    jup_nppa = interpfunc_nppa(events['t'])
    gamma = np.deg2rad(sum(jup_nppa)/len(jup_nppa))
    omega = 0.0
    Del = 1.0
    
    #define latitude and longitude grid for entire surface
    lat = np.zeros((int(360) // int(Del))*(int(180) // int(Del) + int(1)))
    lng = np.zeros((int(360) // int(Del))*(int(180) // int(Del) + int(1)))
    j = np.arange(int(180) // int(Del) + int(1)) * int(Del)
    
    for i in range (int(0), int(360)):# // int(Del) - int(1)):
        lat[j * int(360) // int(Del) + i] = (j* int(Del) - int(90))
        lng[j * int(360) // int(Del) + i] = (i* int(Del) - int(0))
    
    # perform coordinate transfromation from plentocentric -> planteographic (taking into account the oblateness of Jupiter
    # when defining the surface features)
    coord_transfo = gca_tools.ltln2xy(alt=alt0, re0=rad_eq_0, rp0=rad_pole_0, r=rad_eq, e=ecc, h=h, phi1=phi1, phig=phig, lambda0=0.0, p=p, d=dist, gamma=gamma,            omega=omega, latc=np.deg2rad(lat), lon=np.deg2rad(lng))
    
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
    cosmu = gca_tools.findcosmu(rad_eq, rad_pole, phi1, np.deg2rad(lat), np.deg2rad(lng))
    limb = np.where(abs(cosmu) < 0.05)
    
    # This next step creates the parameters used to plot what is measured on Jupiter. In the code, I define this as "props" (properties)
    # which has untis of counts/m^2. "timeprops" has units of seconds
    
    # Creating 2D array of the properties and time properties
    props = np.zeros((int(360) // int(Del), int(180) // int(Del) + int(1)))
    timeprops = np.zeros((int(360) // int(Del), int(180) // int(Del) + int(1)))
    n_events = len(events)
    # define a Gaussian PSF for the instrument
    psfn = np.pi*(fwhm / (2.0 * np.sqrt(np.log(2.0))))**2
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
    
    
    
    # Check interior to planet's limb:
    ellipse_cond = (events['x'] * np.cos(tilt_ang_rad) + events['y'] * np.sin(tilt_ang_rad)) ** 2./(R_eq_as ** 2) + (events['x'] * np.sin(tilt_ang_rad) - events['y'] * np.cos(tilt_ang_rad)) ** 2./(R_pol_as ** 2.) < 1.0
    
    # Find max PSF 
    count_cond = pd.Series(index = ellipse_cond.index, data = False)
    xpi = events['x'] / skyx_scaling
    ypi = events['y'] / skyx_scaling
    for k in range(n_events):
        cmlpi = (np.rad2deg(jup_cml[k]))#.astype(int)

        xtj = xt[condition]
        ytj = yt[condition]
        latj = (laton.astype(int)) % 180
        lonj = ((lngon + cmlpi.astype(int) + 360.0).astype(int)) % 360
        dd = np.sqrt((xpi.iloc[k]-xtj)**2 + (ypi.iloc[k]-ytj)**2) * skyx_scaling
        psfdd = np.exp(-(dd/ (fwhm / (2.0 * np.sqrt(np.log(2.0)))))**2) / psfn # define PSF of instrument

        psf_max_cond = np.where(psfdd == max(psfdd))[0] # finds the max PSF over each point in the grid
        count_mx = np.count_nonzero(psf_max_cond)
        
        if (count_mx == 1) & (ellipse_cond.iloc[k] == True):
            
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file name.')
    
    args = parser.parse_args()
    config = 'config.ini' if args.config is None else args.config
    
    _ = go_chandra(config=config)