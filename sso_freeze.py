# -*- coding: utf-8 -*-
"""
# As described in *Weigt et al. 2020* The following code is a script that takes Gladstone's stop_hrci and translates it into python. 

# This code has been split up into the different section for each different functions carried out.
# 
# <i>End Goal </i>: To simply enter the OBSid -> gather the selected files -> apply ''sso_freeze'' -> output the corrected file
# to a different text file
# 
# Sections are as follows:
# 
# *1)* Reading in the Chandra events file (uncorrected) and extraxt relevant header info </br>
# 
# *2)* Reading in the orbit empheris file and extract relevant header info </br>
# 
# *3)* Read in the ''chandra_horizons file'', extract relevant info and interpolate the data </br>
# 
# *4)* Apply the correction to allow the photons to be tracked to their position on Jupiter </br>
# 
# *5)* The new positions of the photons replace the uncorrected positions from the fits file and is written to a new fits file. </br>
# 

# #### <u><i> Hardwire locations </u></i> 
# 
# In the cells below, there are various hardwire locations that need to be inputted (hopefully once this code is properly optimised, there should be no need for this).
# 
# The hardwire locations are as follows:
# 
# <b>EDIT:</b>
# 
# *Section 1)*: <b>evt_location</b> -> enter path of the event file to be corrected and orbit ephemeris file
# 
# *Section 3)*: <b>eph_location</b> -> enter path of the chandra_horizons2000 file used (from folder of horizons2000 files) 
# 
# *Section 5)*: The save file will now be saved to the same location as the original event and orbit ephemeris file under the name "\hrcf%s_pytest_evt2.fits" where %s is the obs_id (automatically inputted when file is saved).

#Purpose: Read in Chandra event file and empheris files. Correct event file by time-tagging photons to their position
#on Jupiter. New fits file should produce a projection of the x-rays on Jupiter.
#Category: Chandra fits file correction (Jupiter)

@history: 
    Adapted from Randy Gladstone's 'stop_hrci' IDL script. (???)
    Translated into a function and generalized by MJR (2025)

@authors: 
    Dale Weigt (D.M.Weigt@soton.ac.uk), 
    Will Dunn, 
    Brad Snios, 
    Ron Elsner, 
    Caitríona Jackman, 
    Peter Ford, 
    Ralph Kraft, 
    Seán McEntee, 
    Graziella Branduardi-Raymont,
    Matthew J. Rutala
"""

"""All the relevant packages are imported for code below"""

import numpy as np
import pandas as pd
import os
# from astropy.io import fits as pyfits
import astropy
from astropy.time import Time                   #convert between different time coordinates
from astropy.time import TimeDelta              #add/subtract time intervals 
from astropy import units as u
from scipy import interpolate
import datetime

# import the defined functions to analysis Chandra data nad perfrom coordinate transformations
import go_chandra_analysis_tools as gca_tools

"""Reading in the config.ini file containing any hard wired inputs """
import configparser

def find_event_filepath(acis, obs_id, obs_dir, suffix='evt2.fits'):
    """
    Spun off from sso_freeze so other functions can call this directly
    Rather than constantly reinventing the wheel
    """
    event_filepath = []
    
    if acis == 'y':
        for file in os.listdir(str(obs_dir)):
            if file.startswith("acisf") and file.endswith(suffix):
                event_filepath.append(os.path.join(str(obs_dir), file))
    else:
        for file in os.listdir(str(obs_dir)):
            if file.startswith("hrcf"+obs_id) and file.endswith(suffix):
                event_filepath.append(os.path.join(str(obs_dir), file))
    
    event_filepath = event_filepath[0]
    return event_filepath
    
def find_orbit_filepath(acis, obs_id, obs_dir):
    """
    Spun off from sso_freeze so other functions can call this directly
    Rather than constantly reinventing the wheel
    """
    orbit_filepath = []
    
    for file in os.listdir(str(obs_dir)):
        if file.startswith("orbit") and file.endswith("eph1.fits") or file.endswith("eph0.fits"):
            orbit_filepath.append(os.path.join(str(obs_dir), file))
                
    orbit_filepath = orbit_filepath[0]
    return orbit_filepath

def sso_freeze(config=None, acis=None, obs_id=None, obs_dir=None):

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
    
    # =============================================================================
    # Reading in the Chandra events file (uncorrected) and extract relevant header info
    # =============================================================================
    
    # # The code below reads in the Chandra event file and extracts the relevant header info needed later
    # evt_location = []
    # if ACIS == 'y':
    #     for file in os.listdir(str(folder_path)):
    #         if file.startswith("acisf") and file.endswith("evt2.fits"):
    #             evt_location.append(os.path.join(str(folder_path), file))
    # else:
    #     for file in os.listdir(str(folder_path)):
    #         if file.startswith("hrcf") and file.endswith("evt2.fits"):
    #             evt_location.append(os.path.join(str(folder_path), file))
    
    event_filepath = find_event_filepath(acis, obs_id, obs_dir)
    
    # Read the event file and access data and header with context manager
    with astropy.io.fits.open(event_filepath) as hdul:
        event_data = hdul[1].data 
        event_hdr = hdul[1].header
        
        
    event_time = event_data['TIME'] # Spaceraft time of photon event
    event_x = event_data['X'] # Sky X coordinate of photon event
    event_y = event_data['Y'] # Sky Y coordinate of photon event
    
    tstart = event_hdr['TSTART'] # Start time of obs. in s/c time
    tstop = event_hdr['TSTOP'] # Stop time of obs. in s/c time
    # obs_id = event_hdr['OBS_ID']
    start_date = event_hdr['DATE-OBS'] # Start time of obs. as str
    RA_0 = event_hdr['RA_NOM'] # RA origin of target at the start of the observation
    DEC_0 = event_hdr['DEC_NOM'] # DEC origin of target at the start of the observation
    
    event_date = pd.to_datetime(start_date) # converts the date to a timestamp - to allow the time to be separated and used for
    # calculation of the DOY and DOYFRAC
    event_hour = event_date.hour # hour of observation
    event_doy = event_date.strftime('%j') # doy of observation
    event_mins = event_date.minute # minute of observation
    event_secs = event_date.second # second of observation
    event_DOYFRAC = gca_tools.doy_frac(float(event_doy), float(event_hour), float(event_mins), float(event_secs)) # calculating the DOYFRAC
    chand_time = (event_time - tstart)/86400.0 # calculating time cadence of chandra...
    doy_chandra = chand_time + event_DOYFRAC #... to calculate the DOY of chandra
    
    # =============================================================================
    # Reading in the orbit empheris file and extract relevant header info 
    # All of this is done with ephemerides from Horizons now
    # =============================================================================
    
    # orbit_filepath = find_orbit_filepath(ACIS, obs_id, folder_path)
    # with astropy.io.fits.open(orbit_filepath) as hdul:
    #     # orbit empheris file is read in...
    #     # hdr = orb_file[1].header #...header information is extacted...
    #     # data = orb_file[1].data #...and the relevant data us also extracted
    #     orb_time = hdul[1].data['TIME'] # time of observation when Jovian photons reach spaecraft 
    #     orb_x = hdul[1].data['X'] # x position of spacecraft
    #     orb_y = hdul[1].data['Y'] # y position of spacecraft
    #     orb_z = hdul[1].data['Z'] # z position of spacecraft
    # doy_sc = (orb_time - tstart) /86400.0 + event_DOYFRAC # doy of spacecraft
    
    # =============================================================================
    # Read in the ''chandra_horizons file'', extract relevant info and interpolate the data
    # =============================================================================
    
    start_time = Time(tstart, format='cxcsec')
    stop_time = Time(tstop, format='cxcsec')
    delta_time = '5m'
    eph_jup = gca_tools.fetch_ephemerides_fromCXO(start_time, stop_time, delta_time)
    
    # Extracts relevent date/time information needed from ephermeris file
    eph_dates = pd.to_datetime(eph_jup['datetime_str'])
    eph_dates = pd.DatetimeIndex(eph_dates)
    eph_doy = np.array(eph_dates.strftime('%j')).astype(int)
    eph_hours = eph_dates.hour
    eph_minutes = eph_dates.minute
    eph_seconds = eph_dates.second
    
    eph_doyfrac = gca_tools.doy_frac(eph_doy, eph_hours, eph_minutes, eph_seconds) # DOY fraction from ephermeris data
    
    # =============================================================================
    # Apply the correction to allow the photons to be tracked to their position on Jupiter
    # =============================================================================
    
    # Again, all below has been replaced with an improved Horizons ephemerides call
    # Get the orbital (x, y, z) at the times of ephemerides
    # interpfunc_x = interpolate.interp1d(doy_sc, orb_x, fill_value="extrapolate")
    # interpfunc_y = interpolate.interp1d(doy_sc, orb_y, fill_value="extrapolate")
    # interpfunc_z = interpolate.interp1d(doy_sc, orb_z, fill_value="extrapolate")
    
    # orb_x_interp = interpfunc_x(eph_doyfrac)
    # orb_y_interp = interpfunc_y(eph_doyfrac)
    # orb_z_interp = interpfunc_z(eph_doyfrac)
    
    # breakpoint()
    
    # Jupiter-Chandra distance, in meters
    r_jup = np.array(eph_jup['delta'].astype(float))*(u.au.to(u.m))  
    # DEC of Jupiter during observation, in rad
    dec_jup = np.deg2rad(np.array(eph_jup['DEC'].astype(float))) 
    # RA of Jupiter duting observation, in rad
    ra_jup = np.deg2rad(np.array(eph_jup['RA'].astype(float)))
    
    # (xp, yp, zp) are the components of CXO-Jupiter distance
    xp = (r_jup * np.cos(dec_jup) * np.cos(ra_jup))
    yp = (r_jup * np.cos(dec_jup) * np.sin(ra_jup))
    zp = (r_jup * np.sin(dec_jup))
    
    rp = np.sqrt(xp**2 + yp**2 + zp**2)
    rap_jup = (np.rad2deg(np.arctan2(yp,xp)) + 720.0) % 360  # RA of Jupiter at the observed coordinates
    decp_jup = (np.rad2deg(np.arcsin(zp/rp))) # DEC of Jupiter at the observed coordinates
    
    cc = np.cos(np.deg2rad(DEC_0)) # offset from Jupiter to allow photons to be tracked(?)
    
    interpfunc_ra_jup = interpolate.interp1d(eph_doyfrac, rap_jup)
    interpfunc_dec_jup = interpolate.interp1d(eph_doyfrac, decp_jup)
    ra_jup_interp = interpfunc_ra_jup(doy_chandra) # interpolated RA of Jupiter and the DOY of the emphermis file to the Chandra DOY
    dec_jup_interp = interpfunc_dec_jup(doy_chandra) # interpolated DEC of Jupiter and the DOY from the emphermis file to the Chandra
    #DOY
    
    if acis == 'y':
        scale = 0.4920 # units of pixels/arcsec
        xx = (event_x - (RA_0 - ra_jup_interp) * 3600.0 / scale * cc).astype(float) # corrected x position of photons
        yy = (event_y + (DEC_0 - dec_jup_interp) * 3600.0 / scale).astype(float) # corrected y position of photons
    
    else:
        scale = 0.13175 # untis of pixels/arcsec 
        xx = (event_x - (RA_0 - ra_jup_interp) * 3600.0 / scale * cc).astype(float) # corrected x position of photons
        yy = (event_y + (DEC_0 - dec_jup_interp) * 3600.0 / scale).astype(float) # corrected y position of photons
    
    
    # =========================================================================
    # The new positions of the photons replace the uncorrected positions from
    # the fits file and is written to a new fits file
    # =========================================================================
    if acis == 'y':
        new_event_filepath = (str(obs_dir) + f"/acisf{obs_id}_ssofreeze_evt2.fits") # path of the location
        # for the corrected fits file (with the photons corrected for the position).
    else:
        new_event_filepath = (str(obs_dir) + f"/hrcf{obs_id}_ssofreeze_evt2.fits") # path of the location
        # new_evt_location = f"/Users/daleweigt/Documents/DIAS_2022_2023/Chandra_code/{obs_id}test_v3.fits"
        # for the corrected fits file (with the photons corrected for the position)
    
    
    # No context manager here, as we need to edit hdul directly
    hdul = astropy.io.fits.open(event_filepath)
    
    hdul[1].data['X'] = xx
    hdul[1].data['Y'] = yy
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hdul[0].header['COMMENT'] = "Processed by sso_freeze() on {}".format(now)
    
    hdul.writeto(new_event_filepath, overwrite=True)
    hdul.close()
    
    return
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file name.')
    
    args = parser.parse_args()
    config = 'config.ini' if args.config is None else args.config
    
    _ = sso_freeze(config = config)
