# -*- coding: utf-8 -*-
"""
Takes the output photonlist of go_chandra as an input and filters based on the 
Pulse Invariant (PI). Only photons having correspnoding PI values in the range 
10-250 are selected. This aims to account for the gain degradation over time in
 the High Resolution Camera (HRC) on board the Chandra X-ray Observatory (CXO).


@history: 
    Translated into a function and generalized by MJR (2025)

@authors: 
    Seán McEntee, 
    Vinay Kashyap, 
    Dale Weigt, 
    Caitríona Jackman,
    Matthew J. Rutala
"""

#relevant packages 
import numpy as np
import pandas as pd
from astropy.time import Time
import configparser
import astropy
import os

import sso_freeze

def pi_filter(acis=None, obs_id=None, obs_dir=None, config=None):
    
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
    
    # Accounting for different filepaths of ObsIDs that originlly had SAMP values and others that did not.
    # Get the absolute path of the current script file
    current_file_path = os.path.abspath(__file__)
    # Get the directory containing the current script file
    current_directory = os.path.dirname(current_file_path)
    df = pd.read_csv(current_directory + '/ObsIDs_with_samp.txt', header=None, delimiter='\t')
    samp_ids = np.array(df.iloc[:,0])
    
    # if int(obsID) in samp_ids:
    #     folder_path = '/Users/mcentees/Desktop/Chandra/' + str(obsID) + '/primary'
    # else:
    #     folder_path = '/Users/mcentees/Desktop/Chandra/' + str(obsID) + '/repro'
    
    # Reading in ellipse data
    ellipse_data = pd.read_csv(obs_dir + f'/{obs_id}_photonlist_full_obs_ellipse.csv', comment='#')
    
    # reading in amplifier signals
    av1_jup = np.array(ellipse_data['av1'])
    av2_jup = np.array(ellipse_data['av2'])
    av3_jup = np.array(ellipse_data['av3'])
    
    au1_jup = np.array(ellipse_data['au1'])
    au2_jup = np.array(ellipse_data['au2'])
    au3_jup = np.array(ellipse_data['au3'])
    
    # reading in amplifier scale factor 
    amp_sf_jup = np.array(ellipse_data['amp_sf'])
    
    # calculating sumamp values
    sumamp_jup = av1_jup + av2_jup + av3_jup + au1_jup + au2_jup + au3_jup
    
    # calculating samp values
    samp_jup = (sumamp_jup * (2. ** (amp_sf_jup - 1.0)))/148.0
    
    # Read fits header for the date
    corrected_event_filepath = sso_freeze.find_event_filepath(acis, obs_id, obs_dir, suffix="ssofreeze_evt2.fits")
    with astropy.io.fits.open(corrected_event_filepath) as hdul:
        date_start = hdul[1].header['DATE-OBS']
        detector = hdul[1].header['DETNAM']
    
    # Decimal year of beginning of observation needed to perform PI calculation - can be obtained from reading in catalogue containing key info for all observations.
    # catalogue_path = config['PI Filter']['catalogue_path']
    # chandra_props = pd.read_excel('catalogue_all_data.xlsx')
    # index = np.where(chandra_props['ObsID'] == int(obs_id))[0][0]
    # date_start = chandra_props['Start Date'][index]
    date_dec = Time(date_start).decimalyear
    
    # PI calculation
    g= 1.0418475 + 0.020125799 * (date_dec - 2000.) + 0.010877227 * (date_dec - 2000.) ** 2. + - 0.0014310146 * (date_dec - 2000.) ** 3. + 5.8426766e-05 * (date_dec - 2000.) ** 4. # multiplicative scaling factor - Provided by Vinay Kashyap from the Chandra calibration team.
    PI_jup = g * samp_jup
    
    # Save the filtered results to file
    df_mod = ellipse_data.copy()
    
    df_mod.drop(['channel', 'samp', 'sumamp'], axis=1, inplace=True)
    df_mod['pi'] = df_mod['pi'].astype(np.float64)
    df_mod.loc[:, 'pi'] = PI_jup
    
    if detector == 'HRC-I':
        df_filtered = df_mod.query("10 < pi < 250")
    elif detector == 'HRC-S':
        df_filtered = df_mod.query("10 < pi < 300")
        
    new_filepath = obs_dir + f'/{obs_id}_photonlist_filtered_ellipse.csv'
    with open(new_filepath, 'w') as f:
        f.write('#UNITS:  t(s), x(arcsec), y(arcsec), pi, amp_sf, av1, av2, av3, au1, au2, au3, lat (deg), SIII_lon (deg), CML (deg), emiss (deg), Max PSF, MJD (days) \n')
        df_filtered.to_csv(f, header = True, index = False)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file name.')
    
    args = parser.parse_args()
    config = 'config.ini' if args.config is None else args.config
    
    _ = pi_filter(config=config)