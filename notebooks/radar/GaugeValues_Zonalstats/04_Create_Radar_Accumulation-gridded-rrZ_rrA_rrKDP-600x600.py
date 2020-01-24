#!/usr/bin/env python
# coding: utf-8

# # Hourly accumulations
# 
# This notebook takes the scans within the hour, converts them to rain rate then to rainfall amount, then adds it up until the end of the hour. The hour totals are saved in a hdf file.

# Import libraries.

# In[61]:


import warnings
warnings.filterwarnings('ignore')


# In[62]:


import pyart
import wradlib as wrl
import pandas as pd
import tempfile
import os
import numpy as np

import pickle

import pytz
import datetime as dt

from copy import deepcopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature

import boto3
from botocore.handlers import disable_signing


# In[ ]:


# In[63]:


reader = shpreader.Reader(r'C:\Users\iac6311\Documents\Work\Data\GIS\USA\tl_2016_17_cousub\tl_2016_17_cousub.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())


# Read the gauge locations file and gauge observations file into pandas dataframes, and get the location coordinates.

# In[64]:


# load CCN gauge locations
CCN_gauge_locations_fname = 'C:/Users/iac6311/Documents/Work/Data/Cook_County/CookCounty_gage_locations.csv'
# load CCN gauge observations
CCN_gauge_observations_fname = 'C:/Users/iac6311/Documents/Work/Data/Cook_County/WaterYear2013.csv'

df_gauge_loc = pd.read_csv(CCN_gauge_locations_fname,header=0)
df_gauge = pd.read_csv(CCN_gauge_observations_fname,header=0)

x = df_gauge_loc['Longitude - West'].values
y = df_gauge_loc['Latitude'].values


# In[65]:


def rounder(t):
    """
    Rounds the time to the nearest hour.
    """
    if t.minute >= 30:
        return t.replace(second=0, microsecond=0, minute=0, hour=t.hour+1)
    else:
        return t.replace(second=0, microsecond=0, minute=0)


# In[66]:


path_rrfiles = r'C:\Users\iac6311\Documents\Work\SAVEUR\Processed_RainRetrievals\2013\04\17'


# In[67]:


rrfiles = os.listdir(path_rrfiles)


# In[68]:


rrfile = rrfiles[0]

radar = pyart.io.read_cfradial(os.path.join(path_rrfiles,rrfile))


# In[69]:


gatefilter = pyart.filters.GateFilter(radar)
# Develop your gatefilter first
# exclude masked gates from the gridding
#gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_masked('reflectivity')


# In[70]:


radar.fields.keys()


# In[71]:


# perform Cartesian mapping, limit to the reflectivity field.
grid = pyart.map.grid_from_radars(
    [radar], gatefilters=[gatefilter],
    grid_shape=(1, 600, 600),
    grid_limits=((2000, 2000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
    fields=['rainrate','rainrate_from_attenuation','rainrate_from_kdp'])


# In[72]:


x_rad, y_rad = grid.get_point_longitude_latitude()


# In[73]:


# 3.2 Get radar data
# Get slice
radar_slice0 = radar.get_slice(0)
rr_0 = radar.fields['rainrate']['data'][radar_slice0, :]


# In[74]:


sitecoords = (radar.longitude['data'][0],radar.latitude['data'][0],radar.altitude['data'][0])
az = radar.azimuth['data'][radar_slice0]
r = radar.range['data']
proj = wrl.georef.epsg_to_osr(4326)


# In[75]:


# save the radar parameters
with open('radarparams600x600.pkl','wb') as f:
    pickle.dump([sitecoords, az, r],f)


# In[76]:


radar_depth = wrl.trafo.r_to_depth(rr_0,interval=256)


# In[77]:


# create an empty dictionary
fname_dict = {}
for i in arange(1,25,1):
    fname_dict[i] = []
# fill in dictionary
for i in arange(len(rrfiles)):
    fname = rrfiles[i]
    dtime_utc = dt.datetime.strptime(fname,'radar_KLOT_%Y%m%d_%H%M%S.nc')
    fname_dict[dtime_utc.hour+1].append(dt.datetime.strftime(dtime_utc,'%Y/%m/%d/')+rrfiles[i])


# In[78]:


fname_dict


# In[79]:


for hour in list(fname_dict)[17:]:
    print(hour)
    fnames_within_hour = fname_dict[hour]
    hour_accum_rrZ = np.zeros((600,600))
    hour_accum_rrA = np.zeros((600,600))
    hour_accum_rrKDP = np.zeros((600,600))
    for fname in fnames_within_hour:
        print('.',end='')

        # get local time of radar
        fname = fname.rsplit('/',1)[-1]
        dtime_utc = dt.datetime.strptime(fname,'radar_KLOT_%Y%m%d_%H%M%S.nc')
        dtime_utc = pytz.utc.localize(dtime_utc)
        
        # read radar data
        radar = pyart.io.read_cfradial(os.path.join(path_rrfiles,fname))

        # grid 
        gatefilter = pyart.filters.GateFilter(radar)
        gatefilter.exclude_transition()
        gatefilter.exclude_masked('reflectivity')

        # perform Cartesian mapping, limit to the reflectivity field.
        grid = pyart.map.grid_from_radars(
            [radar], gatefilters=[gatefilter],
            grid_shape=(1, 600, 600),
            grid_limits=((2000, 2000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
            fields=['rainrate','rainrate_from_attenuation','rainrate_from_kdp'])

        # 3.2 Get radar data
        # Get slice
        rrZ_0 = grid.fields['rainrate']['data']
        rrA_0 = grid.fields['rainrate_from_attenuation']['data']
        rrKDP_0 = grid.fields['rainrate_from_kdp']['data']
        
        # convert rain rate to rain amount
        rrZ_amount = wrl.trafo.r_to_depth(rrZ_0,interval=345)
        rrA_amount = wrl.trafo.r_to_depth(rrA_0,interval=345)
        rrKDP_amount = wrl.trafo.r_to_depth(rrKDP_0,interval=345)
        
        hour_accum_rrZ += rrZ_amount[0]
        hour_accum_rrA += rrA_amount[0]
        hour_accum_rrKDP += rrKDP_amount[0]

    savefname = dt.datetime.strftime(dtime_utc.replace(microsecond=0,second=0,minute=0)+dt.timedelta(hours=1), '%Y%m%d_%H%M%S')
    wrl.io.to_hdf5('gridded_600x600_KLOT'+savefname+'_rrZ.hdf5', hour_accum_rrZ)
    wrl.io.to_hdf5('gridded_600x600_KLOT'+savefname+'_rrA.hdf5', hour_accum_rrA)
    wrl.io.to_hdf5('gridded_600x600_KLOT'+savefname+'_rrKDP.hdf5', hour_accum_rrKDP)

    print('')


# In[ ]:


x_rad, y_rad = grid.get_point_longitude_latitude()

# Saving the objects:
with open('radar_grid_600x600.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x_rad, y_rad], f)


# In[ ]:




