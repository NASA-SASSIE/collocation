### Authors: Severine Fournier and Kyla Drushka
### Collocate each SIZRS salinity and temperature measurement with the closest in time and space SMAP L3 SSS and OISST L4 observation
### Inputs:
### - path for SIZRS files
### - path for satellite data that will be downloaded
### - path for the collocation output file
### - year or year range
### - lat, lon of the region of interest



### Parameters to change ###
# Paths
from pathlib import Path
insitu_dir=Path('/home/jovyan/SASSIE/collocation/process/')  
satsss_dir=Path('/home/jovyan/Data/SASSIE/collocation/satellite/smap_jpl_l3/') 
satsst_dir=Path('/home/jovyan/Data/SASSIE/collocation/satellite/oisst_l4/') 
output_dir=Path('./data/output/') 

# Year in situ data
year_range=range(2016,2020) #OISST data are not available for now before January 2016

# Region of interest
lonrange = [-160, -130] # Beaufort Sea
latrange = [68, 80]
bounding_box = (lonrange[0], latrange[0], lonrange[1], latrange[1])


##########################################################################
### Imports ###
import numpy as np
import pandas as pd
import xarray as xr
from os.path import dirname, join
from scipy.interpolate import RegularGridInterpolator
from dateutil.parser import parse
from earthdata import DataCollections, DataGranules, Auth, Store

### Access data in the cloud
#Credentials to interact with CMR
auth = Auth().login(strategy='netrc')
if auth.authenticated is None:
    # we ask for credentials
    auth.login()
# auth

# Data collection short name
ShortName_sss = "SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5"
ShortName_sst = "AVHRR_OI-NCEI-L4-GLOB-v2.1"

# Get concept ID; See: https://github.com/nsidc/earthdata
CollectionQuery = DataCollections().short_name(ShortName_sss).cloud_hosted(True)
collections = CollectionQuery.get()
for collection in collections:
    concept_id_sss = collection.concept_id()
    # print(concept_id_sss)
    
CollectionQuery = DataCollections().short_name(ShortName_sst).cloud_hosted(True)
collections = CollectionQuery.get()
for collection in collections:
    concept_id_sst = collection.concept_id()
    # print(concept_id_sst)
    
for year in year_range:
    ### In situ data ###
    # load file
    filename=str(insitu_dir) + '/SIZRS_'+str(year)+'.nc'
    insitu = xr.open_dataset(filename)

    # round the time to the closest day
    days_from_insitu = pd.to_datetime(insitu.t).round('D')
    days = days_from_insitu.values

    # date ranges of insitu measurements
    # we use a Set to avoid repeating queries to the same day to download the data
    date_ranges = set()
    for day in days:
        start_date = str(day)
        end_date = str(day + np.timedelta64(1, 'D'))
        # or end_date = str(day + np.timedelta64(1, 'D') - np.timedelta64(1, 's')) for 23:59:59 of the same day, the search on CMR is the same.
        date_range = (start_date, end_date)
        date_ranges.add(date_range)
    # print(date_ranges)


    ### Satellite data ###
    # download files
    store = Store(auth) 
    for dt in date_ranges:
        GranuleQuery = DataGranules().parameters(
            concept_id=concept_id_sss,
            bounding_box=bounding_box,
            temporal=dt)
        granules = GranuleQuery.get()
        # Since our query at this point is for one day only
        dt_middate = parse(str(dt[0])).strftime('%Y%m%d')
        for i in range(0,len(granules)): #granule in granules:
            # The native_id has the mid_date encoded in this case
            if dt_middate in granules[i]["meta"]["native-id"]:
                files = store.get(granules[i:i+1], str(satsss_dir)+'/')
        GranuleQuery = DataGranules().parameters(
            concept_id=concept_id_sst,
            bounding_box=bounding_box,
            temporal=dt)
        granules = GranuleQuery.get()
        # Since our query at this point is for one day only
        dt_middate = parse(str(dt[0])).strftime('%Y%m%d')
        for i in range(0,len(granules)): #granule in granules:
            # The native_id has the mid_date encoded in this case
            if dt_middate in granules[i]["meta"]["native-id"]:
                files = store.get(granules[i:i+1], str(satsst_dir)+'/')

    # load files
    dates = pd.DatetimeIndex(days)
    dates = set(dates)
    list_files_sss=[]
    list_files_sst=[]
    for dt in dates:
        filename=str(satsss_dir)+'/SMAP_L3_SSS_'+str(dt.year)+str(dt.month).zfill(2)+str(dt.day).zfill(2)+'_8DAYS_V5.0.nc'
        # smap.append(xr.open_dataset(filename))
        list_files_sss.append(filename)
        # print(filename)
        filename=str(satsst_dir)+'/'+str(dt.year)+str(dt.month).zfill(2)+str(dt.day).zfill(2)+'120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc'
        list_files_sst.append(filename)
        # print(filename)

    ds_smap_L3 = xr.open_mfdataset(
        list_files_sss,
        combine='nested',
        concat_dim='time',
        decode_cf=True,
        coords='minimal',
        chunks={'time': 1}
        ).sel(longitude=slice(lonrange[0],lonrange[1]), latitude=slice(latrange[1],latrange[0]))

    ds_oisst_L4 = xr.open_mfdataset(
        list_files_sst,
        combine='nested',
        concat_dim='time',
        decode_cf=True,
        coords='minimal',
        chunks={'time': 1}
        ).sel(lon=slice(lonrange[0],lonrange[1]), lat=slice(latrange[0],latrange[1])) #lat are organized in the other direction than in smap

    # SSS: create a new dataset containing the x/y/t values we want to interpolate to, along a new dimension we call "points":
    interp_to= xr.Dataset(
        dict(longitude = xr.DataArray(insitu.x.values, dims='points'),
            latitude = xr.DataArray(insitu.y.values, dims='points'),
            time = xr.DataArray(insitu.t.values, dims='points')))
    # now, we can imterpolate to that to get the interpolated data along the points dimension:
    collocation_sss = ds_smap_L3['smap_sss'].interp(interp_to)

    # Same for SST
    interp_to= xr.Dataset(
        dict(lon = xr.DataArray(insitu.x.values, dims='points'),
            lat = xr.DataArray(insitu.y.values, dims='points'),
            time = xr.DataArray(insitu.t.values, dims='points')))
    collocation_sst = ds_oisst_L4['analysed_sst'].interp(interp_to)-273.15


    ### Save output file ###
    insitu['sss_smap']=collocation_sss
    insitu['oisst']=collocation_sst

    # change variable names, clean up
    ds=insitu.rename({'x': 'lon_insitu','y': 'lat_insitu','z': 'depth_insitu','t': 't_insitu',
                      'SSS': 'sss_insitu','SST': 'sst_insitu',
                      'time': 't_smap'})
    ds.sss_insitu.attrs['long_name'] = 'SIZRS in situ salinity'
    ds.sss_insitu.attrs['units'] = '1e-3'
    ds.sst_insitu.attrs['long_name'] = 'SIZRS in situ temperature'
    ds.sst_insitu.attrs['units'] = 'degC'

    # save to Disk
    fileout=str(output_dir) + '/SIZRS_SMAP_collocation_'+str(year)+'.nc'
    ds.to_netcdf(fileout)
    print('collocation output file: '+fileout)