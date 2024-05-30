#!/usr/bin/env python
# coding: utf-8

# In[14]:


import csv
import gzip
import pandas as pd
import h5py

# df_p1load_1week = pd.read_csv('load.csv.gz', compression='gzip', header=0, nrows=168)
# print(df_p1load_1week["p1"])

#df_p1_census = pd.read_csv('co-est2021-alldata.csv', header=0, usecols=['STNAME','CTYNAME','POPESTIMATE2021','STATE','COUNTY'], nrows=3195, encoding='latin-1')
#print(df_p1_census)

#df_p1_counties = pd.read_excel('county_reeds_corrected0310.xlsx', header=0, usecols=['NAME','STATE_NAME','FIPS','PCA_REG'])
#print(df_p1_counties)

# filename = "Reeds_Spatialflex\Clean2035clip1pct_load_hourly.h5"
# df_drdec_baseline = h5py.File(filename, 'r')
# df_drdec_baseline.keys()
# #df_drdec_baseline.close()

def read_file(filename, index_columns=1):
    """
    Read input file of various types (for backwards-compatibility)
    """
    # Try reading a .h5 file written by pandas
    try:
        df = pd.read_hdf(filename+'.h5')
    # Try reading a .h5 file written by h5py
    except (ValueError, TypeError, FileNotFoundError, OSError):
        try:
            with h5py.File(filename+'.h5', 'r') as f:
                keys = list(f)
                datakey = 'data' if 'data' in keys else ('cf' if 'cf' in keys else 'load')
                ### If none of these keys work, we're dealing with EER-formatted load
                if datakey not in keys:
                    years = [int(y) for y in keys if y != 'columns']
                    df = pd.concat(
                        {y: pd.DataFrame(f[str(y)][...]) for y in years},
                        axis=0)
                    df.index = df.index.rename(['year','hour'])
                else:
                    df = pd.DataFrame(f[datakey][:])
                    df.index = pd.Series(f['index']).values
                df.columns = pd.Series(f['columns']).map(
                    lambda x: x if type(x) is str else x.decode('utf-8')).values
        # Fall back to .csv.gz
        except (FileNotFoundError, OSError):
            df = pd.read_csv(
                filename+'.csv.gz', index_col=list(range(index_columns)),
                float_precision='round_trip',
            )

    return df


df_p1load_1week = read_file('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\loaddata\\RTS_load_hourly',
            index_columns=2)


# df_drdec_baseline = read_file('Clean2035clip1pct_load_hourly',
#             index_columns=2)



#
# df_drdec_baseline = pd.read_hdf('Reeds_Spatialflex\Clean2035clip1pct_load_hourly.h5', header=0)
df_p1load_1week


# In[15]:


df_p1load_1week_2 = df_p1load_1week.transpose()
df_p1load_1week_2


# In[19]:


df_p1load_1week_2.index


# In[ ]:




