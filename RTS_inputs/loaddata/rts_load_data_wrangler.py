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


df_p1load_1week = read_file('D:\\akash\\reeds\\ReEDS-2.0\\inputs\\loaddata\\RTS_load_hourly',
            index_columns=2)


# df_drdec_baseline = read_file('Clean2035clip1pct_load_hourly',
#             index_columns=2)



df_p1load_1week.rename(columns = {'z1':'1', 'z2':'2', 'z3':'3'}, inplace=True)



df_bus_mod = pd.read_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\RTS-GMLC-master\\RTS-GMLC-master\\RTS_Data\\SourceData\\bus_mod.csv', header=0)
df_bus_mod.rename(columns = {'MW Load':'rated_bus_max_load'}, inplace = True)
#df_bus_mod

df_groupbyreg_bus = df_bus_mod.groupby(['Area'])['rated_bus_max_load'].sum().reset_index(name='region_max_load')
#df_groupbyreg_bus

df_bus_load_proportion = pd.merge(df_bus_mod, df_groupbyreg_bus, on=['Area'], how='inner')
df_bus_load_proportion['regional_proportion'] = df_bus_load_proportion['rated_bus_max_load'] / df_bus_load_proportion['region_max_load']

df_p1load_1week = df_p1load_1week.transpose()
df_p1load_1week.index = df_p1load_1week.index.astype('int64')

df_actual_bus_load_proportion = pd.merge(df_bus_load_proportion, df_p1load_1week, left_on=['Area'], right_index=True, how='inner')
df_actual_bus_load_proportion

df_actual_bus_load_proportion_values = df_actual_bus_load_proportion.iloc[:,19:].multiply(df_actual_bus_load_proportion['regional_proportion'], axis="index")
df_actual_bus_load_proportion_values

df_bus_load_node_num = pd.merge(df_bus_mod, df_actual_bus_load_proportion_values, left_index=True, right_index=True)
df_bus_load_node_num.drop(['Bus Name','BaseKV','Bus Type','rated_bus_max_load','MVAR Load','V Mag','V Angle','MW Shunt G','MVAR Shunt B','Area','Sub Area','Zone','lat','lng','county_name','st'], axis=1, inplace=True)

df_bus_load_node_num.rename(columns = {'Bus ID': 'bus_id'}, inplace=True)
df_bus_load_node_num['bus_id'] = 'b' + df_bus_load_node_num['bus_id'].astype(str)
df_bus_load_node_num.rename(columns = {'bus_id': ''}, inplace=True)

df_bus_load_transpose = df_bus_load_node_num.transpose()
df_bus_load_transpose

df_bus_load_transpose.columns = df_bus_load_transpose.iloc[0]
df_bus_load_transpose = df_bus_load_transpose.iloc[1:]

df_bus_load_transpose = df_bus_load_transpose.reset_index()
df_bus_load_transpose

# #df_bus_load_transpose[['year','hour']] = df_bus_load_transpose['index'].str.split(', ', expand=True)
# for i,row in df_bus_load_transpose.iterrows():
#     df_bus_load_transpose.loc[i, 'year'] = row['index'][0]
#     df_bus_load_transpose.loc[i, 'hour'] = row['index'][1]

df_bus_load_transpose[['year', 'hour']] = pd.DataFrame(df_bus_load_transpose['index'].tolist(), index = df_bus_load_transpose.index)
df_bus_load_transpose

# df_bus_load_transpose['year'] = df_bus_load_transpose['year'].str.replace('(', '')
# df_bus_load_transpose['hour'] = df_bus_load_transpose['hour'].str.replace(')', '')

df_bus_load_transpose.drop(['index'], axis=1, inplace=True)

df_final = df_bus_load_transpose
df_final.rename(columns = {'year':'year_copy', 'hour':'hour_copy'}, inplace=True)
df_final.insert(0, 'year', True)
df_final.insert(1, 'hour', True)

df_final['year'] = df_final['year_copy']
df_final['hour'] = df_final['hour_copy']

df_final.drop(['year_copy', 'hour_copy'], axis=1, inplace=True)
df_final = df_final.set_index(['year', 'hour'])

df_final = df_final.astype('float64', copy=False)
df_final


df_final.to_hdf('C://Users//akarmaka//Documents//GitHub//ReEDS-2.0//inputs//loaddata//rts_nodal_load_hourly.h5', key='data', complevel=9, complib='zlib', index=True, encoding = 'utf-8')
