# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:19:37 2023

This is a general framework of a script that adjusts ReEDS input data from 
balancing area (BA) or resource region area (sreg) to county (FIPS) spatial 
aggregation. 
County-level data is calculated by multiplying each BA's data (respective to 
desired county) by the population/geographic-area fraction of the desired 
county relative to the respective BA

Workflow:
- Import ioi data, frac data, and ba2fips map
- Format frac df 
    - Population
    - Geographic size
    - Transmission Line Size
- Format ioi data to wide format df (cols = BA)
- Merge ioi df to frac df, multiply ioi data by frac data, merge FIPS codes to each county row 
- Transpose ioi data back to original format, final data cleanup steps.
- Data output

@author: jcarag
"""
# %%
# Import packages
import os
import gzip
import h5py
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

# User Inputs:
select_input = 'water_with_cons_rate.csv'#'EPMEDIUM_load_hourly.h5'
select_frac  = 'geosize' #OPTIONS: ['population', 'geosize', 'translinesize']. Not necessary for transmission_distance_cost_*.csv
## Long format inputs only
select_header= True

# Directories
reeds_dir = os.path.join('D:\\','akash','reeds','ReEDS-2.0')
inputs_dir= os.path.join(reeds_dir,'inputs')
output_dir= os.path.join(os.path.dirname(os.getcwd()),'processed_inputs')

print(os.getcwd())

#============
#%% FUNCTIONS
#============
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

#===============
#%% DICTIONARIES
#===============

# Input Paths Dictionary
'''
INPUTS:
key:    input file name to be formatted with new county FIPS data
value:  tuples consisting of:
        [0] function for reading in the input file
        [1] region type to downselect ('r' or 's')
        [2] name of region column
        [3] name of value column
        [4] format of the input data ('wide' or 'long')
'''

INPUTS = {
    # 'load.csv.gz'               :   (read_file(os.path.join(inputs_dir,'variability','multi_year','load')),'r','wide'),
    # 'Baseline_load_hourly.csv.gz'           : (read_file(os.path.join(inputs_dir,'loaddata',
    #                                                                     'Baseline_load_hourly')),'r','','',
    #                                              'wide'),
    # 'Clean2035_load_hourly.csv.gz'          : (read_file(os.path.join(inputs_dir,'loaddata',
    #                                                                     'Clean2035_load_hourly')),'r','','','wide'),
    # 'Clean2035_LTS_load_hourly.csv.gz'      : (read_file(os.path.join(inputs_dir,'loaddata',
    #                                                                     'Clean2035_LTS_load_hourly')),'r','','','wide'),
    # 'Clean2035clip1pct_load_hourly.h5'      : (read_file(os.path.join(inputs_dir,'loaddata',
    #                                                                     'Clean2035clip1pct_load_hourly')),'r','','','wide'),
    # 'EERbaseClip40_load_hourly.csv.gz'      : (read_file(os.path.join(inputs_dir,'loaddata',
    #                                                                     'EERbaseClip40_load_hourly')),'r','','','wide'),
    # 'EERbaseClip80_load_hourly.csv.gz'      : (read_file(os.path.join(inputs_dir,'variability','EFS_Load',
    #                                                                     'EERbaseClip80_load_hourly')),'r','','','wide'),
    # 'EERhighClip40_load_hourly.csv.gz'      : (read_file(os.path.join(inputs_dir,'variability','EFS_Load',
    #                                                                     'EERhighClip40_load_hourly')),'r','','','wide'),
    # 'EERhighClip80_load_hourly.csv.gz'      : (read_file(os.path.join(inputs_dir,'variability','EFS_Load',
    #                                                                     'EERhighClip80_load_hourly')),'r','','','wide'),
    # 'EPHIGH_load_hourly.csv.gz'             : (read_file(os.path.join(inputs_dir,'variability','EFS_Load',
    #                                                                     'EPHIGH_load_hourly')),'r','','','wide'),    
    # 'EPMEDIUM_load_hourly.h5'               : (read_file(os.path.join(inputs_dir,'loaddata','EPMEDIUM_load_hourly')),'r','','','wide'),
    # 'EPMEDIUMStretch2040_load_hourly.csv.gz': (read_file(os.path.join(inputs_dir,'loaddata',
    #                                                                     'EPMEDIUMStretch2040_load_hourly')),'r','','','wide'),
    # 'EPMEDIUMStretch2046_load_hourly.csv.gz': (read_file(os.path.join(inputs_dir,'variability','EFS_Load',
    #                                                                     'EPMEDIUMStretch2046_load_hourly')),'r','','','wide'),
    ## 'EPREFERENCE_load_hourly.csv.gz'        : (read_file(os.path.join(inputs_dir,'variability','EFS_Load',
    ##                                                                     'EPREFERENCE_load_hourly')),'r','','','wide'),

    # 'dr_rsc_Baseline.csv'       :   (pd.read_csv(os.path.join(inputs_dir,'demand_response','dr_rsc_Baseline.csv'),
    #                                     names=['BA','tech','bin','year','var','value'], header=0),'r','r','value','long'),
    # 'dr_rsc_Baseline_shed.csv'  :   (pd.read_csv(os.path.join(inputs_dir,'demand_response','dr_rsc_Baseline_shed.csv'),
    #                                     names=['BA','tech','bin','year','var','value'], header=0),'r','r','value','long'),
    # 'dr_rsc_Baseline_shift.csv' :   (pd.read_csv(os.path.join(inputs_dir,'demand_response','dr_rsc_Baseline_shift.csv'),
    #                                     names=['BA','tech','bin','year','var','value'], header=0),'r','r','value','long'),
    # 'dr_rsc_none.csv'           :   (pd.read_csv(os.path.join(inputs_dir,'demand_response','dr_rsc_none.csv'),
    #                                     names=['BA','tech','bin','year','var','value'], header=0),'r','r','value','long'),
    # 'geo_rsc_BAU.csv'           :   (pd.read_csv(os.path.join(inputs_dir,'geothermal','geo_rsc_BAU.csv'),
    #                                     names=['*i','BA','sc_cat','value'], header=0),'r','r','value','long'),
    # 'geo_rsc_TI.csv'            :   (pd.read_csv(os.path.join(inputs_dir,'geothermal','geo_rsc_TI.csv'),
    #                                     names=['*i','BA','sc_cat','value'], header=0),'r','r','value','long'),                                
    # 'geo_rsc_ATB_2020.csv'      :   (pd.read_csv(os.path.join(inputs_dir,'geothermal','geo_rsc_ATB_2020.csv'),
    #                                     names=['*i','BA','sc_cat','value'], header=0),'r','r','value','long'),
    # 'geo_rsc_ATB_2022.csv'      :   (pd.read_csv(os.path.join(inputs_dir,'geothermal','geo_rsc_ATB_2022.csv'),
    #                                     names=['*i','BA','sc_cat','value'], header=0),'r','r','value','long'),
    # 'h2_ba_share.csv'           :   (pd.read_csv(os.path.join(inputs_dir,'consume','h2_ba_share.csv'),
    #                                     names=['BA','t','fraction'], header=0),'r','*r','fraction','long'),      
    # 'wat_access_cap_cost.csv'   :   (pd.read_csv(os.path.join(inputs_dir,'waterclimate','wat_access_cap_cost.csv'),
    #                                     names=['*wst','sc_cat','BA','value'], header=0),'r','r','value','long'),
    'water_with_cons_rate.csv'  :   (pd.read_csv(os.path.join(inputs_dir,'waterclimate','water_with_cons_rate.csv')),'r','','','wide'),
    # 'ev_static_demand.csv'      :   (pd.read_csv(os.path.join(inputs_dir,'loaddata','ev_static_demand.csv'),
    #                                     names=['BA','h','t','MW'], header=0),'r','*r','MW','long'),
    # 'ev_dynamic_demand.csv'     :   (pd.read_csv(os.path.join(inputs_dir,'loaddata','ev_dynamic_demand.csv'),
    #                                     names=['BA','szn','t','MWh'], header=0),'r','*r','MWh','long'),
    # 'can_imports.csv'           :   (pd.read_csv(os.path.join(inputs_dir,'canada_imports','can_imports.csv')),'r','','','wide'),
    # 'can_exports.csv'           :   (pd.read_csv(os.path.join(inputs_dir,'canada_imports','can_exports.csv')),'r','','','wide'),
    # 'hydcap.csv'                :   (pd.read_csv(os.path.join(inputs_dir,'supplycurvedata','hydcap.csv')),'r','','','wide'),  
    # 'net_trade_can.csv'         :   (pd.read_csv(os.path.join(inputs_dir,'canada_imports','net_trade_can.csv'),
    #                                     names = ['BA','h','t','MWh'], header=0),'r','*r','MWh','long'),
    # 'can_trade_8760.h5'         :   (read_file(os.path.join(inputs_dir,'canada_imports','can_trade_8760')),'r','r','net','long'),
}

# Population, Geographic Size, and Transmission Line Size Data Dictionary
'''
INPUTS:
key:    the type of data used to downselect
value:  tuple consisting of:
        [0] function for reading in the downselection data
        [1] the downselection data column
'''
FRAC_DATA = {
    'population'    :   (pd.read_csv('co-est2021-alldata.csv', header=0, usecols=['STNAME','CTYNAME','POPESTIMATE2021','STATE','COUNTY'],nrows=3195,encoding='latin-1'),'POPESTIMATE2021'),
    'geosize'       :   (pd.read_csv('geoareamap.csv', header=0, usecols=['STNAME','CTYNAME','ALAND','STATE','COUNTY'],encoding='latin-1'),'ALAND'),
    'translinesize' :   (pd.read_csv('transmission_fractions.csv', header=0, usecols=['US PCA','nonUS PCA','FIPS','total BA MW','total FIPS MW','transfrac'],encoding='latin-1'),'transfrac')
}

#============
#%% PROCEDURE
#============
#%% STEP 1: import data using instructions from the INPUTS dictionary

# Import data to be downselected from BA to FIPS (Input of Interest: 'ioi')
ioi = INPUTS[select_input][0]
# Import file containing the data that will be used to create the fraction for BA-to-FIPS downselection (population or geographic size data file)
frac = FRAC_DATA[select_frac][0]
# Import the BA-to-FIPS map file
BAorSREG = 'PCA_REG' if INPUTS[select_input][1] == 'r' else 'DEMLAB'
bafipsmap = pd.read_excel('county_reeds_corrected0310.xlsx',header=0,usecols=['NAME','STATE_NAME','FIPS',BAorSREG])

# Set region and value columns
select_BAcol = INPUTS[select_input][2]
select_valcol= INPUTS[select_input][3]

#%% STEP 2: frac_df formatting

df_frac = frac.copy()

if select_frac == 'translinesize':
    df_frac.rename(columns={'US PCA':'PCA_REG','transfrac':'proportion_ofdata'},inplace=True)
    df_fracfinal = df_frac.copy()
else:
    datacol = FRAC_DATA[select_frac][1]
    # Zero padding data in State and County codes present in Population estimates file to match Reeds County file format
    df_frac['STATE']=frac['STATE'].apply(lambda x: '{0:0>2}'.format(x))
    df_frac['COUNTY']=df_frac['COUNTY'].apply(lambda x: '{0:0>3}'.format(x))
    # Creating new column for entire FIPS code needed to join this dataframe to Population dataframe
    df_frac['FIPS']=df_frac['STATE'] + df_frac['COUNTY']
    # Ensuring datatype consistency for created column.
    df_frac['FIPS']= df_frac['FIPS'].apply(pd.to_numeric)

    # Inner join between Downselection dataframe and ReEDS Counties dataframe
    df_fracwithmapping = pd.merge(df_frac, bafipsmap, on='FIPS', how='inner')
    
    # To get aggregate of BA region population/area, group by BA region number (PCA_REG column) and get sum of populations/areas for
    # each set of counties inside respective BA region
    df_groupedbyPCA = df_fracwithmapping.groupby([BAorSREG])[datacol].sum().reset_index(name='County_sum_data')

    # Full Outer join between above dataframe which contains aggregates of populations/areas with dataframe containing county names, FIPS
    # and county-wise population/area (going to be required to get the proportional factor in next steps)
    df_fracfinal = pd.merge(df_groupedbyPCA, df_fracwithmapping, on=BAorSREG, how='outer')
    # Dividing each county's population/area by the cumulative sum of it's respective BA Region
    df_fracfinal['proportion_ofdata']=df_fracfinal[datacol]/df_fracfinal['County_sum_data']

#%% STEP 3: ioi df formatting

if INPUTS[select_input][4] == 'wide':
    print('wide formatting')
    select_header = True
    # Load data formatting
    # Cleaning Load data - Renaming Columns from format "p(BARegion)" to just BARegion number
    # Transposing the entire dataframe so now we have all BA region numbers as 1 column and subsequent columns are hourly timesteps.
    # Now the load df is 135 rows (1 for each BA region (1st row is just headers)) by 61320 columns (7 years worth of hourly data)
    df_ioi = ioi.copy()
    if select_input in ['can_imports.csv','can_exports.csv']:
        df_ioi = df_ioi.set_index('r').T
        indices = ['r']
        for prefix in ['p','s']:
            df_ioi.columns = df_ioi.columns.str.replace(prefix,'')
        df_ioi_transposed = df_ioi.copy()
        df_ioi_transposed = df_ioi_transposed.T
        df_ioi_transposed.index = df_ioi_transposed.index.astype(int)
    else:
        for prefix in ['p','s']:
            df_ioi.columns = df_ioi.columns.str.replace(prefix,'')
        df_ioi.rename(columns = {"": "PCA_REG"}, inplace=True)
        colnames = list(df_ioi.columns)
        indices = [i for i in colnames if i not in np.arange(1,206).astype(str)]
        df_ioi = df_ioi.set_index(indices)
        df_ioi_transposed = df_ioi.T
        # Resetting column headers to make sure correct 1st row is treated as header
        # df_ioi_transposed.columns = df_ioi_transposed.iloc[0]
        # df_ioi_transposed = df_ioi_transposed.iloc[1:]
        # Changing index datatype to int
        df_ioi_transposed.index = df_ioi_transposed.index.astype(int)

    # Outer join between dataframe containing proportional factors for each county and df containing transposed load data
    print('   Starting join of df_frac with df_ioi...')
    df_fracbycounty_ioi = pd.merge(df_fracfinal, df_ioi_transposed, left_on=BAorSREG, how='inner', right_index=True)
    # Dropping columns from df that are not required
    dropcols = ['COUNTY','STATE','STNAME','NAME','nonUS PCA','total BA MW','total FIPS MW']
    for col in dropcols:
        if col in df_fracbycounty_ioi.columns:
            df_fracbycounty_ioi.drop(col, axis=1, inplace=True)
    # Multiplying the proportional factors column with each load data column. Earlier join ensures
    # we have counties by BA (PCA_REG column)
    if select_input in ['can_imports.csv','can_exports.csv']:
        df_newioi = df_fracbycounty_ioi.iloc[:, 3:].multiply(df_fracbycounty_ioi['proportion_ofdata'], axis="index")
    else:
        df_newioi = df_fracbycounty_ioi.iloc[:, 7:].multiply(df_fracbycounty_ioi['proportion_ofdata'], axis="index")

    # Assigning the FIPS number for each county. This merge ensures we pick out only the FIPS column
    # and all newly calculated load columns
    df_newioi_fips = df_fracbycounty_ioi[['FIPS']]
    print('   Starting join of FIPS codes to df_ioi...')
    df_newioi_withfips = df_newioi_fips.merge(df_newioi, how='right', left_index=True, right_index=True)

    # Transpose data back to wide format (columns = FIPS), flatten multiindex if required
    print('   Transposing df_ioi back to original wide format...')
    df_widefinal = df_newioi_withfips.copy()
    # Converting FIPS codes to integers (to remove decimal '0'), adding 'p' prefix to FIPS codes
    df_widefinal.loc[~df_widefinal['FIPS'].isna(),'FIPS'] = df_widefinal.loc[~df_widefinal['FIPS'].isna(),'FIPS'].astype(int).astype(str)
    df_widefinal.loc[~df_widefinal['FIPS'].isna(),'FIPS'] = (df_widefinal.loc[~df_widefinal['FIPS'].isna(),'FIPS']
                                                            .apply(lambda x: '{}{}'.format('p0',x) if len(x)<5 
                                                                else '{}{}'.format('p',x)))
    df_widefinal.set_index('FIPS', drop=True, inplace=True)
    df_widefinal = df_widefinal.T
    
    # Final data formatting: set values to float64, round to 2 decimals
    print('   Final data formatting...')
    print('      - converting values to float64...')
    df_widefinal = df_widefinal.astype('float64', copy=False)
    print('      - rounding values to 2 decimals...')
    df_widefinal = df_widefinal.round(decimals=2)
    if len(indices) > 1:
        print('      - flatten index back to non-data columns...')
        df_widefinal.reset_index(inplace=True)
        df_widefinal[indices] = pd.DataFrame(df_widefinal['index'].tolist(), index=df_widefinal.index)
        for idx in reversed(indices):
            df_widefinal.insert(0,idx,df_widefinal.pop(idx))
        df_widefinal.drop(columns=['index'],inplace=True)
    df_newioi = df_widefinal.copy()
    # Final formatting for specific files
    if select_input in ['can_imports.csv','can_exports.csv']:
        df_newioi = df_newioi.T.reset_index().rename(columns={'FIPS':'r'})
    if select_input in ['hydcap.csv']: # Not sure why this is happening...
        df_newioi.rename(columns={'cla':'class'},inplace=True)
    print('Done!\n')
else:
    print('')


if INPUTS[select_input][4] == 'long':
    print('long formatting')
    df_ioi = ioi.copy()
    if select_input == 'can_trade_8760.h5':
        df_ioi.columns = ['BA','h','t','net']
    df_ioi_colorder = df_ioi.columns.tolist()
    df_newioi_colorder = ['FIPS' if item == 'BA' else item for item in df_ioi_colorder]
    for prefix in ['p','s']:
        df_ioi['BA'] = df_ioi.BA.str.replace(prefix,'')
    df_ioi.set_index('BA',drop=True, inplace=True)
    df_ioi.index = df_ioi.index.astype(int)

    # Outer join between dataframe containing proportional factors for each county and df containing transposed load data
    print('   Starting join of df_frac with df_ioi...')
    df_fracbycounty_ioi = pd.merge(df_fracfinal, df_ioi, left_on=BAorSREG, how='right', right_index=True)
    # Dropping columns from df that are not required
    if select_frac == 'translinesize':
        df_fracbycounty_ioi.drop(['nonUS PCA','total BA MW','total FIPS MW'], axis=1, inplace=True)
    else:
        df_fracbycounty_ioi.drop(['COUNTY', 'STATE','STNAME','NAME'], axis=1, inplace=True)
    # df_popbycounty_load.dropna(subset=df_ioi.columns,inplace=True)

    # Multiplying the proportional factors column with value column. Earlier join ensures
    # we have counties by BA Region number (PCA_REG column)
    df_fracbycounty_ioi['prop_value'] = df_fracbycounty_ioi['proportion_ofdata'].multiply(df_fracbycounty_ioi[select_valcol], axis="index")

    # convert df_ioi back to original format, sort all columns besides value columns in ascending order:
    print('   Converting df_ioi back to original long format...')
    if select_input == 'can_trade_8760.h5':
        df_newioi = df_fracbycounty_ioi[['FIPS']+ioi.columns.drop('r').tolist()]
    else:
        df_newioi = df_fracbycounty_ioi[['FIPS']+ioi.columns.drop('BA').tolist()]
    df_newioi = df_newioi[df_newioi_colorder]
    df_newioi.sort_values(by=df_newioi.columns[:-1].tolist(), inplace=True)
    # df_newioi.sort_values(by=['t','FIPS'], inplace=True)
    final_colorder = []
    for item in df_newioi_colorder:
        if item == 'FIPS':
            colname = select_BAcol
        elif item == 'value':
            colname = select_valcol
        else:
            colname = item
        final_colorder.append(colname)
    df_newioi.columns = final_colorder
    
    # Final data formatting: set values to float64, round to 2 decimals
    print('   Final data formatting...')
    print('      - converting values to float64...')
    df_newioi[select_valcol] = df_newioi[select_valcol].astype('float64', copy=False)
    print('      - rounding values to 2 decimals...')
    df_newioi[select_valcol] = df_newioi[select_valcol].round(decimals=2)
    print('      - adding \'p\' prefix to FIPS codes, renaming \'FIPS\' column back to \'r\'...')
    df_newioi.loc[~df_newioi[select_BAcol].isna(),select_BAcol] = df_newioi.loc[~df_newioi[select_BAcol].isna(),select_BAcol].astype(int).astype(str)
    df_newioi.loc[~df_newioi[select_BAcol].isna(),select_BAcol] = (df_newioi.loc[~df_newioi[select_BAcol].isna(),select_BAcol]
                                                            .apply(lambda x: '{}{}'.format('p0',x) if len(x)<5 
                                                                else '{}{}'.format('p',x)))
    df_newioi.rename(columns={'FIPS':'r'},inplace=True)
    print('Done!')
    # # Code to use if you need to order the dataframe by a specific column data order
    # df_newioi.szn = pd.Categorical(df_newioi.szn,categories=['summ','fall','wint','spri'])
    # df_newioi = df_newioi.sort_values(by=['*r','szn','t'])
else:
    print('')

#%% STEP 4: Save updated ioi to processed_inputs folder

print('Export final df')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if select_input[-3:] == '.h5':
    df_newioi.to_hdf(os.path.join(output_dir,select_input), key='data', complevel=4, format='table')
else:
    df_newioi.to_csv(os.path.join(output_dir,select_input), index=False, header=select_header)

# Check filesize. 
# If >10MB, then convert to .csv.gz format
# If previously in csv.gz format, then maintain this.
# if select_input[-6:] == 'csv.gz':
#     df_newioi.to_csv(os.path.join(outputdir,select_input))
filesize = os.path.getsize(os.path.join(output_dir,select_input))
filesizeMB = filesize / (1024)
print('filename: {}\nfilesize: {} KB'.format(select_input,round(filesizeMB)))

# %%