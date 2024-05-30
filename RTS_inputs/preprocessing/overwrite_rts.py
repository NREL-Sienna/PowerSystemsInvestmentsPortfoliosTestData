#%% IMPORTS ###
import os, sys, site, math, importlib
import gdxpds
import pandas as pd
import numpy as np
import h5py
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.options.display.max_rows = 20
pd.options.display.max_columns = 200

### Shared paths
reedspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(os.path.join(reedspath,'input_processing'))
import LDC_prep
site.addsitedir(os.path.join(reedspath,'postprocessing'))
import plots
plots.plotparams()
rtspath = os.path.expanduser('~/github/RTS-GMLC')

#%%### Inputs
## DA is hourly, RT is 5-minute
timetype = 'DAY_AHEAD'

### Make up growth profile
growth_yoy = 0.01
baseyear = 2020
years = np.arange(2010,2051,1)
growthfrac = pd.Series((1 + growth_yoy)**(years - baseyear), index=years)

#%% ReEDS files for comparison
eia = pd.read_csv(os.path.join(
    reedspath,'inputs','capacitydata','ReEDS_generator_database_final_EIA-NEMS.csv'))

#%%### Lookup tables
## Not sure about oil plants
rts2reeds = {
    ('tech','Unit Type'): {
        ('CSP','CSP'): 'csp-ns',
        ('Coal','STEAM'): 'coalolduns',
        ('Gas CC','CC'): 'gas-cc',
        ('Gas CT','CT'): 'gas-ct',
        ('Hydro','HYDRO'): 'hydED',
        ('Hydro','ROR'): 'hydEND',
        ('Solar RTPV','RTPV'): 'distpv',
        ('Solar PV','PV'): 'upv',
        ('Nuclear','NUCLEAR'): 'nuclear',
        ('Oil CT','CT'): 'o-g-s',
        ('Oil ST','STEAM'): 'o-g-s',
        ('Storage','STORAGE'): 'battery_4',
        ('Sync_Cond','SYNC_COND'): None,
        ('Wind','WIND'): 'wind-ons',
    },
    'HeatRate': {0:np.nan}
}
## Zones
zones = ['z1', 'z2', 'z3']
## Just pick an s region for each p region from rsmap.csv
p2s = {'p1':2, 'p2':5, 'p3': 8, 'z1':328, 'z2': 333, 'z3':342}
## Define the mapping between RTS-GMLC areas and ReEDS zones
area2zone = {1:'z1', 2:'z2', 3:'z3'}
area2zone = {**area2zone, **{f'z{k}':v for k,v in area2zone.items()}}
## Define the available capacity [MW] at each existing PV/wind site
cap_avail = 4000
## Define the spur-line cost [$/MW]
cost_spur = 10e3
cost_reinforcement = 100e3
dist_km = 10
reinforcement_dist_km = 100
## Random seed
seed = 42
sigma = 0.1
## Additional variables for mini-hourlize
start_year = 2010

#%% Take a look at some RTS files
dfbus = pd.read_csv(
    os.path.join(rtspath, 'RTS_Data','SourceData','bus.csv')
)
bus2zone = dfbus.set_index(['Bus ID'])['Area'].map(area2zone)

###############
#%%### unitdata
dfgen = pd.read_csv(
    os.path.join(rtspath, 'RTS_Data','SourceData','gen.csv')
)
### Rename and reformat columns
dfcap = dfgen.copy()
dfcap['reeds_ba'] = dfcap['Bus ID'].map(bus2zone)
rename_columns = {
    'VOM': 'T_VOM',
    'Category': 'tech',
    'PMax MW': 'cap',
    'HR_avg_0': 'HeatRate',
}
dfcap = dfcap.rename(columns=rename_columns).replace(rts2reeds)
dfcap['tech'] = dfcap.apply(
    lambda row: rts2reeds['tech','Unit Type'][row.tech, row['Unit Type']], axis=1)
dfcap = dfcap.dropna(subset=['tech'])
## Zero out techs that shouldn't have a heatrate
dfcap.loc[dfcap.tech.isin(['hydED','hydEND','csp-ns']),'HeatRate'] = np.nan
### Add a bunch of columns with zeros
dfcap[[
    'TCOMB_V','TSNCR_V','TSCR_V','T_FFV','T_DSIV',
    'T_FOM','T_CAPAD','TCOMB_F','TSNCR_F','TSCR_F','T_FFF','T_DSIF',
    'T_CCSV','T_CCSF','T_CCSROV','T_CCSHR','T_CCSCAPA','T_CCSLOC',
]] = 0
### Add some other columns with values
toadd = {
    'ctt': 'n',
    'Nuke80RetireYear': 2050,
    'StartYear': 2009,
    'IsExistUnit': True,
    'wst': 'fsa',
}
for col, val in toadd.items():
    dfcap[col] = val

dfcap['resource_region'] = dfcap.reeds_ba.map(p2s)
## Assume winter cap is same as summer cap
dfcap['TC_WIN'] = dfcap.cap.copy()
dfcap['coolingwatertech'] = dfcap.tech + '_' + dfcap.ctt + '_' + dfcap.wst
## Specify a few units to be added after 2009 to prevent empty dataframes
index_late_add = dfcap.tech.drop_duplicates().index
dfcap.loc[index_late_add, 'StartYear'] = 2012
dfcap['RetireYear'] = dfcap['StartYear'] + 30

### Use the row index as sc_point_gid (weird)
dfcap['sc_point_gid'] = dfcap.index
site2gen = dfcap.set_index('sc_point_gid')['GEN UID']

windsites = dfcap.loc[dfcap['Unit Type']=='WIND'].set_index('sc_point_gid')
pvsites = dfcap.loc[dfcap['Unit Type']=='PV'].set_index('sc_point_gid')
rtpvsites = dfcap.loc[dfcap['Unit Type']=='RTPV'].set_index('sc_point_gid')

### Drop distributed PV
dfcap_bulk = dfcap.loc[dfcap.tech != 'distpv'].copy()
dfcap_distpv = dfcap.loc[dfcap.tech == 'distpv'].copy()

### Write it
dfcap_bulk.to_csv(os.path.join(
    reedspath,'inputs','capacitydata','ReEDS_generator_database_final_RTS-GMLC.csv'),
    index=False,
)

################
#%% VRE profiles
### UPV
cfupv = pd.read_csv(
    os.path.join(rtspath,'RTS_Data','timeseries_data_files','PV',f'{timetype}_pv.csv')
).iloc[:8760].drop(['Year','Month','Day','Period'], axis=1)
## Normalize
sitecap_upv = pvsites.set_index('GEN UID')['cap']
cfupv /= sitecap_upv
## Concat 7 times
cfupv = pd.concat([cfupv]*7, axis=0, ignore_index=True)
### Assume that the utility-scale PV profile is in AC terms.
### Assume ILR=1.3
### Convert from MW_AC/MW_AC to MW_AC/MW_DC
ilr_utility = 1.3
cfupv /= ilr_utility


### Wind
cfwind = pd.read_csv(
    os.path.join(rtspath,'RTS_Data','timeseries_data_files','WIND',f'{timetype}_wind.csv')
).iloc[:8760].drop(['Year','Month','Day','Period'], axis=1)
## Normalize
sitecap_wind = windsites.set_index('GEN UID')['cap']
cfwind /= sitecap_wind
## Concat 7 times
cfwind = pd.concat([cfwind]*7, axis=0, ignore_index=True)
# cfwind.mean()


### Distributed (rooftop) PV
gen_rtpv = pd.read_csv(
    os.path.join(rtspath,'RTS_Data','timeseries_data_files','RTPV',f'{timetype}_rtpv.csv')
).iloc[:8760].drop(['Year','Month','Day','Period'], axis=1)
## Convert to zonal
gen_rtpv.columns = gen_rtpv.columns.map(dfcap_distpv.set_index('GEN UID').reeds_ba)
gen_rtpv = gen_rtpv.groupby(axis=1, level=0).sum()
## Convert to CF
cap_rtpv = dfcap_distpv.groupby('reeds_ba').cap.sum()
cf_rtpv = gen_rtpv / cap_rtpv
## Convert from MW_AC/MW_AC to MW_AC/MW_DC
ilr_dist = 1.1
cf_rtpv = cf_rtpv / ilr_dist

#%% Write ditributed PV
cf_rtpv.index = range(1, len(cf_rtpv.index) + 1)
cf_rtpv.T.round(3).to_csv(
    os.path.join(reedspath,'inputs','dGen_Model_Inputs','RTS','distPVCF_hourly_RTS.csv')
)

### For distpv capaciy, just keep it fixed over time
write_cap_rtpv = pd.concat(
    {y: cap_rtpv for y in range(2010,2051,2)}, axis=1
).rename_axis(None).multiply(growthfrac).dropna(axis=1)
write_cap_rtpv.round(3).to_csv(
    os.path.join(reedspath,'inputs','dGen_Model_Inputs','RTS','distPVcap_RTS.csv')
)

# #%% Get the wind class boundaries from the default ReEDS data
# exwind = LDC_prep.read_file(
#     os.path.join(reedspath,'inputs','variability','multi_year','wind-ons-reference')
# ).astype(np.float32)
# #%%
# exwind_mean = exwind.mean().rename('CF').reset_index().rename(columns={'index':'resource'})
# exwind_mean['class'] = exwind_mean.resource.map(lambda x: int(x.split('_')[0]))
# # exwind_mean = exwind_mean.groupby(['class']).CF.describe()

# #%% Take a look
# dfplot = pd.concat(
#     {c:exwind_mean.loc[exwind_mean['class']==c,'CF'] for c in exwind_mean['class'].unique()},
#     axis=1).sort_index(axis=1)
# plt.close()
# f,ax = plt.subplots(dpi=300)
# plots.plotquarthist(ax, dfplot)
# for col in cfwind:
#     ax.axhline(cfwind[col].mean(), c='k', lw=0.1)
# ax.plot()

### Compare to typical distpv data
# distPVcap = pd.read_csv(
#     os.path.join(
#         reedspath,'inputs','dGen_Model_Inputs',
#         'StScen2022_Mid_Case','distPVcap_StScen2022_Mid_Case.csv'),
#     index_col=0,
# )
# distPVcf = pd.read_csv(
#     os.path.join(
#         reedspath,'inputs','dGen_Model_Inputs',
#         'StScen2022_Mid_Case','distPVCF_hourly_StScen2022_Mid_Case.csv'),
#     index_col=0,
# )

#%% Eyeball the wind classes
windsite2class = {
    '309_WIND_1': 9,
    '317_WIND_1': 8,
    '303_WIND_1': 9,
    '122_WIND_1': 8,
}

### Create the supply curve
## region,class,sc_point_gid,capacity,supply_curve_cost_per_mw,dist_km
windsc = pd.DataFrame({
    'region': windsites['Bus ID'].map(bus2zone),
    'class': windsites['GEN UID'].map(windsite2class),
    'sc_point_gid': windsites.index,
    'capacity': cap_avail,
    'supply_curve_cost_per_mw': cost_spur,
    'trans_adder_per_MW': cost_reinforcement,
    'capital_adder_per_MW': 0,
    'dist_km': dist_km,
    'reinforcement_dist_km': reinforcement_dist_km,
}).reset_index(drop=True)
### Add some randomness
np.random.seed(seed)
windsc['supply_curve_cost_per_mw'] += np.random.standard_normal(len(windsc)) * cost_spur * sigma
windsc['dist_km'] = windsc['supply_curve_cost_per_mw'] / cost_spur * dist_km
windsc['GEN UID'] = windsc.sc_point_gid.map(site2gen)
# windsc
### Write it
windsc.drop('GEN UID', axis=1).round(3).to_csv(
    os.path.join(reedspath,'inputs','supplycurvedata','wind-ons_supply_curve-rts.csv'),
    index=False,
)


# #%% Get the PV class boundaries from the default ReEDS data
# exupv = LDC_prep.read_file(
#     os.path.join(reedspath,'inputs','variability','multi_year','upv-reference')
# ).astype(np.float32)
# #%%
# exupv_mean = exupv.mean().rename('CF').reset_index().rename(columns={'index':'resource'})
# exupv_mean['class'] = exupv_mean.resource.map(lambda x: int(x.split('_')[0]))
# # exupv_mean = exupv_mean.groupby(['class']).CF.describe()

# #%% Take a look
# dfplot = pd.concat(
#     {c:exupv_mean.loc[exupv_mean['class']==c,'CF'] for c in exupv_mean['class'].unique()},
#     axis=1).sort_index(axis=1)
# plt.close()
# f,ax = plt.subplots(dpi=300)
# plots.plotquarthist(ax, dfplot)
# for col in cfupv:
#     ax.axhline(cfupv[col].mean(), c='k', lw=0.1)
# ax.plot()

#%% Eyeball the PV classes
pvsite2class = {
    k: 3 if v <= 0.2 else 5
    for k,v in cfupv.mean().items()
}

## region,class,sc_point_gid,capacity,supply_curve_cost_per_mw,dist_km
pvsc = pd.DataFrame({
    'region': pvsites['Bus ID'].map(bus2zone),
    'class': pvsites['GEN UID'].map(pvsite2class),
    'sc_point_gid': pvsites.index,
    'capacity': cap_avail,
    'supply_curve_cost_per_mw': cost_spur,
    'dist_km': dist_km,
    'reinforcement_dist_km': reinforcement_dist_km,
}).reset_index(drop=True)
### Add some randomness
np.random.seed(seed)
pvsc['supply_curve_cost_per_mw'] += np.random.standard_normal(len(pvsc)) * cost_spur * sigma
pvsc['dist_km'] = pvsc['supply_curve_cost_per_mw'] / cost_spur * dist_km
pvsc['GEN UID'] = pvsc.sc_point_gid.map(site2gen)
# pvsc
### Write it
pvsc.drop('GEN UID', axis=1).round(3).to_csv(
    os.path.join(reedspath,'inputs','supplycurvedata','upv_supply_curve-rts.csv'),
    index=False,
)

#%%### Aggregate the profiles
scaler = windsc.groupby(['class','region']).capacity.sum()
scaler.index = scaler.index.map(lambda x: str(x[0])+'_'+x[1])
wind_sitescaled = cfwind * windsc.set_index('GEN UID').capacity
wind_sitescaled = (
    wind_sitescaled
    .rename(columns=dict(zip(
        windsc['GEN UID'], windsc['class'].astype(str)+'_'+windsc['region'])))
    .groupby(axis=1, level=0).sum()
    / scaler
)
wind_sitescaled.index += 1

scaler = pvsc.groupby(['class','region']).capacity.sum()
scaler.index = scaler.index.map(lambda x: str(x[0])+'_'+x[1])
pv_sitescaled = cfupv * pvsc.set_index('GEN UID').capacity
pv_sitescaled = (
    pv_sitescaled
    .rename(columns=dict(zip(
        pvsc['GEN UID'], pvsc['class'].astype(str)+'_'+pvsc['region'])))
    .groupby(axis=1, level=0).sum()
    / scaler
)
pv_sitescaled.index += 1

#%% Write wind
LDC_prep.write_h5(
    df=wind_sitescaled, filename=os.path.join(
        reedspath,'inputs','variability','multi_year','wind-ons-rts.h5'),
    tablename='cf', overwrite=True,
)
#%% Write PV
LDC_prep.write_h5(
    df=pv_sitescaled, filename=os.path.join(
        reedspath,'inputs','variability','multi_year','upv-rts.h5'),
    tablename='cf', overwrite=True,
)


#################################
#%% Prescribed and exogenous wind
### Prescribed
## wind-ons_prescribed_builds_rts.csv
## region,year,capacity
## year values range from 2010-2026
prescribed = (
    windsites
    .assign(resource_region=windsites['Bus ID'].map(bus2zone))
    .loc[windsites.StartYear >= start_year]
    .groupby(['resource_region','StartYear'], as_index=False).cap.sum()
    .rename(columns={'resource_region':'region','StartYear':'year','cap':'capacity'})
    .round(1)
)
prescribed.to_csv(
    os.path.join(reedspath,'inputs','capacitydata','wind-ons_prescribed_builds_rts.csv'),
    index=False,
)

#%% Exogenous (copied from hourlize)
## wind-ons_exog_cap_rts.csv
## *tech,region,sc_point_gid,year,capacity
## year values range from 2010-2039
exogenous = (
    windsites.assign(resource_region=windsites['Bus ID'].map(bus2zone))
    .reset_index().rename(columns={
        'resource_region':'region','cap':'capacity'})
)
exogenous['class'] = exogenous['GEN UID'].map(windsite2class)
exogenous = exogenous[['sc_point_gid','class','region','RetireYear','capacity']].copy()
max_exog_ret_year = exogenous['RetireYear'].max()
ret_year_ls = list(range(start_year,max_exog_ret_year + 1))
exogenous = exogenous.pivot_table(
    index=['sc_point_gid','class','region'], columns='RetireYear', values='capacity')
# Make a column for every year until the largest retirement year
exogenous = exogenous.reindex(columns=ret_year_ls).fillna(method='bfill', axis='columns')
exogenous = pd.melt(
    exogenous.reset_index(), id_vars=['sc_point_gid','class','region'],
    value_vars=ret_year_ls, var_name='year', value_name='capacity')
exogenous = exogenous[exogenous['capacity'].notnull()].copy()
exogenous['tech'] = 'wind-ons_' + exogenous['class'].astype(str)
exogenous = exogenous[['tech','region','sc_point_gid','year','capacity']].copy()
exogenous = exogenous.groupby(
    ['tech','region','sc_point_gid','year'], sort=False, as_index=False).sum()
exogenous['capacity'] =  exogenous['capacity'].round(3)

exogenous.rename(columns={'tech':'*tech'}).to_csv(
    os.path.join(reedspath,'inputs','capacitydata','wind-ons_exog_cap_rts.csv'),
    index=False,
)


###########
#%%### Load
dfload = (
    pd.read_csv(
        os.path.join(rtspath,'RTS_Data','timeseries_data_files','Load',
                     f'{timetype}_regional_Load.csv'))
    .rename(columns={str(i):area2zone[i] for i in area2zone})
    .drop(['Year','Month','Day','Period'], axis=1)
    .iloc[:8760]
)

dfout = {y:dfload*growthfrac[y] for y in years}

#%% Write it
outpath = os.path.join(reedspath,'inputs','loaddata','RTS_load_hourly.h5')
with h5py.File(outpath, 'w') as f:
    f.create_dataset('columns', data=dfload.columns, dtype=h5py.special_dtype(vlen=str))
    for y in years:
        f.create_dataset(str(y), data=dfout[y], dtype=np.float32, compression='gzip')

# growthfrac.round(3).to_csv(
#     os.path.join(reedspath,'inputs','loaddata','demand_RTS.csv'),
#     header=None,
# )

###
# pd.read_hdf(os.path.join(reedspath,'inputs','loaddata','EPHIGH_load_hourly.h5'))
# LDC_prep.read_file(outpath.replace('.h5',''))

################
#%% Transmission
### Calculated using https://github.nrel.gov/pbrown/TSC/tree/rts
### Load TSC outputs
cases = [
    '20230208_RTS_R0',
    '20230208_RTS_R1',
]

runspath = os.path.expanduser('~/github/TSC/runs')
###### Load the ratings
dictin = {}
for case in cases:
    reverse = int(case.split('_')[2].strip('R'))
    ### Get output folders
    folders = glob(os.path.join(runspath,case,'*'))
    for f in folders:
        ### Parse it
        interface = os.path.basename(f).replace('\uf07c','|')
        ### Load the transfer capacities
        outputs = glob(os.path.join(f,'outputs','transfer*'))
        for i in outputs:
            ### Parse it
            contingency = int(os.path.basename(i).split('_')[-1].split('.')[0].strip('n'))
            ### Load it
            dictin[reverse,interface,contingency] = pd.read_csv(
                i, header=0, index_col='i', squeeze=True
            ## Use reindex in case the dataframe is empty
            ).reindex([interface]).fillna(0).values[0]

#%% Combine into dataframe
dfin = pd.DataFrame(dictin, index=['MW']).T
dfin.index = dfin.index.rename(['R','interface','contingency'])
dfin = dfin.reset_index()

transfer_capacities = dfin.pivot(index='interface',columns=['R','contingency'],values='MW')
transfer_capacities.columns = (
    transfer_capacities.columns.map(lambda x: f"MW_{'f' if not x[0] else 'r'}{x[1]}"))
transfer_capacities = (
    transfer_capacities.fillna(0).abs()
    .reset_index()
)
transfer_capacities['r'] = transfer_capacities.interface.map(lambda x: area2zone[x.split('||')[0]])
transfer_capacities['rr'] = transfer_capacities.interface.map(lambda x: area2zone[x.split('||')[1]])
transfer_capacities.interface = transfer_capacities.r + '||' + transfer_capacities.rr

#%% Write it
transfer_capacities[['interface','r','rr','MW_f0','MW_r0','MW_f1','MW_r1']].round(3).to_csv(
    os.path.join(reedspath,'inputs','transmission',f'transmission_capacity_init_AC_RTS.csv')
)

#%% Transmission distances - just take the centroid of the buses in each area
import geopandas as gpd
os.environ['PROJ_NETWORK'] = 'OFF'
dfbus = (
    gpd.read_file(
        os.path.join(rtspath,'RTS_Data','FormattedData','GIS','bus.geojson'))
    .to_crs('ESRI:102008')
)
dfbus['zone'] = 'z' + dfbus['Area'].astype(str)
centroids = dfbus.dissolve('zone').centroid
distance = {}
for zone in centroids.index:
    distance[zone] = np.sqrt(
        (centroids.x - float(centroids[[zone]].x))**2
        + (centroids.y - float(centroids[[zone]].y))**2
    ) / 1e3 / 1.609
dfdistance = (
    pd.concat(distance).round(2)
    .rename_axis(['r','rr']).rename('length_miles')
    .replace(0,np.nan).dropna()
    .reset_index()
)
### Use the median $/MWmile cost from US ReEDS
transcost = pd.read_csv(
    os.path.join(
        reedspath,'inputs','transmission','transmission_distance_cost_500kVac.csv'))
transcost['USD2004perMWmile'] = transcost.USD2004perMW / transcost.length_miles

dfdistance['USD2004perMW'] = (
    dfdistance.length_miles * transcost.USD2004perMWmile.median()
).round(2)

### Write it
dfdistance.to_csv(
    os.path.join(
        reedspath,'inputs','transmission','transmission_distance_cost_RTS.csv'),
    index=False,
)


##############
#%% Financials
reg_cap_cost_mult_default = pd.read_csv(
    os.path.join(reedspath,'inputs','financials','reg_cap_cost_mult_default.csv')
)
techs = reg_cap_cost_mult_default.i.unique()
reg_cap_cost_mult = pd.Series({
    (r,i):1 for r in zones for i in techs
}).rename_axis(['r','i']).rename('reg_cap_cost_mult')
reg_cap_cost_mult.to_csv(
    os.path.join(reedspath,'inputs','financials','reg_cap_cost_mult_rts.csv')
)


#########
#%% Hydro
### TODO

