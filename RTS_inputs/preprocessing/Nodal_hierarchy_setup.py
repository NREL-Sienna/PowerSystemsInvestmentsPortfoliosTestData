#!/usr/bin/env python
# coding: utf-8

# In[21]:
#Run this cell only for generating county names for bus locations - RTS dataset

import csv
# from geopy.geocoders import Nominatim

# def county_locator(latitude, longitude):
#     geolocator = Nominatim(user_agent="county-lookup")
#     location = geolocator.reverse((latitude, longitude), exactly_one=True)
#     address = location.raw['address']
#     county_name = address.get('county', None)
#     state_name = address.get('state', '')
#     return county_name, state_name

# #def main():
# bus_file_csv = "C:\\Users\\akarmaka\\Documents\\GitHub\\RTS-GMLC-master\\RTS-GMLC-master\\RTS_Data\\SourceData\\bus.csv"
# mod_bus_file = "C:\\Users\\akarmaka\\Documents\\GitHub\\RTS-GMLC-master\\RTS-GMLC-master\\RTS_Data\\SourceData\\bus_mod.csv"
# with open(bus_file_csv, 'r') as file:
#     reader = csv.DictReader(file)
#     fieldnames = reader.fieldnames
#     fieldnames += ['county_name', 'st']
#     with open(mod_bus_file, 'w', newline='') as output_file:
#         writer = csv.DictWriter(output_file, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in reader:
#             latitude = float(row['lat'])
#             longitude = float(row['lng'])
            
#             county_name, state_name = county_locator(latitude, longitude)
            
#             row['county_name'] = county_name
#             row['st'] = state_name
            
#             writer.writerow(row)
            
#             #print(f"Lat: {latitude}, longitude: {longitude}, County FIPS: {county_name}, State: {state_name}")
        
# print("Bus file now has county names and state names!")
    

# In[57]:


import pandas as pd

df_bus_mod = pd.read_csv('D:\\akash\\reeds\\ReEDS-2.0\\RTS_Data\\SourceData\\bus_mod_updatedwithBA.csv', header=0)

#df_p1_census = pd.read_csv('co-est2021-alldata.csv', header=0, usecols=['STNAME','CTYNAME','POPESTIMATE2021','STATE','COUNTY'], nrows=3195, encoding='latin-1')
#print(df_p1_census)

df_p1_counties = pd.read_excel('C:\\Users\\akarmaka\\Downloads\\county_reeds_corrected0310.xlsx', header=0, usecols=['NAME','STATE_NAME','FIPS','PCA_REG'])
#print(df_p1_counties)

df_heirarchy_others = pd.read_csv('D:\\akash\\reeds\\ReEDS-2.0\\inputs\\hierarchy.csv', header = 0)
#df_heirarchy_others = pd.read_csv('Reeds_SpatialFlex\\fprice\\hierarchy_bokeh.csv', header = 0) #, usecols = ['FIPS','BA','nercr','transreg','cendiv','st', 'interconnect','country','usda','aggreg'])

df_heirarchy_others.rename(columns = {'*county':'FIPS'}, inplace = True)
df_heirarchy_others['FIPS'] = df_heirarchy_others['FIPS'].astype(str)
df_heirarchy_others['FIPS'] = df_heirarchy_others['FIPS'].str.replace('p','')

df_heirarchy_others
# In[58]:


df_bus_mod['county_name'] = df_bus_mod['county_name'].str.replace(' County', '')
df_bus_mod['county_name'] = df_bus_mod['county_name'].str.replace('CAL Fire ', '')
df_bus_mod['county_name'] = df_bus_mod['county_name'].str.replace(' Unit', '')
#df_bus_mod.rename(columns = {'Bus ID':'Bus_ID', 'Bus Name':'Bus_Name', 'Bus Type': 'Bus_Type', 'MW Load': 'MW_Load', 'MVAR Load': 'MVAR Load'}, inplace = True)
df_bus_mod.columns = df_bus_mod.columns.str.replace(' ', '_')
df_bus_mod


# In[59]:


df_bus_mod.dtypes


# In[60]:


#df_bus_mod = df_bus_mod.insert(17,'Nodal_Zone', 0)
df_bus_mod['Bus_ID'] = df_bus_mod['Bus_ID'].astype(str)

df_bus_mod['Nodal'] = "b" + df_bus_mod['Bus_ID']
df_bus_mod


# In[66]:


df_p1_counties['FIPS']=df_p1_counties['FIPS'].apply(lambda x: '{0:0>5}'.format(x))
df_p1_counties


# In[67]:


df_p1_counties['FIPS'] = df_p1_counties['FIPS'].astype('int64')
df_p1_counties
df_heirarchy_others['FIPS'] = df_heirarchy_others['FIPS'].astype('int64')
df_heirarchy_others

# In[68]:


df_heirarchy_counties = pd.merge(df_heirarchy_others, df_p1_counties, on='FIPS', how='right')
#df_heirarchy_counties = pd.merge(df_heirarchy_others, df_p1_counties, left_on='rb', right_on='PCA_REG', how='inner')
df_heirarchy_counties


# In[73]:


df_hierarchy_nodal_add = pd.merge(df_heirarchy_counties, df_bus_mod, left_on=['NAME', 'STATE_NAME'], right_on=['county_name', 'st'], how='inner')
df_hierarchy_nodal_add


# In[ ]:





# In[74]:


df_hierarchy_nodal_add.drop(['Bus_Name','BaseKV','Bus_Type', 'MW_Load', 'MVAR_Load','V_Mag', 'V_Angle', 'MW_Shunt_G', 'MVAR_Shunt_B', 'Area', 'Zone', 'lat', 'lng', 'st_y', 'Bus_ID', 'Sub_Area', 'STATE_NAME','PCA_REG','NAME','STATE_NAME'], axis=1, inplace=True)
df_hierarchy_nodal_add


# In[77]:


df_hierarchy_nodal_add.rename(columns = {'st_x':'st', 'Nodal':'nodal'}, inplace = True)
df_hierarchy_nodal_add['FIPS_x'] = df_hierarchy_nodal_add['FIPS_y']
df_hierarchy_nodal_add.drop(['county_name_y','FIPS_y'], axis=1, inplace=True)
df_hierarchy_nodal_add.rename(columns = {'FIPS_x':'FIPS', 'county_name_x':'county_name'}, inplace = True)
df_hierarchy_nodal_add['FIPS']=df_hierarchy_nodal_add['FIPS'].apply(lambda x: '{0:0>5}'.format(x))
df_hierarchy_nodal_add.sample(20)


# In[76]:


# df_hierarchy_nodal_add['FIPS'] = "p" + df_hierarchy_nodal_add['FIPS'].astype(str)
# df_hierarchy_nodal_add


# In[78]:


len_check = df_hierarchy_nodal_add['FIPS'].str.len()==5
len_check


# In[79]:


df_hierarchy_nodal_add.sample(15)


# In[80]:


df_hierarchy_nodal_add = df_hierarchy_nodal_add.reindex(columns=['nodal', 'FIPS', 'BA', 'nercr', 'transreg', 'transgrp', 'cendiv', 'st', 'interconnect', 'st_interconnect', 'country', 'usda_region', 'aggreg', 'county_name'])


# In[81]:


df_hierarchy_nodal_add


# In[82]:


df_hierarchy_nodal_add.rename(columns = {'nodal':'*nodal', 'FIPS':'county', 'BA':'ba'}, inplace = True)
df_hierarchy_nodal_add


# In[83]:


# Write dataframe to csv file (UNCOMPRESSED)
df_hierarchy_nodal_add.to_csv('D:\\akash\\reeds\\ReEDS-2.0\\inputs\\hierarchy_rts_nodal_v2.csv', sep=',', encoding='utf-8', index=False)


# In[ ]:




