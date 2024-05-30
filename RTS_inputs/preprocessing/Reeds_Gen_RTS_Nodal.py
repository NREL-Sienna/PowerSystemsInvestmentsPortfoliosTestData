#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df_nodal_hierarchy = pd.read_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\hierarchy_rts_nodal.csv', header=0)

df_reeds_gens = pd.read_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\capacitydata\\ReEDS_generator_database_final_RTS-GMLC_withRTPV.csv', header=0)

df_reeds_gens
#df_p1_census = pd.read_csv('co-est2021-alldata.csv', header=0, usecols=['STNAME','CTYNAME','POPESTIMATE2021','STATE','COUNTY'], nrows=3195, encoding='latin-1')
#print(df_p1_census)

#df_p1_counties = pd.read_excel('county_reeds_corrected0310.xlsx', header=0, usecols=['NAME','STATE_NAME','FIPS','PCA_REG'])
#print(df_p1_counties)

#df_heirarchy_others = pd.read_csv('hierarchy_fips.csv', header = 0, usecols = ['FIPS','BA','nercr','transreg','cendiv','st', 'interconnect','country','usda','aggreg'])
#df_heirarchy_others = pd.read_csv('Reeds_SpatialFlex\\fprice\\hierarchy_bokeh.csv', header = 0) #, usecols = ['FIPS','BA','nercr','transreg','cendiv','st', 'interconnect','country','usda','aggreg'])

#df_heirarchy_others


# In[4]:


df_reeds_gens['Bus ID'] = df_reeds_gens['Bus ID'].astype(str)

df_reeds_gens['reeds_ba'] = "b" + df_reeds_gens['Bus ID']
df_reeds_gens


# In[5]:


#left_on = reeds_ba before switching column name; updated should be reeds_nodal
df_hierarchy_nodal_gens = pd.merge(df_reeds_gens, df_nodal_hierarchy, left_on=['reeds_ba'], right_on=['*Nodal'], how='inner')
df_hierarchy_nodal_gens


# In[6]:


df_hierarchy_nodal_gens.drop(['*Nodal', 'FIPS', 'nercr', 'transreg','st', 'interconnect', 'country', 'usda', 'aggreg', 'county_name'], axis=1, inplace=True)
df_hierarchy_nodal_gens


# In[7]:


df_hierarchy_nodal_gens.rename(columns = {'reeds_ba':'reeds_nodal', 'BA':'reeds_ba'}, inplace = True)
df_hierarchy_nodal_gens


# In[8]:


# Write dataframe to csv file (UNCOMPRESSED)
df_hierarchy_nodal_gens.to_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\capacitydata\\ReEDS_generator_database_final_RTS-GMLC_updated_nodal.csv', sep=',', encoding='utf-8', index=False)


# In[ ]:




