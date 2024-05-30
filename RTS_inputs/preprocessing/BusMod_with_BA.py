#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df_bus_mod = pd.read_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\RTS-GMLC-master\\RTS-GMLC-master\\RTS_Data\\SourceData\\bus_mod.csv', header=0)#, usecols=['STNAME','CTYNAME','POPESTIMATE2021','STATE','COUNTY'])
df_heirarchy_nodal = pd.read_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\hierarchy_rts_nodal.csv', header = 0)

df_bus_mod.columns = df_bus_mod.columns.str.replace(' ', '_')
df_bus_mod['Bus_ID'] = df_bus_mod['Bus_ID'].astype(str)
df_bus_mod['reeds_nodal'] = "b" + df_bus_mod['Bus_ID']



# In[3]:


df_new_bus_mod = pd.merge(df_bus_mod, df_heirarchy_nodal, left_on='reeds_nodal', right_on='*Nodal', how='inner')

df_new_bus_mod


# In[4]:


df_new_bus_mod.drop(['nercr', 'transreg', 'st_y', 'interconnect', 'country', 'usda', 'aggreg'], axis=1, inplace=True)

df_new_bus_mod


# In[5]:


df_new_bus_mod.drop(['county_name_y', '*Nodal'], axis=1, inplace=True)

df_new_bus_mod


# In[6]:


df_new_bus_mod.rename(columns = {'st_x':'st', 'county_name_x':'county_name'}, inplace = True)
df_new_bus_mod


# In[7]:


df_new_bus_mod.to_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\RTS_Data\\SourceData\\bus_mod_updatedwithBA.csv', sep=',', encoding='utf-8', index=False)


# In[ ]:




