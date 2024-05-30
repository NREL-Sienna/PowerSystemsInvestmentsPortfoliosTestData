#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd

df_branch_mod = pd.read_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\RTS-GMLC-master\\RTS-GMLC-master\\RTS_Data\\SourceData\\branch.csv', header=0)

df_branch_mod


# In[2]:


df_branch_mod.columns = df_branch_mod.columns.str.replace(' ', '_')
df_branch_mod


# In[3]:


df_branch_mod['From_Bus'] = df_branch_mod['From_Bus'].astype(str)
df_branch_mod['To_Bus'] = df_branch_mod['To_Bus'].astype(str)

df_branch_mod['From_Bus'] = "b" + df_branch_mod['From_Bus']
df_branch_mod['To_Bus'] = "b" + df_branch_mod['To_Bus']
df_branch_mod


# In[4]:


df_AC_trx_cap_nodal = df_branch_mod[['From_Bus', 'To_Bus', 'Cont_Rating']]
df_AC_trx_cap_nodal


# In[5]:


df_AC_trx_cap_nodal['interface'] = df_AC_trx_cap_nodal['From_Bus'] + "||" + df_AC_trx_cap_nodal['To_Bus']

df_AC_trx_cap_nodal


# In[6]:


df_AC_trx_cap_nodal = df_AC_trx_cap_nodal.reindex(columns = ['interface', 'From_Bus', 'To_Bus', 'Cont_Rating'])
df_AC_trx_cap_nodal.rename(columns = {'Cont_Rating':'MW_f0'}, inplace = True)
df_AC_trx_cap_nodal


# In[7]:


df_AC_trx_cap_nodal['MW_r0'] = df_AC_trx_cap_nodal['MW_f0']
#df_AC_trx_cap_nodal['MW_f1'] = df_AC_trx_cap_nodal['MW_f0']
#df_AC_trx_cap_nodal['MW_r1'] = df_AC_trx_cap_nodal['MW_f0']
df_AC_trx_cap_nodal


# In[8]:


# Write dataframe to csv file (UNCOMPRESSED)
df_AC_trx_cap_nodal.to_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\transmission\\transmission_capacity_init_AC_rts_nodal.csv', sep=',', encoding='utf-8', index=False)


# In[9]:


df_r_rr = df_AC_trx_cap_nodal[['From_Bus', 'To_Bus']]

df_r_rr


# In[10]:


df_r_rr.rename(columns = {'From_Bus':'*r', 'To_Bus':'rr'}, inplace = True)

df_r_rr


# In[11]:


# Write dataframe to csv file
df_r_rr.to_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\transmission\\r_rr_rts_nodal.csv', sep=',', encoding='utf-8', index=False)


# In[12]:


df_r_rr_dist = df_branch_mod[['From_Bus', 'To_Bus', 'Length']]
df_r_rr_dist


# In[13]:


df_r_rr_dist['USD2004perMW'] = df_r_rr_dist['Length'] * 1758.919
df_r_rr_dist


# In[14]:


# Write dataframe to csv file
df_r_rr_dist.to_csv('C:\\Users\\akarmaka\\Documents\\GitHub\\ReEDS-2.0\\inputs\\transmission\\transmission_distance_cost_rts.csv', sep=',', encoding='utf-8', index=False)


# In[ ]:




