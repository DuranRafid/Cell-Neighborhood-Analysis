#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


import itertools


# In[33]:


def get_contact_matrices(patient_group):
    with open(patient_group+'_CN_CN_CM.txt','r') as f:
        text = f.readlines()
    allMat = []
    start = 2
    while(start<len(text)):
        numbers = []
        for i in range(start,start+14):
            for x in text[i].split(' '):
                try:
                    numbers.append(float(x.rstrip('\n').strip("[]")))
                except:
                    print(end='')
        arr = np.array(numbers).reshape(7,7)
        allMat.append(arr)
        start += 15
    return allMat


# In[34]:


IDCmatrices = get_contact_matrices('IDC')
ILCmatrices = get_contact_matrices('ILC')


# In[35]:


len(IDCmatrices), len(ILCmatrices)


# In[37]:


def get_list(matrices, i, j):
    l = []
    for x in matrices:
        l.append(x[i][j])
    return l


# In[58]:


from scipy.stats import mannwhitneyu


# In[83]:


#f = open('CN-CN contact statistical significance corrected p-values.txt','w')
#f.write('The following CN-CN contact pairs are statistically significantly different among ILC and IDC after bonferroni correction'+'\n')
pvals = []
for i in range(7):
    for j in range(i,7):
        IDClist = get_list(IDCmatrices, i,j)
        ILClist = get_list(ILCmatrices, i,j)
        w,p = mannwhitneyu(IDClist, ILClist)
        #pval = perm_test(IDClist,ILClist)
        pvals.append(p)
        #f.write('Bonferroni corrected p-value for CN {}-CN {} contacts: {}'.format(i,j,p/30)+'\n')
#f.close()


# In[82]:


from statsmodels.stats.multitest import multipletests


# In[86]:


_,corrected_pvals, _, _ = multipletests(pvals,method='bonferroni')


# In[88]:


for x in corrected_pvals:
    print(x)


# In[ ]:




