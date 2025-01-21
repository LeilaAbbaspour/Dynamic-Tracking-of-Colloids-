#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.font_manager as font_manager

# Map the missing module and class to the current version
from pandas import Index

# Create a dummy module to redirect `Int64Index` and `Float64Index`
class NumericIndexDummy:
    pass

# Redirect the missing module and attributes to the current `pandas.Index`
sys.modules['pandas.core.indexes.numeric'] = NumericIndexDummy
NumericIndexDummy.Int64Index = Index
NumericIndexDummy.Float64Index = Index

mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext']=True
plt.rcParams['text.usetex'] = True


# In[97]:


dataaddr = r"/Tracking-Experimental-Data/AnalyzedData/20240725/emulsion data_stoma with or without fuel/"
nfile=0
filename=os.listdir(dataaddr)[nfile]
filename = os.path.splitext(filename)[0]
directory_path = os.path.join(dataaddr, filename)
directory_path

#list of files
files = os.listdir(dataaddr)

# Dictionary to hold categorized files
categorized_files = defaultdict(list)

for file in files:
    filename_without_ext = os.path.splitext(file)[0]
    parts = filename_without_ext.split('_')

    # Automatically extract the part containing `urea+dye`
    for part in parts:
        if 'mMurea+dye' in part:
            category = part
            break
    else:
        category = "Unknown"  # Handle cases where 'urea+dye' is not found

    # Add the full file name to the corresponding category
    categorized_files[category].append(file)





specific_categories = ['0mMurea+dye', '10mMurea+dye', '25mMurea+dye', '100mMurea+dye', '500mMurea+dye']
specific_category = '100mMurea+dye'
fig,ax=plt.subplots(1,1,figsize=(10,8))
labelsize=20
fontsize=30
for specific_category in specific_categories:
    for f in categorized_files[specific_category]:
        path = os.path.join(dataaddr, os.path.splitext(f)[0])
    # How to read Data after storing
        with open(path + "/df.pkl", "rb") as file:
            df = pickle.load(file)
            plt.plot(df.em)
    
    plt.plot(df.em.index.tolist(),df.em.index.tolist(),lw=5, color='black')
    plt.xlabel('lag time t',fontsize=fontsize)
    plt.ylabel('$<\Delta r^2>[\mu m^2]$',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.savefig("./MSD-Plot/")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()



# Plot the exponent of the MSD for different specific catagories


# Initialize a dictionary to store alpha values for each category
alpha_values_by_category = {category: [] for category in specific_categories}

# Read the data for each category and store alpha values
for specific_category in specific_categories:
    for f in categorized_files[specific_category]:
        try:
            path = os.path.join(dataaddr, os.path.splitext(f)[0])
            
            # Read the DataFrame with the MSD data
            with open(path + "/df.pkl", "rb") as file:
                df = pickle.load(file)
            
            # Ensure alpha_opt is treated as a list before extending the category list
            if isinstance(df.alpha_opt, np.ndarray): 
                alpha_values_by_category[specific_category].extend(df.alpha_opt)
            else:  
                alpha_values_by_category[specific_category].append(df.alpha_opt)
        except:
            continue
        

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
labelsize =25
fontsize = 30
# Create a list of alpha values for each category
alpha_data = [alpha_values_by_category[category] for category in specific_categories]

# Create the boxplot for alpha values comparison across categories
ax.boxplot(alpha_data, tick_labels=specific_categories, patch_artist=True, medianprops={'color': 'black'}, 
           boxprops=dict(facecolor='royalblue', color='black'), whiskerprops=dict(color='black'))

ax.set_xlabel('Specific Categories', fontsize=fontsize)
ax.set_ylabel('Optimized Alpha', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
plt.xticks(rotation=45, ha='right')

plt.show()

# Plot the diffusion coefficent of the colloids for different specific catagories


# Initialize a dictionary to store alpha values for each category
D_values_by_category = {category: [] for category in specific_categories}

# Read the data for each category and store alpha values
for specific_category in specific_categories:
    for f in categorized_files[specific_category]:
        try:
            path = os.path.join(dataaddr, os.path.splitext(f)[0])
            
            # Read the DataFrame with the MSD data
            with open(path + "/df.pkl", "rb") as file:
                df = pickle.load(file)
            
            # Ensure D_opt is treated as a list before extending the category list
            if isinstance(df.D_opt, np.ndarray):  # If it's an array-like object
                alpha_values_by_category[specific_category].extend(df.D_opt)
            else:  # If it's a single number, wrap it in a list
                alpha_values_by_category[specific_category].append(df.D_opt)
        except:
            continue


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
labelsize = 20
fontsize = 30

for specific_category in specific_categories:
    ax.scatter([specific_category] * len(alpha_values_by_category[specific_category]), 
               alpha_values_by_category[specific_category], 
               label=specific_category, 
               alpha=0.7, edgecolors='w', s=100)  # Scatter plot with a bit of transparency



ax.set_xlabel('Specific Categories', fontsize=fontsize)
ax.set_ylabel('Optimized $D_{eff}$', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=1,linestyle='--',c="black")

plt.savefig("./MSD-Plot/D.png",dpi=300)
plt.show()



