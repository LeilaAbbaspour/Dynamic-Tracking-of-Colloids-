#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from pathlib import Path
import os
import ColloidsTracking as trk
import pims
import sys
import cv2

if len(sys.argv) != 2:
    print("Usage: python trackscript.py nfile")
    sys.exit(1)
nfile = int(sys.argv[1])

folders=["20240725/emulsion data_stoma with or without fuel/",#0
         "20240725/stomatocytes(+tracers in the second portion of video)/",#1
         "20240405_emptystoma +fuel in emulsion/",#2
        "20240412_fueled stoma in emulsion/",#3
         "20240514_emu_titrationcontrol/",#4
         "20240514_titration ctrl in emu/",#5
        ]

dataaddr = r"/ParticleTracking/Raw-Data-Experiment/"+folders[0]
base_path = "./AnalyzedData/Latest/"+folders[0]


filename=os.listdir(dataaddr)[nfile]
datafilename = dataaddr+filename
nd2 = pims.open(datafilename) 
T = len(nd2)



filename = os.path.splitext(filename)[0]

# Combine into the full directory path
directory_path = os.path.join(base_path, filename)

# Create the directory
os.makedirs(directory_path, exist_ok=True)


min_stub_frame_length = int(len(nd2)*0.1)
shared_parameters = {
    'savefilename': None,
    'datafilename': datafilename,
    'savefiledirectory': directory_path,
    'series_id': None,
    'T': T,

    'locator_params': {
        'diameter' : 11, 
        'minmass' : 1500,
        'separation' : 11,
        'engine' : 'numba'
    },
    'l_blur':5,
    'cluster_radius': 30,

    'search_range': 11,
    'memory': 10,
    'min_stub_frame_length': 30
}


syst = trk.System(**shared_parameters)
syst.savefilename = directory_path+"/df.pkl"
syst.Get_Frames()
syst.Find_Tracks()
syst.min_stub_frame_length = min_stub_frame_length
syst.Filter_Stubs()
syst.Calculate_MSD_quadratic()
syst.Calculate_MSD_linear()
syst.create_video()
syst.Save()







