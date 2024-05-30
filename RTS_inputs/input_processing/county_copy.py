#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:41:27 2023

@author: mbrown1
"""

import os
import shutil
from pathlib import Path

county_dir = "/users/mbrown1/Desktop/countylevel/countylevel/"
indir = "/users/mbrown1/Documents/GitHub/ReEDS-2.0/inputs/"

verbose = False

for path, subdirs, files in os.walk(county_dir):
    #for all the files
    for name in files:
        # remove _county from file name
        #name_copy = name.replace("_county","")
        name_copy = name
        
        #outpath replace the input path from county_dir with the reeds/inputs/ directory
        outpath = os.path.join(str(os.path.join(path, name)).replace(county_dir,indir))

        # file name ultimately written out
        out_file = outpath.replace(name,name_copy)

        # file name for keeping track of original file
        orig_copy = out_file.replace(name_copy,"orig_"+name_copy)

        #print out info if verbose is true
        if verbose:
            print("copying file: " + name)
            print("as: " + name_copy)
            print(" ")
        # if the file already exists..
        if Path(out_file).is_file():
            # and if the orig_... file does not already exist
            if not Path(orig_copy).is_file():
                # copy the original file as: orig_...
                shutil.copy(out_file,orig_copy)
        
        # copy over the file to the .. outpath
        shutil.copy(os.path.join(path, name),out_file)
