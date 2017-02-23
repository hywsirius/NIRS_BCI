# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 14:23:33 2015

@author: bci2000
"""
import os
from ctypes import *
os.chdir('C:\NIRx\NIRx SDK package 2011-08-15\TomographyMATLAB_API\LibraryM\TomoAPI')
def hello():
    print 'hello'
API=cdll.LoadLibrary('TomographyAPI')

# stop function
def stop():
    API.tsdk_stop()
    API.tsdk_disconnect()
    API.tsdk_close()