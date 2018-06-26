#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import glob
from astropy.io import fits

def LoadImages(Output,DataType):
    
    if DataType == 'E':
        Source = glob.glob('EuclidImages/*fits')
        hdu_list = fits.open(Source[0])
        Images = np.zeros((Output.shape[0],1,hdu_list[0].data.shape[0],hdu_list[0].data.shape[1]))
    else:
        Source = glob.glob('KiDSImages/*fits')
        hdu_list = fits.open(Source[0])
        Images = np.zeros((Output.shape[0],hdu_list[0].data.shape[0],hdu_list[0].data.shape[1],hdu_list[0].data.shape[2]))
    
    for a in xrange(0,Output.shape[0]):
        #Open the FITS file
        hdu_list = fits.open(str(Source[a][:-11])+str(int(Output[a,0]))+'.fits')
        #Extract image data
        Images[a,:,:,:] = hdu_list[0].data
        Images[a,:,:,:] = Images[a,:,:,:]/np.amax(Images[a,:,:,:])
        #Close the FITS file
        hdu_list.close()
        
    return Images


def LoadModel(Directory,Image):
    
    from DefaultModel import CNN
    
    if len(Image.shape) < 3:
        Bands = 1
        Size = Image.shape
    elif Image.shape[0] < Image.shape[2]:
        Bands = Image.shape[0]
        Size = [Image.shape[1],Image.shape[2]]
    else:
        Bands = Image.shape[2]
        Size = [Image.shape[0],Image.shape[1]]
    
    Model = CNN(Bands,Size,Directory)
    
    return Model

def LoadData(Number,Type,COSMOS):

    import re
    import numpy as np
    
    if Type == 'E':
        f = open('EuclidDataFile.csv','r')
    else:
        f = open('KiDSDataFile.csv','r')
    reader = f.read()
    f.close()
    
    reader = re.split('\n',reader)
    Index = int(re.split(',',reader[0])[-1])
    reader = reader[1:-1]
    
    if COSMOS != 'N':
        
        if Type == 'E':
            f = open('KiDSListCOSMOSCut.csv','r')
        else:
            f = open('EuclidListCOSMOSCut.csv','r')
        readerC = f.read()
        f.close()
        
        COSMOSCut = []
        
        for a in xrange(0,len(readerC)/7):
            COSMOSCut.append(readerC[7*a:(7*a)+6])
         
        Output = np.zeros((Number,2))
            
        for a in xrange(0,Number):
            Void = re.split(',',reader[int(int(COSMOSCut[a])-100000)])
            Output[a,:] = [int(Void[0]),1-int(Void[Index])]
    
    else:
        
        Output = np.zeros((Number,2))
    
        for a in xrange(0,Number):
            Void = re.split(',',reader[a])
            Output[a,:] = [int(Void[0]),1-int(Void[Index])]
    
    return Output