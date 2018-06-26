#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np


def Classify(Weights,Images,Model):
    
    Classes = np.zeros((Images.shape[0]))
    Model.set_weights(Weights)
    Classes[:] = np.squeeze(Model.predict(Images, verbose=0))
    
    return Classes

def ClassStatistics(Classes,Data,Threshold,Directory):
    
    # ID, Lens, Class, Correct
    Results = np.zeros((Data.shape[0],5))
    Results[:,:2] = Data[:,:2]
    Results[:,2] = abs(np.round(Classes[:]-(Threshold-0.5)))
    Results[:,3] = np.where(Results[:,1]==Results[:,2],1,0)
    Results[:,4] = Classes[:]
    
    np.savetxt('Output/'+Directory+'/Classifications.txt',Results[:,:],fmt='%06d %1d %1d %1d %0.4f',header='ID,Lens?,AboveThreshold?,Correct?,Classification')
    
    Total = Data.shape[0]
    Lenses = np.sum(Results[:,1])
    Correct = np.sum(Results[:,3])
    CorrectLenses = np.sum(np.multiply(Results[:,1],Results[:,3]))
    
    # Writes the statistics to a text file named ClassificationStatistics.txt
    
    print('Writing Classification Statistics to file')
    
    file = open('Output/'+Directory+'/ClassifcationStatistics.txt','w')
    file.write('Number of images:\t\t\t{0}\t\t{1}\n'.format(int(Total),'100.00%'))
    file.write('Number of lenses:\t\t\t{0}\t\t{1}\n'.format(int(Lenses),str(round(100*float(Lenses)/float(Total),2))+'%'))
    file.write('Correct classifications:\t\t{0}\t\t{1}\n'.format(int(Correct),str(round(100*float(Correct)/float(Total),2))+'%'))
    file.write('Correct Lens classifications:\t\t{0}\n'.format(int(CorrectLenses),str(round(100*float(CorrectLenses)/float(Lenses),2))+'%'))
    file.write('Correct Non-Lens classifications:\t\t{0}\n'.format(int(Correct-CorrectLenses),str(round(100*float(Correct-CorrectLenses)/float(Total-Lenses),2))+'%'))
    file.close()
    
    print('Classification Statistics written to: \n\tOutput/'+str(Directory)+'/ClassificationStatistics.txt')
    
    return