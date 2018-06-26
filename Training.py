#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import keras.callbacks
import numpy as np

# Add this so that the history of each epoch can be recorded
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.vlosses = []
        self.vaccuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.vlosses.append(logs.get('val_loss'))
        self.vaccuracy.append(logs.get('val_acc'))
        
def CheckCondition(Accuracy,Epoch,NumberOfEpochs):
    
    if (Epoch < 6):
        return 0
    elif (Accuracy[2][-1] > np.min(Accuracy[2])) and (Accuracy[2][-2] > np.min(Accuracy[2])) and (Accuracy[2][-3] > np.min(Accuracy[2])) and (NumberOfEpochs == 0):
        return 1
    elif (Epoch == int(NumberOfEpochs)):
        return 1
    else:
        return 0

def TrainNetwork(Model,TrainImages,ValidationImages,TrainData,ValidationData,NumberOfEpochs,Directory):
    
    Weights = []
    Output = []
    # Accuracy records train.losses,train.accuracy,v.losses,v.accuracy
    Accuracy = [[],[],[],[]]
    history = LossHistory()
    Model.fit(TrainImages, TrainData, validation_data=(ValidationImages, ValidationData), epochs=1, batch_size=250, verbose=1, callbacks=[history])
    Output.append(np.squeeze(Model.predict(TrainImages, verbose=0)))
    Weights.append(Model.get_weights())
    Accuracy[0].append(history.losses[0])
    Accuracy[1].append(history.accuracy[0])
    Accuracy[2].append(history.vlosses[0])
    Accuracy[3].append(history.vaccuracy[0])
    Epoch = 1
    Check = CheckCondition(Accuracy,Epoch,NumberOfEpochs)
    
    while (Check != 1): 
        #Model.set_weights(Weights[-1])
        Model.fit(TrainImages, TrainData, validation_data=(ValidationImages, ValidationData), epochs=1, batch_size=250, verbose=1, callbacks=[history])
        Output.append(np.squeeze(Model.predict(TrainImages, verbose=0)))
        Weights.append(Model.get_weights())
        Accuracy[0].append(history.losses[-1])
        Accuracy[1].append(history.accuracy[-1])
        Accuracy[2].append(history.vlosses[-1])
        Accuracy[3].append(history.vaccuracy[-1])
        Epoch = Epoch + 1
        Check = CheckCondition(Accuracy,Epoch,NumberOfEpochs)
        
    import matplotlib.pyplot as plt
    
    Time = np.arange(len(Accuracy[0]))+1
    plt.plot(Time,Accuracy[0][:],'r-',label='Log Loss')
    plt.plot(Time,Accuracy[1][:],'b-',label='Accuracy')
    plt.plot(Time,Accuracy[2][:],'k-',label='Validation Loss')
    plt.plot(Time,Accuracy[3][:],'m-',label='Validation Accuracy')
    plt.axis((0,len(Accuracy[0])+0.5,-0.1,1.1))
    plt.title('Accuracy and loss across epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Value')
    plt.legend(loc='upper left',prop={'size':6})
    plt.savefig('Output/'+Directory+'/TrainingSchedule.png')
    plt.show()
    plt.close()
    
    BestEpoch = np.argmin(Accuracy[2])
    Weights = Weights[BestEpoch]
    Output = Output[BestEpoch]
    Accuracy = [Accuracy[0][BestEpoch],Accuracy[1][BestEpoch],Accuracy[2][BestEpoch],Accuracy[3][BestEpoch]]
    
    Model.save_weights('Output/'+str(Directory)+'/OptimumWeights.h5')
    
    return BestEpoch, Weights, Output, Accuracy

def TrainStatistics():
    
    return