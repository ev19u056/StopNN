"""
Make a Grid Search varying the number of neurons per layer and the number of layers.
Plot the results to using the area under the ROC curve as a metric of performance.
"""


import root_numpy
import os
import numpy as np
import pandas
import keras
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import localConfig as cfg
from commonFunctions import StopDataLoader, FullFOM, getYields

import sys
from math import log

from prepareDATA import *

compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
trainParams = {'epochs': 300, 'batch_size': 30000, 'verbose': 1}
learning_rate = 0.003
myAdam = Adam(lr=learning_rate)
compileArgs['optimizer'] = myAdam

print "Opening file"

runNum = 1
name = "mGS:outputs_run"+str(runNum)+"_"
filepath = cfg.lgbk+"Searches/"+name

f = open(filepath+test_point+'.txt', 'w')

def getDefinedClassifier(nIn, nOut, compileArgs, neurons=12, layers=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
    for x in range(0, layers-1):
        model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(**compileArgs)
    return model

for y in [1,2,3]:   # LAYERS
    for x in range(15, 20):    # NEURONS
        print "  ==> #LAYERS:", y, "   #NEURONS:", x, " <=="

        print("Starting the training")
        model = getDefinedClassifier(len(trainFeatures), 1, compileArgs, x, y)
        history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)
        name = "myNN_run13_L"+str(y)+"_N"+str(x)+"_"+train_DM
        model.save(name+".h5")
        model_json = model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights(name + ".h5")

        print("Getting predictions")
        devPredict = model.predict(XDev)
        valPredict = model.predict(XVal)

        print("Getting scores")

        scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 1)
        scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 1)
        cohen_kappa=cohen_kappa_score(YVal, valPredict.round())

        print "Calculating FOM:"
        dataDev["NN"] = devPredict
        dataVal["NN"] = valPredict

        sig_dataDev=dataDev[dataDev.category==1]
        bkg_dataDev=dataDev[dataDev.category==0]
        sig_dataVal=dataVal[dataVal.category==1]
        bkg_dataVal=dataVal[dataVal.category==0]


        tmpSig, tmpBkg = getYields(dataVal)
        sigYield, sigYieldUnc = tmpSig
        bkgYield, bkgYieldUnc = tmpBkg

        fomEvo = []
        fomCut = []

        bkgEff = []
        sigEff = []

        sig_Init = dataVal[dataVal.category == 1].weight.sum() * 35866 * 2
        bkg_Init = dataVal[dataVal.category == 0].weight.sum() * 35866 * 2

        for cut in np.arange(0.0, 0.9999, 0.001):
            sig, bkg = getYields(dataVal, cut=cut, luminosity=luminosity)
            if sig[0] > 0 and bkg[0] > 0:
                fom, fomUnc = FullFOM(sig, bkg)
                fomEvo.append(fom)
                fomCut.append(cut)
                bkgEff.append(bkg[0]/bkg_Init)
                sigEff.append(sig[0]/sig_Init)

        max_FOM=0

        print "Maximizing FOM"
        for cv_0 in fomEvo:
            if cv_0>max_FOM:
                max_FOM=cv_0

        roc_Integral = 0
        for cv_1 in range(0, len(bkgEff)-1):
            roc_Integral=roc_Integral+0.5*(bkgEff[cv_1]-bkgEff[cv_1+1])*(sigEff[cv_1+1]+sigEff[cv_1])

        Eff = zip(bkgEff, sigEff)

        km_value_s = ks_2samp(sig_dataDev["NN"], sig_dataVal["NN"])[1]
        km_value_b = ks_2samp(bkg_dataDev["NN"], bkg_dataVal["NN"])[1]
        km_value = ks_2samp(dataDev["NN"], dataVal["NN"])[1]

        f.write(str(y)+"\n")
        f.write(str(x)+"\n")
        f.write(str(roc_Integral)+"\n")
        f.write(str(km_value_s)+"\n")
        f.write(str(km_value_b)+"\n")
        f.write(str(km_value)+"\n")
        f.write(str(max_FOM)+"\n")

        print "Plotting"

        plt.figure(figsize=(7,6))
        plt.hist(sig_dataDev["NN"], 50, facecolor='blue', alpha=0.7, normed=1, weights=sig_dataDev["weight"])
        plt.hist(bkg_dataDev["NN"], 50, facecolor='red', alpha=0.7, normed=1, weights=bkg_dataDev["weight"])
        plt.hist(sig_dataVal["NN"], 50, color='blue', alpha=1, normed=1, weights=sig_dataVal["weight"], histtype="step")
        plt.hist(bkg_dataVal["NN"], 50, color='red', alpha=1, normed=1, weights=bkg_dataVal["weight"], histtype="step")
        plt.xlabel('NN output')
        plt.suptitle("MVA overtraining check for classifier: NN", fontsize=13, fontweight='bold')
        plt.title("Roc Curve AUC: {0} \nKolmogorov Smirnov test (s,b,s+b): ({1}, {2}, {3})".format(roc_Integral, km_value_s, km_value_b, km_value), fontsize=10)
        plt.legend(['Signal (Test sample)', 'Background (Test sample)', 'Signal (Train sample)', 'Background (Train sample)'], loc='upper right')
        plt.savefig('hist_'+name+'.png', bbox_inches='tight')


        plt.figure(figsize=(7,6))
        plt.plot(bkgEff, sigEff)
        plt.title("Roc curve", fontweight='bold')
        plt.ylabel("Signal efficiency")
        plt.xlabel("Bakcground efficiency")
        plt.axis([0, 1, 0, 1])
        plt.legend(["Roc curve integral: {0}".format(roc_Integral)], loc='best')
        plt.savefig('Roc_'+name+'.png', bbox_inches='tight')


        plt.figure(figsize=(7,6))
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(211)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("evo_"+name+'.png')
        #plt.show()

sys.exit("Done!")
