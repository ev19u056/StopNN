'''
Train the Neural Network
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.optimizers import Adam
import time
import keras
import pandas
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, AlphaDropout
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
from commonFunctions import getYields, FullFOM, myClassifier, gridClassifier
#from scipy.stats import ks_2samp
import localConfig as cfg
from prepareDATA import *


n_neurons = 25
n_layers = 2
n_epochs = 300
batch_size = 30000 #len(XDev)/10
learning_rate = 0.0003
dropout_rate = 0.0

compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': 1}
myAdam = Adam(lr=learning_rate)
compileArgs['optimizer'] = myAdam

name = "L"+str(n_layers)+"_N"+str(n_neurons)+"_E"+str(n_epochs)+"_Bs"+str(batch_size)+"_Lr"+str(learning_rate)+"_Dr"+str(dropout_rate)+"_TP"+test_point

filepath = cfg.lgbk+"SingleNN/"+name
os.mkdir(filepath)
os.chdir(filepath)
print("Dir "+filepath+" created.")
print("Starting the training")
start = time.time()
#call = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=5, verbose=1, mode='auto')
#model = getDefinedClassifier(len(trainFeatures), 1, compileArgs, n_neurons, n_layers, dropout_rate)
#model = myClassifier(len(trainFeatures),1, compileArgs, dropout_rate, learning_rate)
model = gridClassifier(nIn=len(trainFeatures),nOut=1, compileArgs=compileArgs,layers=n_layers,neurons=n_neurons,learn_rate=learning_rate,dropout_rate=dropout_rate)
history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev,shuffle=True, **trainParams)
print("Training took ", time.time()-start, " seconds")

# To save:
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


print "Calculating FOM:"
dataVal["NN"] = valPredict

tmpSig, tmpBkg = getYields(dataVal)
sigYield, sigYieldUnc = tmpSig
bkgYield, bkgYieldUnc = tmpBkg

sigDataVal = dataVal[dataVal.category==1]
bkgDataVal = dataVal[dataVal.category==0]

fomEvo = []
fomCut = []

for cut in np.arange(0.0, 0.9999999, 0.001):
  sig, bkg = getYields(dataVal, cut=cut)
  if sig[0] > 0 and bkg[0] > 0:
    fom, fomUnc = FullFOM(sig, bkg)
    fomEvo.append(fom)
    fomCut.append(cut)

max_FOM=0

print "Maximizing FOM"
for k in fomEvo:
  if k>max_FOM:
    max_FOM=k

print "Signal@Presel:", sigDataVal.weight.sum() * 35866 * 2
print "Background@Presel:", bkgDataVal.weight.sum() * 35866 * 2
print "Signal:", sigYield, "+-", sigYieldUnc
print "Background:", bkgYield, "+-", bkgYieldUnc

print "Maximized FOM:", max_FOM
print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]

import sys
#sys.exit("Done!")

#########################################################


# Let's repeat the above, but monitor the evolution of the loss function


#history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)

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
plt.savefig(name+'.png')
#plt.savefig('NN2_'+str(y)+''+str(x)+''+test_point+"_"+str(max_FOM)+'.png')

print "Model name: "+name

sys.exit("Done!")
