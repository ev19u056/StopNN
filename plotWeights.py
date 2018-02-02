import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import localConfig as cfg
from keras.models import model_from_json
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LinearSegmentedColormap

#runNum = 20
#model_name = "L1_N5_550_520_run20"
#filepath = cfg.lgbk + "Searches/run"+str(runNum)
model_name = "L2_N25_E300_Bs30000_Lr0.003_Dr0.0_TP550_520_full+pre"
filepath = "/home/diogo/PhD/SingleNN/"+model_name
os.chdir(filepath)

print "Loading Model ..."
with open(model_name+'.json', 'r') as json_file:
  loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_name+".h5")

cdict = {'red':   ((0.0, 0.97, 0.97),
                   (0.25, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.25, 0.25),
                   (0.25, 0.15, 0.15),
                   (0.75, 0.39, 0.39),
                   (1.0, 0.78, 0.78)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.25, 0.65, 0.65),
                   (0.75, 0.02, 0.02),
                   (1.0, 0.0, 0.0))
        }
myColor = LinearSegmentedColormap('myColorMap', cdict)

nLayers = 0
for layer in model.layers:
    if len(layer.get_weights()) == 0:
        continue
    nLayers+=1

maxWeights = 0

'''
plt.imshow(model.layers[0].get_weights()[0])
plt.show()
'''
figure = plt.figure()
figure.suptitle("Weights", fontsize=12)

i=1
nRow=2
nCol=3

if nLayers < 5:
    nRow = 2.0
    nCol = 2

elif nLayers < 10:
    nRow = math.ceil(nLayers / 3)
    nCol = 3

else:
    nRow = math.ceil(nLayers / 4)
    nCol = 4

for layer in model.layers:
    print "\n"
    print layer.name
    print layer.get_weights()
    #print "\nNew LAYER\n"
    #print layer.get_weights()
    #print "\n!"
    if len(layer.get_weights()) == 0:
        continue
    print layer.get_weights()[0]
    ax = figure.add_subplot(nRow, nCol,i)

    im = plt.imshow(layer.get_weights()[0], interpolation="none", vmin=-2, vmax=2, cmap=myColor)
    plt.title(layer.name, fontsize=10)
    plt.xlabel("Neuron", fontsize=9)
    plt.ylabel("Input", fontsize=9)
    plt.colorbar(im, use_gridspec=True)

    i+=1
plt.tight_layout()
plt.savefig('Weights_'+model_name+'.pdf', bbox_inches='tight')
plt.show()

"""
    if layer.get_weights() > maxWeights:
        maxWeights = layer.get_weights()
"""
