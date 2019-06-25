import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import localConfig as cfg

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Process the command line options')
#   parser.add_argument('-c', '--configFile', required=True, help='Configuration file describing the neural network topology and options as well as the samples to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-r', '--runNum', type=int, help='Run number')
    parser.add_argument('-l', '--learningRate', type=float, help='Learning rate')
    parser.add_argument('-d', '--decay', type=float, help='Learning rate decay')
    parser.add_argument('-f', '--file', help='File name')


    args = parser.parse_args()

    test_point = "550_520"

    if args.file != None:
        model_name = args.file
        #learning_rate = str(float(model_name[model_name.find("Lr")+2:model_name.find("_D")]))
        #my_decay = str(float(model_name[model_name.find("_D")+2:]))
        filepath = cfg.lgbk+"Searches/"+model_name
        os.chdir(filepath)
        name = "ROC_"+model_name
        #name = "mGS:outputs_run_"+test_point+"_"+learning_rate+"_"+my_decay
    else:
        runNum = args.runNum
        learning_rate = args.learningRate
        my_decay = args.decay
        filepath = cfg.lgbk+"Searches/run"+str(runNum)
        os.chdir(filepath)
        name = "mGS:outputs_run"+str(runNum)+"_"+test_point+"_"+str(learning_rate)+"_"+str(my_decay)


    #f = open(filepath+name+'.txt', 'r')
    # Until run5
    #name = "mGS:outputs_run"+str(runNum)+"_"+test_point

    f = open(name + '.txt', 'r')

    layer = []
    neurons = []
    roc_AUC = []
    ks = []
    ks_s = []
    ks_b = []
    FOM = []
    line_index=0

    for line in f:
        if line_index%7==0:
            layer.append(float(line,))
        if line_index%7==1:
            neurons.append(float(line,))
        if line_index%7==2:
            roc_AUC.append(float(line,))
        if line_index%7==3:
            ks_s.append(float(line,))
        if line_index%7==4:
            ks_b.append(float(line,))
        if line_index%7==5:
            ks.append(float(line,))
        if line_index%7==6:
            FOM.append(float(line,))
        line_index=line_index+1

    layers_legend = ["1 layer","2 layers", "3 layers"]
    nLayers = len(layers_legend)

    plt.figure(figsize=(7,6))
    plt.xlabel("Number of Neurons")
    plt.ylabel('Roc AUC')
    plt.suptitle("Roc curve integral for several configurations of Neural Nets", fontsize=13, fontweight='bold')
    #plt.title("Learning rate: {0}\nDecay: {1}".format(learning_rate, my_decay), fontsize=10)

    lim = len(neurons)/nLayers

    for i in range(0,nLayers):
        plt.plot(neurons[i*lim:(i+1)*lim], roc_AUC[i*lim:(i+1)*lim])

    plt.legend(layers_legend, loc='best')
    if args.file != None:
#        plt.savefig(name+'.pdf')
        plt.savefig('ROC_'+model_name+'.pdf')
    else:
        plt.savefig("ROC_run"+str(runNum)+"_"+str(test_point)+"_"+str(learning_rate)+"_"+str(my_decay)+".png")
    plt.show()
