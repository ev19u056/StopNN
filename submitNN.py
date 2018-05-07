if __name__ == "__main__":
    import os
    import subprocess
    import argparse
    import sys
    import localConfig as cfg
    from commonFunctions import assure_path_exists
    import datetime

    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-l', '--layers', type=int, required=True, help='Number of layers')
    parser.add_argument('-n', '--neurons', type=int, required=True, help='Number of neurons per layer')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-a', '--batchSize', type=int, required=True, help='Batch size')
    parser.add_argument('-b', '--learningRate', type=float, required=True, help='Learning rate')
    parser.add_argument('-c', '--decay', type=float, default=0, help='Learning rate decay')
    parser.add_argument('-d', '--dropoutRate', type=float, default=0, help='Drop-out rate')
    parser.add_argument('-r', '--regularizer', type=float, default=0, help='Regularizer')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration number i')

    args = parser.parse_args()

    n_layers = args.layers
    n_neurons = args.neurons
    n_epochs = args.epochs
    batch_size = args.batchSize #len(XDev)/100
    learning_rate = args.learningRate
    my_decay = args.decay
    dropout_rate = args.dropoutRate
    regularizer = args.regularizer
    iteration = args.iteration

    dateSubmission = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    baseName = cfg.lgbk+"SingleNN/" + dateSubmission +"/"
    assure_path_exists(baseName+"dummy.txt")

    for i in range(1,iteration+1):
        shPath=baseName+'trainNN_Ver'+str(i)+'.sh'
        with open(shPath, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#$ -cwd\n")
            f.write("#$ -pe mcore 3\n")
            f.write("#$ -l container=True\n")
            f.write("#$ -v CONTAINER=CENTOS7\n")
            f.write("#...$ -v CONTAINER=UBUNTU16\n")
            f.write("#$ -l gpu,release=el7\n")
            f.write("cd /exper-sw/cmst3/cmssw/users/dbastos/StopNN/\n")
            f.write("module load root-6.10.02\n")
            f.write("python trainNN.py -z -l " + str(n_layers) + " -n " + str(n_neurons) + " -e " + str(n_epochs) + " -a " + str(batch_size) + " -b " + str(learning_rate) + " -c " + str(my_decay) + " -d " + str(dropout_rate) + " -r " + str(regularizer) + " -i " + str(i) + "\n")

            mode = os.fstat(f.fileno()).st_mode
            mode |= 0o111
            os.fchmod(f.fileno(), mode & 0o7777)

        #submissionCmd = "qsub -e " + baseName + "log.err -o "+ baseName + "log.out " + shPath
        submissionCmd = "qsub " + shPath + " -e "+ baseName + " -o " + baseName
        p = subprocess.Popen(submissionCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
