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
    parser.add_argument('-f', '--iteration', type=int, default=0, help='Iteration number i')

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
        shPath=baseName+'trainNN_Ver'+str(iteration)+'.sh'
        with open(shPath, 'w') as f:
            f.write("#!/bin/bash")
            f.write("#$ -cwd")
            f.write("#$ -pe mcore 3")
            f.write("#$ -l container=True")
            f.write("#...$ -v CONTAINER=CENTOS7")
            f.write("#$ -v CONTAINER=UBUNTU16")
            f.write("#...$ -v SGEIN=script.py")
            f.write("#...$ -v SGEIN=pima-indians-diabetes.data")
            f.write("#...$ -v SGEOUT=accuracy.pickle")
            f.write("#...$ -v SGEOUT=loss.pickle")
            f.write("#$ -l gpu,release=el7")
            f.write("cd /exper-sw/cmst3/cmssw/users/dbastos/StopNN/")
            f.write("module load root-6.10.02")
            f.write("python trainNN.py -z -l n_layers -n n_neurons -e n_epochs -a batchSize -b learning_rate -c my_decay -d dropout_rate -r regularizer -e iteration")
            mode = os.fstat(f.fileno()).st_mode
            mode |= 0o111
            os.fchmod(f.fileno(), mode & 0o7777)

        submissionCmd = "qsub -e " + baseName + "log.err -o "+ baseName + "log.out " + shPath
        p = subprocess.Popen(submissionCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
