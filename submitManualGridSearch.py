if __name__ == "__main__":
    import os
    import subprocess
    import argparse
    import sys
    import localConfig as cfg
    from commonFunctions import assure_path_exists

    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-l' ,'--scanLearningRate', action='store_true', help='Wether to scan learning rate')
    parser.add_argument('-d' ,'--scanLearningRateDecay', action='store_true', help='Wether to scan learning rate decay')
    parser.add_argument('-b' ,'--scanBatchSize', action='store_true', help='Wether to scan batch size')
    parser.add_argument('-e' ,'--scanEpochs', action='store_true', help='Wether to scan epochs')

    args = parser.parse_args()

    filepath = cfg.lgbk+"Searches/"

    lrArray = [0.003]
    dArray = [0]
    bsArray = [30000]
    eArray = [300]

    if args.scanEpochs:
        eArray = eArray + [500, 1000]
    if args.scanBatchSize:
        bsArray = bsArray + [300, 3000, 300000]
    if args.scanLearningRateDecay:
        dArray = dArray + [0.1]
    if args.scanLearningRate:
        lrArray = lrArray + [0.0001, 0.0003, 0.001, 0.01, 0.05, 0.1, 0.5, 5]

    for epoch in eArray:
        for bacthSize in bsArray:
            for decay in dArray:
                for learningRate in lrArray:
                    baseName = filepath + "E"+str(epoch)+"_Bs"+str(bacthSize)+"_Lr"+str(learningRate) + "_D" + str(decay) + "/"
                    assure_path_exists(baseName+"dummy.txt")
                    with open(baseName+'manualGridSearch.sh', 'w') as f:
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
                        f.write("python manualGridSearch.py + -lr " + str(learningRate) +" -d " + str(decay) + " -e " + str(epoch) + " -b "+ str(batchSize) + " -o " + baseName)

                        mode = os.fstat(f.fileno()).st_mode
                        mode |= 0o111
                        os.fchmod(f.fileno(), mode & 0o7777)

                    submissionCmd = "qsub " + baseName + "manualGridSearch.sh"
                    p = subprocess.Popen(submissionCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = p.communicate()
