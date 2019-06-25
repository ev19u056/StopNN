#!/bin/bash

#$ -cwd
#$ -pe mcore 3

#$ -l container=True
#...$ -v CONTAINER=CENTOS7
#$ -v CONTAINER=UBUNTU16

#...$ -v SGEIN=script.py
#...$ -v SGEIN=pima-indians-diabetes.data

#...$ -v SGEOUT=accuracy.pickle
#...$ -v SGEOUT=loss.pickle

#$ -l gpu,release=el7

cd /exper-sw/cmst3/cmssw/users/dbastos/StopNN/

module load root-6.10.02

python manualGridSearch.py -r 0.003 -d 0 -e 10 -b 30000 -o test
