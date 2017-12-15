#!/bin/bash

#$ -pe mcore 3

#$ -l container=True
#...$ -v CONTAINER=CENTOS7
#$ -v CONTAINER=UBUNTU16

#$ -v SGEIN=script.py
#$ -v SGEIN=pima-indians-diabetes.data

#$ -v SGEOUT=accuracy.pickle
#$ -v SGEOUT=loss.pickle

module load root-6.10.02

python manualGridSearch.py
