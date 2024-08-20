#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q long
#$ -N vgg-resnet

# Required modules
module load conda
conda init bash
source activate face_morph2

python modeling.py
