#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q long
#$ -N evt_vgg_long_tail_refined

# Required modules
module load conda
conda init bash
source activate face_morph2

python modeling.py
