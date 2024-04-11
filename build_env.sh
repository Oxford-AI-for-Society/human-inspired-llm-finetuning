!# /bin/bash

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/inference
conda create --prefix $CONPREFIX

# Activate your environment
source activate $CONPREFIX

# Install packages...
conda install -r requirements.txt
