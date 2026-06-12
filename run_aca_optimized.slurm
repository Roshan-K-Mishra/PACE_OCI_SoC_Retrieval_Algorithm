#!/bin/bash
#SBATCH --job-name=cs_pace
#SBATCH --output=aca_optimized_%J.out
#SBATCH --error=aca_optimized_%J.err
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cluster=chip-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=04:00:00                 # Time limit 
#SBATCH --account=pi_zzbatmos
#SBATCH --mail-type=ALL                 #Send email on all job events
#SBATCH --mail-user=kn82613@umbc.edu    #Send all emails to email_address

# Go to your working directory
cd /umbc/rs/pi_zzbatmos/users/kn82613/My_WORK/PACA_SoC_Algorithm/V1

# Activate your environment
source activate ACA_algorithm

# Make sure the retrieval code finds the model and normalization files
export REPO_DIR=/umbc/rs/pi_zzbatmos/users/kn82613/My_WORK/PACA_SoC_Algorithm/V1

# Optional: print diagnostic information
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Step 1: create eval_inputs.npz from PACE_SoC_retrieval.nc
python make_inputs.py . eval_inputs_real.npz

# Step 2: run optimized retrieval
python run_optimized.py eval_inputs_real.npz 0 -1 optimized_real_out.npz 4096

echo "Done."