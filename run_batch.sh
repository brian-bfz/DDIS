#!/bin/bash
###### 1. Slurm directives (edit as needed) ######
#SBATCH -A tensorlab
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -o slurm/log/%x-%j.out
#SBATCH -e slurm/log/%x-%j.err

###### 2. Pre‑run housekeeping ######
mkdir -p logs                          # create log dir if missing

# Print hostname and show GPU info
echo "Job launched on $(hostname) at $(date)"
nvidia-smi || echo "No GPUs visible"

###### 3. Module environment ######
module purge                           # start from a clean slate
module load gcc cuda                   # compiler & CUDA toolkit

###### 4. Python virtual‑env activation ######
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/resnick/groups/tensorlab/bzhao2/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/resnick/groups/tensorlab/bzhao2/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/resnick/groups/tensorlab/bzhao2/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/resnick/groups/tensorlab/bzhao2/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate edm_no

###### 5. Sanity check ######
python - <<'PY'
import jax, platform, os
print("Python :", platform.python_version())
print("JAX    :", jax.__version__)
print("JAX devices:", jax.devices())
if len(jax.devices()) > 0:
    print("Device :", jax.devices()[0])
PY

###### 6. Your workload ######
python scripts/train/train.py -c configs/training/pipe.yml --name initial