#!/bin/bash
#MSUB -A p30190
#!/bin/bash
#SBATCH -A p30190             ## account
#SBATCH -p short             ## "-p" instead of "-q"
#SBATCH -N 10                 ## number of nodes
#SBATCH --ntasks-per-node 3  ## number of cores
#SBATCH --mem 20G
#SBATCH -t 01:00:00          ## walltime
#SBATCH	--job-name="py_ivf"    ## name of job

module purge all
module load anaconda3           ## Load modules (unchanged)

cd /projects/p30190/python-ivf

python -m scoop --hosts $(scontrol show hostnames $SLURM_JOB_NODELIST)  -n$SLURM_NTASKS main.py
exit
