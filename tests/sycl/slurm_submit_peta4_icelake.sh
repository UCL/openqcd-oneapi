#!/bin/bash

#SBATCH -J cpu_sycl_job_openqcd
#SBATCH -A DIRAC-DR004-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --mail-type=NONE

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e 's/^\([0-9][0-9]*\).*$/\1/')

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh # Leave this line (enables the module command)
module purge                # Removes all modules still loaded
module load rhel8/slurm

#! Insert additional module load commands after this line if needed:
source /usr/local/software/intel/oneapi/2022.1/setvars.sh >/dev/null 2>&1

#! Full path to application executable:
application="./main.oneapi_intel_cpu"

#! Run options for the application:
options="16 16 16 16 ../../data/"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR" # The value of SLURM_SUBMIT_DIR sets workdir to the directory
# in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 76:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$((${numnodes} * ${mpi_tasks_per_node}))

CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"

if [ "$SLURM_JOB_NODELIST" ]; then
  #! Create a machine file:
  export NODEFILE=$(generate_pbs_nodefile)
  cat $NODEFILE | uniq >machine.file.$JOBID
  echo -e "\nNodes allocated:\n================"
  echo $(cat machine.file.$JOBID | sed -e 's/\..*$//g')
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

log_dir=logs_of_job_output_and_error
mkdir $log_dir 2>/dev/null
mv slurm*.out $log_dir/ 2>/dev/null

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

mv machine.file.* $log_dir/
