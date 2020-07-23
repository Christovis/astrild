# prepare
module purge
module load intel_comp/2018-update2 intel_mpi/2018 fftw/2.1.5
unset I_MPI_HYDRA_BOOTSTRAP

# compile
#make clean && make

export OMP_NUM_THREADS=56
# Run the program
./powmes lc_box3_12.config

# report
echo ""
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AllocCPUS,Elapsed,ExitCode
exit
