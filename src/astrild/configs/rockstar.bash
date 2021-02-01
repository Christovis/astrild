#!/bin/bash -l
#SBATCH --ntasks=1     # nr. of tasks
#SBATCH -t 0-00:00:00  # Runtime in D-HH:MM:SS
#SBATCH -J RockstarTemplate 
#SBATCH -o RockstarTemplate.out
#SBATCH -e RockstarTemplate.err
#SBATCH -p ----
#SBATCH -A ----
#SBATCH --exclusive

module purge
module load intel_comp/2018 intel_mpi/2018 hdf5/1.10.2 gsl/2.4 fftw/3.3.7

# Rockstar How To: https://github.com/yt-project/rockstar

cpunum=700
scale_factor=1
h0=0.6774
Ol=0.3089
Om=0.6911

dir_lc="/path/to/directory/containing/simulation/folders"
sim_roots=("simulation_folder_name")  # s
for index in ${!sim_roots[*]}; do
    sim_root=${sim_roots[$index]}
    dir_sim="${dir_lc}/${sim_root}"
    snap_nrs=$(find ${dir_sim}/ -name "snapdir_*" | wc -l)
    
    for _snap_nr in $(seq 1 $snap_nrs); do
        snap_nr=$(printf %03d $_snap_nr)
    
        dir_snap="${dir_sim}/snapdir_${snap_nr}"
        
        echo $dir_snap
        # Create output folder
        dir_rockstar="${dir_sim}/rockstar_${snap_nr}"
        if [[ ! -d ${dir_rockstar} ]]; then
            mkdir -p ${dir_rockstar}
        fi

        # Copy template to new config file
        newfile="iswrs_${sim_root}.cfg"
        if [[ ! -f ${newfile} ]]; then
            rm ${newfile}
            cp template_iswrs_${sim_root}.cfg ${newfile}
        else
            cp template_iswrs_${sim_root}.cfg ${newfile}
        fi
        
        # Edit new config file
        sed -i "19s@.*@h0=${h0}@" ${newfile}
        sed -i "20s@.*@Ol=${Ol}@" ${newfile}
        sed -i "21s@.*@Om=${Om}@" ${newfile}
        sed -i "33s@.*@INBASE='${dir_snap}'@" ${newfile}
        sed -i "34s@.*@OUTBASE='${dir_rockstar}'@" ${newfile}
        sed -i "35s@.*@FILENAME='snap_${snap_nr}.<block>'@" ${newfile}
       
        #./rockstar -c "${newfile}" &
        #auto_file="${dir_rockstar}/auto-rockstar.cfg"
        #while [[ ! -f ${auto_file} ]]; do sleep 2; done
        ./rockstar -c "${dir_rockstar}/auto-rockstar.cfg" &
        halo_file="${dir_rockstar}/out_0.list"
        while [[ ! -f ${halo_file} ]]; do sleep 2; done
    done
done
