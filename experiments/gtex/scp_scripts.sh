#!/bin/sh

scp ./gtex_find_pairwise_as.py aj13@della.princeton.edu:/scratch/gpfs/aj13/multi-group-GP/experiments/gtex
scp ./job.slurm* aj13@della.princeton.edu:/scratch/gpfs/aj13/multi-group-GP/experiments/gtex

scp ../../models/gaussian_process.py aj13@della.princeton.edu:/scratch/gpfs/aj13/multi-group-GP/models

scp ../../kernels/kernels.py aj13@della.princeton.edu:/scratch/gpfs/aj13/multi-group-GP/kernels


echo "Done."