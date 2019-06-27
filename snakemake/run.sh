module load anaconda3
conda activate aspyre

snakemake --cluster "sbatch --cpus-per-task={cluster.n} \
                            --mem={cluster.memory} \
                            --gres=gpu:{cluster.gpu} \
                            --time={cluster.time} \
                            --job-name={cluster.jobname} \
                            --output=slurm_out/%x-%A \
                            --parsable " \
    --cluster-config 'cluster.yaml' \
    -w 120 \
    -j 250 \
    --max-jobs-per-second 1
