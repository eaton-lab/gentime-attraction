#!/usr/bin/env python

from pathlib import Path
import argparse
from subprocess import Popen, PIPE, STDOUT
import numpy as np


SBATCH = """\
#!/bin/bash
#SBATCH --account=dsi
#SBATCH --job-name={jobname}
#SBATCH --output={jobname}.out
#SBATCH --error={jobname}.err
#SBATCH --time=11:59:00
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G

python run-sim.py \
    --tree {tree} \
    --parameter {parameter} \
    --nsites {nsites} \
    --nloci {nloci} \
    --seed {rep} \
    --outdir {outdir} \
    --rep {rep} \
    --nthreads 6
"""


def write_and_submit_sbatch_script(
    rep,
    seed,
    tree,
    parameter,
    nsites,
    nloci,
    outdir,
):
    """..
    """
    jobname = f"{tree}-{parameter}-{int(nsites)}-rep{rep}"
    jobpath = Path(outdir) / jobname

    # expand sbatch shell script with parameters
    sbatch = SBATCH.format(**dict(
        jobname=jobname,
        tree=tree,
        parameter=parameter,
        nsites=nsites,
        nloci=nloci,
        rep=rep,
        seed=seed,
        outdir=outdir,
    ))

    # b/c the params string name has a '.' in it for decimal ctime.

    tmpfile = jobpath.with_suffix('.sh')
    with open(tmpfile, 'w', encoding='utf-8') as out:
        out.write(sbatch)

    # submit job to HPC SLURM job manager
    cmd = ['sbatch', str(tmpfile)]
    with Popen(cmd, stdout=PIPE, stderr=STDOUT) as proc:
        out, _ = proc.communicate()
    tmpfile.unlink()

    # remove err file if no errors
    errfile = jobpath.with_suffix(".err")
    if errfile.exists():
        if not errfile.stat().st_size:
            errfile.unlink()


def single_command_line_parser():
    """..."""
    parser = argparse.ArgumentParser("...")
    parser.add_argument(
        '--tree', type=str, default="bal", help='bal or imb')
    parser.add_argument(
        '--parameter', type=str, default="Ne", help='Ne or gt')
    parser.add_argument(
        '--nsites', type=float, default=1e4, help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=int, default=10, help='Number of independent loci to simulate')
    parser.add_argument(
        '--nreps', type=int, default=100, help="Number of reps")
    parser.add_argument(
        '--outdir', type=Path, default=".", help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--njobs', type=int, default=1, help='N jobs to run concurrently')
    parser.add_argument(
        '--nthreads', type=int, default=4, help='N threads per job')
    parser.add_argument(
        '--seed', type=int, default=123, help="RNG seed")

    return vars(parser.parse_args())


if __name__ == "__main__":

    params = single_command_line_parser()
    rng = np.random.default_rng(params["seed"])
    seeds = rng.integers(1e12, size=params["nreps"])
    for rep in range(params["nreps"]):

        # check if rep outfile exists
        jobname = Path(f"{params['tree']}-{params['parameter']}-{int(params['nsites'])}-rep{rep}")
        if jobname.with_suffix(".out").exists():
            print(f"skipping {jobname}; already done.")
            continue

        kwargs = dict(
            rep=rep,
            seed=seeds[rep],
            tree=params["tree"],
            parameter=params["parameter"],
            nsites=int(params["nsites"]),
            nloci=params["nloci"],
            outdir=params["outdir"],
        )
        write_and_submit_sbatch_script(**kwargs)

