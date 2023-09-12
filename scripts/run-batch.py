#!/usr/bin/env python

from pathlib import Path
import argparse
from subprocess import Popen, PIPE, STDOUT
import numpy as np


# OPTIMIZED FOR HPC THAT ALLOWS FAST JOBS WITH <= 12 CORES
SBATCH = """\
#!/bin/bash
#SBATCH --account=dsi
#SBATCH --job-name={jobname}
#SBATCH --output={outdir}/{jobname}.out
#SBATCH --error={outdir}/{jobname}.err
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
    --tmpdir {tmpdir} \
    --rep {rep} \
    --njobs {njobs} \
    --nthreads {nthreads}
"""


def write_and_submit_sbatch_script(
    rep: int,
    seed: int,
    tree: str,
    parameter: str,
    nsites: int,
    nloci: int,
    outdir: Path,
    tmpdir: Path,
    njobs: int,
    nthreads: int,
):
    """..
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    jobname = f"{tree}-{parameter}-{int(nsites)}-rep{rep}"
    jobpath = outdir / jobname

    # expand sbatch shell script with parameters
    sbatch = SBATCH.format(**dict(
        jobname=jobname,
        tree=tree,
        parameter=parameter,
        nsites=int(nsites),
        nloci=int(nloci),
        rep=rep,
        seed=seed,
        outdir=str(outdir),
        tmpdir=str(tmpdir),
        njobs=njobs,
        nthreads=nthreads,
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
        '--nloci', type=float, default=10, help='Number of independent loci to simulate')
    parser.add_argument(
        '--nreps', type=int, default=100, help="Number of reps")
    parser.add_argument(
        '--outdir', type=Path, default=".", help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--tmpdir', type=Path, default="/tmp", help='directory to write tmp files (e.g., /tmp)')
    parser.add_argument(
        '--njobs', type=int, default=1, help='N jobs to run concurrently')
    parser.add_argument(
        '--nthreads', type=int, default=4, help='N threads per job')
    parser.add_argument(
        '--seed', type=int, default=123, help="RNG seed. Used default as starting seed in manuscript.")

    return vars(parser.parse_args())


if __name__ == "__main__":

    params = single_command_line_parser()
    rng = np.random.default_rng(params["seed"])
    seeds = rng.integers(1e12, size=params["nreps"])
    outdir = Path(params["outdir"])
    outdir.mkdir(exist_ok=True)
    tmpdir = Path(params["tmpdir"])
    tmpdir.mkdir(exist_ok=True)

    for rep in range(params["nreps"]):

        # check if rep outfile exists
        jobname = Path(f"{params['tree']}-{params['parameter']}-{int(params['nsites'])}-rep{rep}")
        jobpath = outdir / jobname

        if jobpath.with_suffix(".out").exists():
            print(f"skipping {jobname}; .out file exists.")
            continue

        kwargs = dict(
            rep=rep,
            seed=seeds[rep],
            tree=params["tree"],
            parameter=params["parameter"],
            nsites=int(params["nsites"]),
            nloci=params["nloci"],
            outdir=params["outdir"],
            tmpdir=params["tmpdir"],
            njobs=params["njobs"],
            nthreads=params["nthreads"],
        )
        write_and_submit_sbatch_script(**kwargs)
