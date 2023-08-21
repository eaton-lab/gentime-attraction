#!/usr/bin/env python

"""Run a single simulation rep.

This script is meant to be called within a SLURM submission script.
"""

from typing import Dict, Any, Tuple, Iterator, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import subprocess
import toytree
import ipcoal
import numpy as np
import pandas as pd


# get an ultrametric imbalanced tree
# BAL_EDGES = [0, 1, 8, 10, 6, 7, 12, 13, 14]
BAL_EDGES = [2, 3, 6, 7, 9, 12]  # 10, 13
IMB_EDGES = [3, 4, 10, 11]
TREEHEIGHT = 1e6
NE_DEFAULT = 5e5
GT_DEFAULT = 1
MUT = 1e-8
RECOMB = 1e-9
RNG = np.random.default_rng(123)
NLOCI = 10
NREPS = 4
LOCUS_LENS = [1e6, 1e5, 1e4, 1e3]

COLNAMES = [
    "nloci",
    "nsites",
    "rep",
    "concat_tree_raxml",
    "prop_genealogies_concordant",
    "prop_raxtrees_concordant",
    "nsnps_per_locus",
    "mean_unique_topologies_per_locus",
    "n_unique_inferred_topologies",
    "astral_tree_genealogies",
    "dist_astral_tree_genealogies",
    "astral_tree_raxtrees",
    "dist_astral_tree_raxtrees",
    "snaq_net1_genealogies",
    "snaq_net1_genealogies_loglik_diff",
    "snaq_net1_raxtrees",
    "snaq_net1_raxtrees_loglik_diff",
]


def setup_tree(
    tree_type: str,
    parameter: str,
) -> toytree.ToyTree:
    """Return a species tree with Ne and GT parameters fixed or variable.
    """
    assert tree_type in ("bal", "imb")
    assert parameter in ("Ne", "gt")

    # set parameters on the species tree
    if tree_type == "bal":
        tree = toytree.rtree.baltree(8, treeheight=TREEHEIGHT)
        edges = BAL_EDGES
    else:
        tree = toytree.rtree.imbtree(8, treeheight=TREEHEIGHT)
        edges = IMB_EDGES

    if parameter == "Ne":
        tree = tree.set_node_data("Ne", {i: NE_DEFAULT * 20 for i in edges}, default=NE_DEFAULT)
        tree = tree.set_node_data("gt", default=GT_DEFAULT)
    else:
        tree = tree.set_node_data("Ne", default=NE_DEFAULT)
        tree = tree.set_node_data("gt", {i: GT_DEFAULT * 20 for i in edges}, default=GT_DEFAULT)

    tree = tree.set_node_data("tg", {i: i.dist / i.gt for i in tree})
    tree = tree.set_node_data("tc", {i: i.tg / (2 * i.Ne) for i in tree})
    tree = tree.set_node_data("theta", {i: 4 * i.Ne * MUT for i in tree})
    tree = tree.set_node_data("rho", {i: 4 * i.Ne * RECOMB for i in tree})
    tree = tree.set_node_data("tg_rho", {i: i.tg * i.rho for i in tree})
    tree = tree.set_node_data("tg_theta", {i: i.tg * i.theta * RECOMB for i in tree})

    # convert edge lens to units of generations.
    tree = tree.set_node_data("dist", {i: i.tg for i in tree})
    return tree


def get_n_topos(
    model_df: pd.DataFrame,
) -> float:
    """Return the mean number of topologies per locus in a simulated data"""
    ntopos = []
    for _, locus in model_df.groupby("locus"):
        mtree = toytree.mtree(locus.genealogy)
        ntopos.append(len(mtree.get_unique_topologies()))
    return np.mean(ntopos)


def iter_first_genealogies(
    model_df: pd.DataFrame,
) -> Iterator[toytree.ToyTree]:
    """Iterator to return only the first genealogy at each locus in a dataframe."""
    for _, df in model_df.groupby("locus"):
        yield toytree.tree(df.iloc[0, 6])


def one_batch_sim(
    tree: toytree.ToyTree,
    nloci: int,
    nsites: int,
    nthreads: int,
    seed: Optional[int],
    infer: bool,
    trim_returned_sequences_to: Optional[int] = None,
    tmpdir: Path = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return dataframes with true and (optionally) inferred gene trees.

    """
    # build a Model and simulate the trees and sequences.
    model = ipcoal.Model(
        tree=tree,
        seed_trees=seed,
        seed_mutations=seed,
        mut=MUT,
        recomb=RECOMB,
    )
    model.sim_loci(nloci, nsites)

    # infer a raxml tree from the sequences, or don't.
    if (nsites == 1) or (not infer):
        raxdf = None
    else:
        # serial execution fixed at 1 core because we parallelize externally
        raxdf = ipcoal.phylo.infer_raxml_ng_trees(
            model,
            nthreads=nthreads,
            nproc=1,
            nworkers=1,
            do_not_autoscale_threads=True,
            perf_threads=True,
            tmpdir=tmpdir,
        )

    # if returning the simulated sequences, they can be subset here.
    if trim_returned_sequences_to is not None:
        model.seqs = model.seqs[:, :, :trim_returned_sequences_to]

    # return true trees, simulated sequences, and inferred raxml trees
    return model.df, model.seqs, raxdf


def batch_sims(
    tree: toytree.ToyTree,
    nloci: int = 1000,
    nsites: int = 1e4,
    njobs: int = 10,
    nthreads: int = 4,
    seed: int = None,
    infer: bool = True,
    max_sites_per_locus_in_concat: int = 10_000,
    tmpdir: Path = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return a (genealogy, genetree) data frames.

    Nloci x nsites are simulated...
    """
    if nloci == 1:
        return one_batch_sim(tree, nloci, nsites, nthreads, seed=None, infer=infer)

    # not perfect
    nloci_per = int(nloci / njobs)
    rasyncs = {}
    seeds = np.random.default_rng(seed)
    with ProcessPoolExecutor(max_workers=njobs) as pool:
        for i in range(njobs):
            args = (
                tree,
                nloci_per,
                nsites,
                nthreads,
                seeds.integers(0, 9e12),
                infer,
                max_sites_per_locus_in_concat,
                tmpdir,
            )
            rasyncs[i] = pool.submit(one_batch_sim, *args)
    gdata = []
    sdata = []
    rdata = []

    # ...
    for i in range(njobs):
        gdf, seqs, rdf = rasyncs[i].result()
        gdf.locus += i * nloci_per
        gdata.append(gdf)
        sdata.append(seqs)
        if rdf is not None:
            rdf.locus += i * nloci_per
            rdata.append(rdf)

    # ...
    gdata = pd.concat(gdata, ignore_index=True)
    gseqs = np.concatenate(sdata, axis=2)

    # ...
    if rdata:
        rdata = pd.concat(rdata, ignore_index=True)
    return gdata, gseqs, rdata


def sim_and_infer_one_rep(
    species_tree: toytree.ToyTree,
    nloci: int,
    nsites: int,
    rep: int,
    seed: int,
    tmpdir: Path,
    njobs: int,
    nthreads: int,
    julia_path: Path,
) -> None:
    """...

    """
    # batch simulate true genealogies & sequences, and infer gene trees
    simdf, seqs, raxdf = batch_sims(
        species_tree, nloci, nsites, njobs, nthreads,
        infer=True, tmpdir=tmpdir,
    )

    # infer a concatenation tree from the SAME sequences (hack applies the
    # simulated sequences from last step onto a new Model object). However,
    # Some datasets are just way too big so the batch_sims step limits the
    # max size of returned matrix to be ...
    model = ipcoal.Model(species_tree)
    model.seqs = seqs
    concat_tree = ipcoal.phylo.infer_raxml_ng_tree(
        model,
        nworkers=1,
        nthreads=int(njobs * nthreads),
        seed=seed,
        do_not_autoscale_threads=True,
        perf_threads=True,
        tmpdir=tmpdir,
    )

    # get distribution of unlinked genealogies from simulation dataframe
    gtrees = list(iter_first_genealogies(simdf))

    # get distribution of inferred gene trees from raxml dataframe
    raxtrees = toytree.mtree(raxdf.gene_tree)

    # proportion true trees matching the species tree
    gtrees_concordant = np.array([species_tree.distance.get_treedist_rf(i) for i in gtrees])
    gtrees_concordant = np.sum(gtrees_concordant == 0) / gtrees_concordant.size
    # proportion rax trees matching the species tree
    raxtrees_concordant = np.array([species_tree.distance.get_treedist_rf(i) for i in raxtrees])
    raxtrees_concordant = np.sum(raxtrees_concordant == 0) / raxtrees_concordant.size
    # get mean topologies per locus in true genealogies
    ntopos_true = get_n_topos(simdf)
    # get number of topologies in empirical gene trees
    ntopos_inferred = len(raxtrees.get_unique_topologies())

    # infer astral tree inferred from true genealogies
    atree_true = ipcoal.phylo.infer_astral_tree(gtrees, tmpdir=tmpdir)

    # infer astral tree inferred raxml gene trees
    atree_empirical = ipcoal.phylo.infer_astral_tree(raxtrees, tmpdir=tmpdir)  # .gene_tree)

    # get distances from true species tree
    true_dist_rf = species_tree.distance.get_treedist_rf(atree_true)

    # get distances from true species tree
    emp_dist_rf = species_tree.distance.get_treedist_rf(atree_empirical)

    # infer a SNAQ net1 from the TRUE input genealogies
    ipcoal.phylo.infer_snaq_network(
        gtrees,
        tmpdir=tmpdir,
        name="sim",
        nedges=1,
        starting_tree=species_tree.write(None, None, None),
        binary_path=julia_path,
        nproc=int(njobs * nthreads),
        nreps=int(njobs * nthreads),
    )
    net0 = list(tmpdir.glob("analysis-snaq/sim-*.snaq.net-0.out"))[0]
    net1 = list(tmpdir.glob("analysis-snaq/sim-*.snaq.net-1.out"))[0]
    with open(net0, 'r') as netio:
        _, other = netio.readline().strip().split(";")
        net0_loglik_sim = other.split()[-1]
    with open(net1, 'r') as netio:
        net1_sim, other = netio.readline().strip().split(";")
        net1_loglik_sim = other.split()[-1]
    snaqdir = tmpdir / "analysis-snaq"
    for tmpfile in snaqdir.glob("*"):
        tmpfile.unlink()
    snaqdir.rmdir()

    # infer a SNAQ net1 from the INFERRED input gene trees
    ipcoal.phylo.infer_snaq_network(
        raxtrees,
        tmpdir=tmpdir,
        name="rax",
        nedges=1,
        starting_tree=species_tree.write(None, None, None),
        binary_path=julia_path,
        nproc=int(njobs * nthreads),
        nreps=int(njobs * nthreads),
    )
    net0 = list(tmpdir.glob("analysis-snaq/rax-*.snaq.net-0.out"))[0]
    net1 = list(tmpdir.glob("analysis-snaq/rax-*.snaq.net-1.out"))[0]
    with open(net0, 'r') as netio:
        _, other = netio.readline().strip().split(";")
        net0_loglik_rax = other.split()[-1]
    with open(net1, 'r') as netio:
        net1_rax, other = netio.readline().strip().split(";")
        net1_loglik_rax = other.split()[-1]
    snaqdir = tmpdir / "analysis-snaq"
    for tmpfile in snaqdir.glob("*"):
        tmpfile.unlink()
    snaqdir.rmdir()

    # store data
    data = [
        nloci,
        nsites,
        rep,
        concat_tree.write(),
        gtrees_concordant,
        raxtrees_concordant,
        raxdf.nsnps.mean(),
        ntopos_true,
        ntopos_inferred,
        atree_true.write(),
        true_dist_rf,
        atree_empirical.write(),
        emp_dist_rf,
        net1_sim + ";",
        float(net0_loglik_sim) - float(net1_loglik_sim),
        net1_rax + ";",
        float(net0_loglik_rax) - float(net1_loglik_rax),
    ]
    print("\t".join([str(i) for i in data]))
    for tmpfile in tmpdir.glob("*"):
        tmpfile.unlink()


def single_command_line_parser() -> Dict[str, Any]:
    """...

    python run-sim.py \
        --tree bal --parameter gt \
        --nsites 1e3 --nloci 1e3 \
        --rep 0 --seed 0 \
        --njobs 6 --nthreads 2 \
        --julia /home/deren/local/src/julia-1.6.2/bin/julia
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--tree', type=str, default="bal", help='bal or imb')
    parser.add_argument(
        '--parameter', type=str, default="Ne", help='Ne or gt')
    parser.add_argument(
        '--nsites', type=float, default=1e4, help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=float, default=10, help='Number of independent loci to simulate')
    parser.add_argument(
        '--rep', type=int, default=0, help='replicate id.')
    parser.add_argument(
        '--seed', type=int, default=123, help='random seed.')
    parser.add_argument(
        '--outdir', type=Path, default=".", help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--njobs', type=int, default=1, help='N jobs to run concurrently')
    parser.add_argument(
        '--nthreads', type=int, default=4, help='N threads per job')
    parser.add_argument(
        '--julia', type=Path, default=None, help='Path to julia')

    return vars(parser.parse_args())


#####################################################################
# TESTS
#####################################################################

def test_sim_and_infer_gtrees():
    """Test that returns a simple simulation"""
    tree = toytree.rtree.imbtree(8, treeheight=2e6)
    tree.set_node_data("Ne", default=1e5, inplace=True)
    return batch_sims(tree, nloci=10, nsites=1000, njobs=1, nthreads=4)


def test_sim_and_infer_gtrees_and_astral():
    gdf, rdf = test_sim_and_infer_gtrees()
    atree_true = ipcoal.phylo.infer_astral_tree(gdf.genealogy)
    atree_emp = ipcoal.phylo.infer_astral_tree(rdf.gene_tree)
    return gdf, rdf, atree_true, atree_emp


def test():
    outdir = Path("/tmp/table-test")
    outdir.mkdir(exist_ok=True)
    sim_and_infer_one_rep(
        species_tree=setup_tree("imb", "Ne"),
        nloci=10,
        nsites=1e4,
        rep=0,
        seed=123,
        tmpdir=outdir,
        njobs=4,
        nthreads=2,
        julia_path="julia",
    )


def setup_output_dir(**kwargs) -> Path:
    outdir = Path(kwargs["outdir"])
    tmpdir = outdir / f"tmp-{kwargs['parameter']}-{int(kwargs['nsites'])}-{kwargs['rep']}"
    tmpdir.mkdir(exist_ok=True)
    return tmpdir


if __name__ == "__main__":

    # print(test())
    # raise SystemExit(0)
    # print(test_sim_and_infer_gtrees())
    # raise SystemExit(0)

    kwargs = single_command_line_parser()
    species_tree = setup_tree(kwargs["tree"], kwargs["parameter"])
    tmpdir = setup_output_dir(**kwargs)

    julia_path = kwargs["julia"]
    if julia_path is None:
        proc = subprocess.run(['which', 'julia'], capture_output=True, check=True)
        julia_path = Path(proc.stdout.decode().strip())
    assert julia_path.exists(), f"Julia binary not found: {julia_path}"

    sim_and_infer_one_rep(
        species_tree=species_tree,
        seed=kwargs["seed"],
        nsites=kwargs["nsites"],
        nloci=kwargs["nloci"],
        rep=kwargs["rep"],
        tmpdir=tmpdir,
        njobs=kwargs["njobs"],
        nthreads=kwargs["nthreads"],
        julia_path=julia_path,
    )
    tmpdir.rmdir()
