#!/usr/bin/env python

"""Run a single simulation rep.

This script is meant to be called within a SLURM submission script.
"""

from typing import Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import sys
import toytree
import toyplot
import ipcoal
import numpy as np
import pandas as pd


# get an ultrametric imbalanced tree
BAL_EDGES = [0, 1, 8, 10, 6, 7, 12, 13, 14]
IMB_EDGES = [3, 4, 10, 11]
NE_DEFAULT = 2e5
GT_DEFAULT = 1
RNG = np.random.default_rng(123)
NLOCI = 1000
NREPS = 100
LOCUS_LENS = [1e6, 1e5, 1e4, 1e3]


def setup_tree(
    tree_type: str,
    parameter: str,
) -> toytree.ToyTree:
    """Return a species tree with parameters set.

    """
    assert tree_type in ("bal", "imb")
    assert parameter in ("Ne", "gt")

    # set parameters on the species tree
    if tree_type == "bal":
        tree = toytree.rtree.baltree(8, treeheight=6e5)
        edges = BAL_EDGES
    else:
        tree = toytree.rtree.imbtree(8, treeheight=6e5)
        edges = IMB_EDGES

    if parameter == "Ne":
        tree = tree.set_node_data("Ne", {i: NE_DEFAULT * 20 for i in edges}, default=NE_DEFAULT)
        tree = tree.set_node_data("gt", default=GT_DEFAULT)
    else:
        tree = tree.set_node_data("Ne", default=NE_DEFAULT)
        tree = tree.set_node_data("gt", {i: GT_DEFAULT * 20 for i in edges}, default=GT_DEFAULT)

    tree = tree.set_node_data("tg", {i: i.dist / i.gt for i in tree})
    tree = tree.set_node_data("tc", {i: i.tg / (2 * i.Ne) for i in tree})
    tree = tree.set_node_data("theta", {i: 4 * i.Ne * 1e-8 for i in tree})
    tree = tree.set_node_data("rho", {i: 4 * i.Ne * 1e-9 for i in tree})
    tree = tree.set_node_data("tg_rho", {i: i.tg * i.rho for i in tree})
    tree = tree.set_node_data("tg_theta", {i: i.tg * i.theta * 1e-9 for i in tree})

    # convert edge lens to units of generations.
    tree = tree.set_node_data("dist", {i: i.tg for i in tree})
    return tree


def get_n_topos(model_df: pd.DataFrame) -> float:
    ntopos = []
    for _, locus in model_df.groupby("locus"):
        mtree = toytree.mtree(locus.genealogy)
        ntopos.append(len(mtree.get_unique_topologies()))
    return np.mean(ntopos)


def iter_first_genealogies(model_df: pd.DataFrame):
    for _, df in model_df.groupby("locus"):
        yield toytree.tree(df.iloc[0, 6])


def one_batch_sim(tree, nloci, nsites, nthreads, seed):
    """Return a list of ToyTrees as inferred raxml-ng gene trees."""
    model = ipcoal.Model(tree=tree, seed_trees=seed, seed_mutations=seed)
    model.sim_loci(nloci, nsites)
    raxdf = ipcoal.phylo.infer_raxml_ng_trees(model, nthreads=nthreads, nproc=1, nworkers=1, do_not_autoscale_threads=True)
    return model.df, raxdf


def batch_sims(tree: toytree.ToyTree, nloci: int = 1000, nsites: int = 1e4, njobs: int = 10, nthreads: int = 4):    

    if nloci == 1:
        return one_batch_sim(tree, nloci, nsites, nthreads, None)

    # not perfect
    nloci_per = int(nloci / njobs)
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=njobs) as pool:
        for i in range(njobs):
            rasyncs[i] = pool.submit(one_batch_sim, *(tree, nloci_per, nsites, nthreads, None))
    gdata = []
    rdata = []
    for i in range(njobs):
        gdf, rdf = rasyncs[i].result()
        gdf.locus += i * nloci_per
        rdf.locus += i * nloci_per
        gdata.append(gdf)
        rdata.append(rdf)
    gdata = pd.concat(gdata, ignore_index=True)
    rdata = pd.concat(rdata, ignore_index=True)
    return gdata, rdata


def sim_and_infer_one_rep(
    species_tree: toytree.ToyTree,
    nloci: int,
    nsites: int,
    rep: int,
    seed: int,
    tmpdir: Path,
    njobs: int,
    nthreads: int,
) -> None:
    """...

    """
    # set up model and simulate loci
    # model = ipcoal.Model(species_tree, seed_mutations=seed, seed_trees=seed)

    # batch simulate loci
    # model.sim_loci(nloci=nloci, nsites=nsites)
    simdf, raxdf = batch_sims(species_tree, nloci, nsites, njobs, nthreads)

    # get distribution of true genealogies
    gtrees = list(iter_first_genealogies(simdf))

    # get distribution of inferred gene trees
    raxtrees = raxdf.gene_tree
    # raxtrees = ipcoal.phylo.infer_raxml_ng_trees(
    #     model, nproc=njobs, nthreads=nthreads, nworkers=1, tmpdir=tmpdir)
    # raxtrees = raxtrees.gene_tree
    # raxtrees = [ipcoal.phylo.infer_raxml_ng_tree(model, idxs=i, nthreads=nthreads, nworkers=1, tmpdir=tmpdir) for i in range(nloci)]

    # single tree is the result
    if nloci == 1:
        raxtree = toytree.tree(raxtrees[0])
        print(
            nloci, nsites, rep,
            raxdf.nsnps.mean(),
            get_n_topos(simdf),
            1,
            "",
            0,
            0,
            raxtree.write(),
            raxtree.distance.get_treedist_rfg_mci(species_tree),
            raxtree.distance.get_treedist_quartets(species_tree).similarity_to_reference,
        )
        return

    # get astral tree inferred from genealogies
    atree_true = ipcoal.phylo.infer_astral_tree(gtrees, tmpdir=tmpdir)

    # get astral tree inferred from gene trees
    atree_empirical = ipcoal.phylo.infer_astral_tree(raxtrees, tmpdir=tmpdir)  # .gene_tree)

    # get distances from true species tree
    true_dist_rf = species_tree.distance.get_treedist_rfg_mci(atree_true, normalize=True)
    true_dist_qrt = species_tree.distance.get_treedist_quartets(atree_true).similarity_to_reference

    # get distances from true species tree
    emp_dist_rf = species_tree.distance.get_treedist_rfg_mci(atree_empirical, normalize=True)
    emp_dist_qrt = species_tree.distance.get_treedist_quartets(atree_empirical).similarity_to_reference

    # get mean topologies per locus in true genealogies
    ntopos_true = get_n_topos(simdf)

    # get number of topologies in empirical gene trees
    ntopos_inferred = len(toytree.mtree(raxtrees).get_unique_topologies())

    # infer a SNAQ net1...


    # store data
    print(
        nloci, nsites, rep,
        raxdf.nsnps.mean(), ntopos_true, ntopos_inferred,
        atree_true.write(), true_dist_rf, true_dist_qrt,
        atree_empirical.write(), emp_dist_rf, emp_dist_qrt,
    )


def single_command_line_parser() -> Dict[str, Any]:
    """..."""
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--tree', type=str, default="bal", help='bal or imb')
    parser.add_argument(
        '--parameter', type=str, default="Ne", help='Ne or gt')
    parser.add_argument(
        '--nsites', type=float, default=1e4, help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=int, default=10, help='Number of independent loci to simulate')
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
    return vars(parser.parse_args())


def test_sim():
    tree = toytree.rtree.imbtree(8, treeheight=2e6)
    tree.set_node_data("Ne", default=1e5, inplace=True)
    return batch_sims(tree, nloci=100, nsites=1000, njobs=5, nthreads=2)


def test_sim_and_infer():
    gdf, rdf = test_sim()
    atree_true = ipcoal.phylo.infer_astral_tree(gdf.genealogy)
    atree_emp = ipcoal.phylo.infer_astral_tree(rdf.gene_tree)
    return atree_true, atree_emp


def test():
    tree = toytree.rtree.imbtree(8, treeheight=2e6)
    tree.set_node_data("Ne", default=1e5, inplace=True)
    sim_and_infer_one_rep(tree, nloci=10, nsites=1e4, rep=0, seed=123, tmpdir=Path("/tmp"), njobs=4, nthreads=2)


if __name__ == "__main__":

    # print(test_sim_and_infer())
    # test()

    kwargs = single_command_line_parser()
    species_tree = setup_tree(kwargs["tree"], kwargs["parameter"])

    outdir = Path(kwargs["outdir"])
    tmpdir = outdir / f"tmp-{kwargs['parameter']}-{int(kwargs['nsites'])}-{kwargs['rep']}"
    tmpdir.mkdir(exist_ok=True)

    sim_and_infer_one_rep(
        species_tree=species_tree,
        seed=kwargs["seed"],
        nsites=kwargs["nsites"],
        nloci=kwargs["nloci"],
        rep=kwargs["rep"],
        tmpdir=tmpdir,
        njobs=kwargs["njobs"],
        nthreads=kwargs["nthreads"]
    )

    tmpdir.rmdir()
