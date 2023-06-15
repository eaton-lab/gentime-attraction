#!/usr/bin/env python

"""Run a single simulation rep.

This script is meant to be called within a SLURM submission script.
"""

from typing import Dict, Any
from pathlib import Path
import argparse
import sys
import toytree
import toyplot
import ipcoal
import numpy as np
import pandas as pd


# get an ultrametric imbalanced tree
EDGES = [0, 1, 8, 10, 6, 7, 12, 13, 14]
NE_DEFAULT = 1e5
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
        tree = toytree.rtree.baltree(8, treeheight=1.5e5)
    else:
        tree = toytree.rtree.imbtree(8, treeheight=1.5e5)

    if parameter == "Ne":
        tree = tree.set_node_data("Ne", {i: NE_DEFAULT * 10 for i in EDGES}, default=NE_DEFAULT)
        tree = tree.set_node_data("gt", default=GT_DEFAULT)
    else:
        tree = tree.set_node_data("Ne", default=NE_DEFAULT)
        tree = tree.set_node_data("gt", {i: GT_DEFAULT * 10 for i in EDGES}, default=GT_DEFAULT)

    tree = tree.set_node_data("tg", {i: i.dist / i.gt for i in tree})
    tree = tree.set_node_data("tc", {i: i.tg / (2 * i.Ne) for i in tree})
    tree = tree.set_node_data("theta", {i: 4 * i.Ne * 1e-8 for i in tree})
    tree = tree.set_node_data("rho", {i: 4 * i.Ne * 1e-9 for i in tree})
    tree = tree.set_node_data("tg_rho", {i: i.tg * i.rho for i in tree})
    tree = tree.set_node_data("tg_theta", {i: i.tg * i.theta * 1e-9 for i in tree})

    # convert edge lens to units of generations.
    tree = tree.set_node_data("dist", {i: i.tg for i in tree})
    return tree


def get_n_topos(model: ipcoal.Model) -> float:
    ntopos = []
    for _, locus in model.df.groupby("locus"):
        mtree = toytree.mtree(locus.genealogy)
        ntopos.append(len(mtree.get_unique_topologies()))
    return np.mean(ntopos)


def iter_first_genealogies(model: ipcoal.Model):
    for _, df in model.df.groupby("locus"):
        yield toytree.tree(df.iloc[0, 6])


def sim_and_infer_one_rep(
    species_tree: toytree.ToyTree,
    nsites: int,
    nloci: int,
    rep: int,
    seed: int,
    tmpdir: Path,
    njobs: int,
    nthreads: int,
) -> None:
    """...

    """
    # set up model and simulate loci
    model = ipcoal.Model(species_tree, seed_mutations=seed, seed_trees=seed)
    model.sim_loci(nloci=nloci, nsites=nsites)

    # get distribution of true genealogies
    gtrees = list(iter_first_genealogies(model))

    # get distribution of inferred gene trees
    # raxtrees = ipcoal.phylo.infer_raxml_ng_trees(model, nproc=njobs, nthreads=nthreads, nworkers=1, tmpdir=tmpdir)
    raxtrees = [ipcoal.phylo.infer_raxml_ng_tree(model, idxs=i, nthreads=nthreads, nworkers=1, tmpdir=tmpdir) for i in range(nloci)]
    # raxtrees = raxtrees.gene_tree

    # get astral tree inferred from genealogies
    atree_true = ipcoal.phylo.infer_astral_tree(gtrees)

    # get astral tree inferred from gene trees
    atree_empirical = ipcoal.phylo.infer_astral_tree(raxtrees)  # .gene_tree)

    # get distances from true species tree
    true_dist_rf = species_tree.distance.get_treedist_rfg_mci(atree_true, normalize=True)
    true_dist_qrt = species_tree.distance.get_treedist_quartets(atree_true).similarity_to_reference

    # get distances from true species tree
    emp_dist_rf = species_tree.distance.get_treedist_rfg_mci(atree_empirical, normalize=True)
    emp_dist_qrt = species_tree.distance.get_treedist_quartets(atree_empirical).similarity_to_reference

    # get mean topologies per locus in true genealogies
    ntopos_true = get_n_topos(model)

    # get number of topologies in empirical gene trees
    ntopos_inferred = len(toytree.mtree(raxtrees).get_unique_topologies())

    # store data
    print(
        nloci, nsites, rep,
        model.df.groupby("locus").nsnps.mean().mean(), ntopos_true, ntopos_inferred,
        atree_true.write(), true_dist_rf, true_dist_qrt,
        atree_empirical.write(), emp_dist_rf, emp_dist_qrt,
    )


def single_command_line_parser() -> Dict[str, Any]:
    """..."""
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--tree-type', type=str, default="bal", help='bal or imb')
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


if __name__ == "__main__":

    kwargs = single_command_line_parser()
    species_tree = setup_tree(kwargs["tree_type"], kwargs["parameter"])

    outdir = Path(kwargs["outdir"])
    tmpdir = (outdir / f"tmp{kwargs['rep']}")
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
