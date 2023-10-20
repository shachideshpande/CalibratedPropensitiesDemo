# Attempt to port https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/simdat.py to pytorch
import argparse
import random
from os import path as osp

import numpy as np
import numpy.random as npr
import pandas as pd
import torch
import yaml
import pdb
from attrdict import AttrDict
from sklearn.decomposition import PCA
from utils import (
    get_dataset_dir, get_raw_data_dir, get_sim_gwas_dataset_file,
    sim_genes_BN, sim_genes_HGDP, sim_genes_PSD, sim_genes_SP, sim_genes_TGP,
    sim_single_traits
)



def simulate_bolt_dataset(config):
    """
    Adapted from https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/simdat.py
    :param: conf: (attridict) Dictionary assumed to have the following parameters:
        simset: (str) Simulation set, i.e., one of: ['BN', 'TGP', 'HGDP', 'PSD', 'SP'].
        seed: (int) Random seed used.
        numsnps: (int) Number of causal SNPs.
        numunits: (int) Number of individuals.
        snpsig: (float) Control signal-to-noise ratio (SNR) for SNPs
        confint: (float) Control SNR for confounders
        causalprop: (float) Fraction of SNPs that are causal
        alpha: (float) Parameter for Beta distribution (used for alpha and beta) -- used in PSD and SP sims.
    :return: (dict) Dictionary with
    """
    randseed = config.seed
    var_genetic = config.snpsig
    var_strat = config.confint
    var_std_snps = config.stdsig
    bin_scale = config.bin_scale
    n_individuals_sim = config.numsnps
    n_causes_sim = config.numunits
    A = config.numparents
    ld_present = config.ld_present
    causal_num = config.causal_num

    #############################################################
    # set random seed
    #############################################################
    random.seed(randseed)
    npr.seed(randseed)
    torch.manual_seed(randseed)

    #############################################################
    # simulate genes (causes) and traits (outcomes)
    #############################################################
    

    G_sim = simulate_genotypes(G, A, n_individuals_sim, n_causes_sim)

    # remove genes that take the same value on all individuals
    const_cols = np.where(np.var(G_sim, axis=0) < 0.001)[0]
    print('SNPs removed:')
    print(const_cols)
    if len(const_cols) > 0:
        G_sim = G_sim[:, list(set(range(n_snps)) - set(const_cols))]
        n_causes_sim -= len(const_cols)

    print('Shape of SNPs:', G_sim.shape)

    hgdp = np.array(pd.read_csv("../../raw_data/hgdp/hgdp_subset.csv").T[1:])
    # top 1 PC from HGDP
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(hgdp)
    pcs = pca.transform(hgdp)

    y, y_bin, true_betas, true_lambdas = sim_single_traits_lmm(G_sim, pcs, var_genetic=var_genetic, var_std_snps=var_std_snps, var_strat=var_strat, ld_present=ld_present, causal_num=causal_num, base_scale=10, bin_scale=bin_scale)


    return {
        'snps': G,
        'outcomes': y,
        'y_bin': y_bin, 
        'true_betas': true_betas
    }

    
def simulate(config):
    """
    Adapted from https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/simdat.py
    :param: conf: (attridict) Dictionary assumed to have the following parameters:
        simset: (str) Simulation set, i.e., one of: ['BN', 'TGP', 'HGDP', 'PSD', 'SP'].
        seed: (int) Random seed used.
        numsnps: (int) Number of causal SNPs.
        numunits: (int) Number of individuals.
        snpsig: (float) Control signal-to-noise ratio (SNR) for SNPs
        confint: (float) Control SNR for confounders
        causalprop: (float) Fraction of SNPs that are causal
        alpha: (float) Parameter for Beta distribution (used for alpha and beta) -- used in PSD and SP sims.
    :return: (dict) Dictionary with
    """
    n_snps = config.numsnps
    n_units = config.numunits
    simset = config.simset
    randseed = config.seed
    alpha = config.alpha
    a = config.snpsig
    b = config.confint
    causalprop = config.causalprop
    bin_scale = config.bin_scale

    #############################################################
    # set random seed
    #############################################################
    random.seed(randseed)
    npr.seed(randseed)
    torch.manual_seed(randseed)

    raw_data_dir = get_raw_data_dir()
    if not osp.exists(raw_data_dir):
        raise FileNotFoundError(f'Missing raw data directory: \'{raw_data_dir}\'!')

    #############################################################
    # simulate genes (causes) and traits (outcomes)
    #############################################################
    # load Hapmap data for BN, PSD, SP
    # to preprocess the data, run clean_hapmap.py
    Fs = np.loadtxt(osp.join(raw_data_dir, 'clean_csv', 'Fs.csv'))
    ps = np.loadtxt(osp.join(raw_data_dir, 'clean_csv', 'ps.csv'))
    genes = pd.read_csv(osp.join(raw_data_dir, 'clean_csv', 'genes.csv'))
    n_genes = genes.shape[1]

    if simset == "BN":
        G, lambdas = sim_genes_BN(Fs, ps, n_snps, n_units, n_genes)
    elif simset == "TGP":
        tgp_pc = np.array(pd.read_csv("../../raw_data/tgp/tgp_pca.csv"))[:, :2]
        G, lambdas = sim_genes_TGP(n_snps, n_units, tgp_pc)
    elif simset == "HGDP":
        hgdp = np.array(pd.read_csv("../../raw_data/hgdp/hgdp_subset.csv").T[1:])
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(hgdp)
        hgdp_pc = pca.transform(hgdp)
        G, lambdas = sim_genes_HGDP(n_snps, n_units, hgdp_pc)
    elif simset == "PSD":
        G, lambdas = sim_genes_PSD(Fs, ps, n_snps, n_units, n_genes, alpha=alpha)
    elif simset == "SP":
        G, lambdas = sim_genes_SP(n_snps, n_units, a=alpha)
    else:
        raise ValueError('Unsupported simset provided.')

    # remove genes that take the same value on all individuals
    const_cols = np.where(np.var(G, axis=0) < 0.001)[0]
    print('SNPs removed:')
    print(const_cols)
    if len(const_cols) > 0:
        G = G[:, list(set(range(n_snps)) - set(const_cols))]
        n_snps -= len(const_cols)

    print('Shape of SNPs:', G.shape)
    y, y_bin, true_betas, true_lambdas = sim_single_traits(lambdas, G, a=a, b=b, causalprop=causalprop, bin_scale=bin_scale)
    return {
        'snps': G,
        'clusters': lambdas,
        'outcomes': y,
        'y_bin': y_bin, 
        'true_betas': true_betas,
        'true_lambdas': true_lambdas
    }


if __name__ == '__main__':
    # Defaults for argparse come from:
    # https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/run_script_simdat.sh
    parser = argparse.ArgumentParser(description='Simulate GWAS data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (yaml).')
    parser.add_argument('--save_file', type=str, default=None,
                        help='Path to where data file will be saved.')
    parser.add_argument('-sim', '--simset',
                        choices=['BN', 'TGP', 'HGDP', 'PSD', 'SP'], default='BN',
                        help='Simulation set to use (see https://arxiv.org/abs/1805.06826)')
    parser.add_argument('-ns', '--numsnps', type=int, default=100,
                        help='Number of SNP.')
    parser.add_argument('-nu', '--numunits', type=int, default=5000,
                        help='Number of individuals.')
    parser.add_argument('-cp', '--causalprop', type=float, default=0.01,
                        help='Fraction of SNPs that are causal.')
    parser.add_argument('-ss', '--snpsig', type=float, default=0.4,
                        help='Control signal to noise ratio for SNPs.')
    parser.add_argument('-ci', '--confint', type=float, default=0.4,
                        help='Control signal to noise ratio for confounders.')
    parser.add_argument('-alpha', '--alpha', type=int, default=0.01,
                        help='Alpha and beta parameter (same) for Beta distribution.')

    parser.add_argument('--seed', type=int, default=29200017,
                        help='Random seed initialization.')

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            conf = AttrDict(yaml.safe_load(f))
    else:
        conf = args
    # pdb.set_trace()
    
    for k, v in vars(args).items():
        if k not in conf.keys():
            setattr(conf, k, v)

    # can remove the following items from yaml file to avoid explicitly assigning values from args 
    # setting seed and numsnps explicitly for ease of experimentation via command line
    conf.seed = args.seed
    conf.numsnps = args.numsnps
    conf.causalprop=args.causalprop
    print(conf.seed)
    np.random.seed(conf.seed)

    if not conf.save_file:
        save_file = get_sim_gwas_dataset_file(conf)
        conf.save_file = osp.join(get_dataset_dir('simulated_gwas'), save_file)

    simulated_data = simulate(conf)
    torch.save(simulated_data, conf.save_file)


