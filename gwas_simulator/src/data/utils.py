import os
from os import path as osp

import numpy as np
import numpy.random as npr
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pdb

def get_dataset_dir(dataset):
    """
    Helper method to return dataset directory.

    :param dataset: (str) Name of the current dataset.
    :return: (str) dataset_dir.
    """
    current_file_path = osp.abspath(__file__)
    current_dir = osp.abspath(osp.join(current_file_path, os.pardir))
    repo_dir = osp.abspath(osp.join(osp.abspath(osp.join(current_dir, os.pardir)), os.pardir))
    os.makedirs(osp.join(repo_dir, 'datasets', 'simulated_gwas'), exist_ok=True)
    return osp.join(repo_dir, 'datasets', dataset)


def get_raw_data_dir():
    current_file_path = osp.abspath(__file__)
    current_dir = osp.abspath(osp.join(current_file_path, os.pardir))
    repo_dir = osp.abspath(osp.join(osp.abspath(osp.join(current_dir, os.pardir)), os.pardir))
    return osp.join(repo_dir, 'raw_data', 'hapmap')


def get_sim_gwas_dataset_file(conf):
    """
    Helper method to build file name for simulated GWAS.

    :param: conf: (attridict) Dictionary assumed to have the following parameters:
        simset: (str) Simulation set, i.e., one of: ['BN', 'TGP', 'HGDP', 'PSD', 'SP'].
        seed: (int) Random seed used.
        numsnps: (int) Number of causal SNPs.
        numunits: (int) Number of individuals.
        snpsig: (float) Control signal-to-noise ratio (SNR) for SNPs
        confint: (float) Control SNR for confounders
        causalprop: (float) Fraction of SNPs that are causal
        alpha: (float) Parameter for Beta distribution (used for alpha and beta) -- used in PSD and SP sims.
    :return: (str) File name (with .pt extension) based on config params.
    """
    if conf.simset == 'SP' or conf.simset == 'PSD':
        simset = f'{conf.simset}_alpha-{conf.alpha}'
    else:
        simset = conf.simset
    return f'sim-{simset}_seed-{conf.seed}_nu-{conf.numunits}_ns-{conf.numsnps}_snpsig-{conf.snpsig}_' + \
           f'confint-{conf.confint}_cp-{conf.causalprop}.pt'


# Copied from https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/utils.py
def get_G_lambdas(Gammamat, S):
    F = S.dot(Gammamat.T)
    # changing to remove SNP '2'
    G = npr.binomial(1, F)
    # TODO: In "Blessings" the clustering is done on random columns of SNPs matrix (commented lines below)
    # scaledS = G[:, npr.randint(G.shape[1], size=10)]
    # lambdas = KMeans(n_clusters=3, random_state=123).fit(scaledS).labels_
    lambdas = KMeans(n_clusters=3, random_state=123).fit(S).labels_  # TODO: Changed kmeans to be on the matrix S
    return G, lambdas


def sim_genes_BN(Fs, ps, n_causes, n_units, n_genes, D=3):
    """
    Copied from:
    https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/utils.py
    """

    idx = npr.randint(n_genes, size=n_causes)
    p = ps[idx]
    F = Fs[idx]
    Gammamat = np.zeros((n_causes, D))
    for i in range(D):
        Gammamat[:, i] = npr.beta((1 - F) * p / F, (1 - F) * (1 - p) / F)
    S = npr.multinomial(1, (60 / 210, 60 / 210, 90 / 210), size=n_units)
    F = S.dot(Gammamat.T)
    # changing to remove SNP '2'
    G = npr.binomial(1, F)
    # TODO: In "Blessings" the clustering is done on random columns of SNPs matrix (commented lines below)
    # scaledS = G[:, npr.randint(G.shape[1], size=10)]
    # lambdas = KMeans(n_clusters=3, random_state=42).fit(scaledS).labels_
    lambdas = np.where(S == 1)[1]  # TODO: Changed cluster to just be index of 1 in each row of S
    return G, lambdas


def sim_genes_TGP(n_causes, n_units, tgp_pc):
    """
    Copied from:
    https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/utils.py
    """
    S = np.column_stack([tgp_pc])
    scaler = MinMaxScaler()
    scaler.fit(S)
    S = scaler.transform(S)
    Gammamat = np.zeros((n_causes, 3))
    # import pdb;pdb.set_trace()
    Gammamat[:, 0] = 0.45 * npr.uniform(size=n_causes)
    Gammamat[:, 1] = 0.45 * npr.uniform(size=n_causes)
    Gammamat[:, 2] = 0.05 * np.ones(n_causes)
    S = np.column_stack((S[npr.choice(S.shape[0], size=n_units, replace=True), ], np.ones(n_units)))
    return get_G_lambdas(Gammamat, S)


def sim_genes_HGDP(n_causes, n_units, tgp_pc):
    """
    Copied from:
    https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/utils.py
    """
    return sim_genes_TGP(n_causes, n_units, tgp_pc)


def sim_genes_PSD(Fs, ps, n_causes, n_units, n_genes, alpha=0.5):
    """
    Copied from:
    https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/utils.py
    """
    idx = npr.randint(n_genes, size=n_causes)
    p = ps[idx]
    F = Fs[idx]
    Gammamat = np.zeros((n_causes, 3))
    for i in range(3):
        Gammamat[:, i] = npr.beta((1 - F) * p / F, (1 - F) * (1 - p) / F)
    S = npr.dirichlet((alpha, alpha, alpha), size=n_units)
    return get_G_lambdas(Gammamat, S)


def sim_genes_SP(n_causes, n_units, D=3, a=0.1):
    """
    Copied from:
    https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/utils.py
    """
    Gammamat = np.zeros((n_causes, D))
    Gammamat[:, 0] = 0.45 * npr.uniform(size=n_causes)
    Gammamat[:, 1] = 0.45 * npr.uniform(size=n_causes)
    Gammamat[:, 2] = 0.05 * np.ones(n_causes)
    S = npr.beta(a, a, size=(n_units, 2))
    S = np.column_stack((S, np.ones(n_units)))
    return get_G_lambdas(Gammamat, S)


def sim_single_traits(lambdas, G, a, b, causalprop=0.05, base_scale=10, bin_scale=0.5):
    # pdb.set_trace()
    tau = 1./npr.gamma(3, 1, size=3)
    sigmasqs = tau[lambdas]
    epsilons = npr.normal(0, sigmasqs)
    n_causes = G.shape[1]
    betas = npr.normal(0, 0.5**2, size=n_causes)
    causal_snps = int(causalprop*n_causes)
    betas[causal_snps:] = 0.0

    c = 1 - a - b

    raw_y = G.dot(betas)
    raw_y_std = raw_y.std()

    y = raw_y + \
        np.sqrt(b)*raw_y_std/np.sqrt(a)*(lambdas-lambdas.mean())/lambdas.std() + \
        np.sqrt(c)*raw_y_std/np.sqrt(a)*epsilons/epsilons.std()
    
    scale = raw_y_std / base_scale
    y = (y - y.mean()) / scale

    y_bin = npr.binomial(1, expit(y / bin_scale))

    true_lambdas = np.sqrt(b) * raw_y_std / np.sqrt(a) * lambdas / lambdas.std() / scale
    true_betas = betas / scale

    print('confounding strength np.corrcoef(y, true_lambdas)', np.corrcoef(y, true_lambdas)[0, 1])

    return y, y_bin, true_betas, true_lambdas


def simulate_genotypes(G, pops, base_locs, A=2, n_individuals_sim=1000, n_causes_sim=5000):
    # simulated genotype matrix
    block_size=1000
    n_individuals = G.shape[0]
    n_causes = G.shape[1]

    # the ancestor for the i-th simulated individual needs to be selected without replacement but the following code is more efficient
    ancestors = npr.randint(n_individuals, size=( n_individuals_sim, A))
    parent_selection = npr.randint(A, size=(n_individuals_sim, int(n_causes_sim/1000)))
    G_sim = np.zeros((n_individuals_sim, n_causes_sim))
    ethnicity_sim = npr.randint(len(base_locs), size=(n_individuals_sim))
    
    # simulate genotype for each individual

    for i in range(n_individuals_sim):
        
        ancestor_indices = base_locs[ethnicity_sim[i]]
        ancestors[i] = npr.randint(ancestor_indices[0], ancestor_indices[1], size=(1, A))
        parent_selector = ancestors[i] # randomly chosen A individuals for i-th simulated individual
        # ethnicity.append((pops[parent_selector[0]], pops[parent_selector[1]]))
        
        for j in range(int(n_causes_sim/1000)):

            segment_selector = parent_selection[i][j]
            # copy segment from the relevant individual
            start = block_size*j
            end = (block_size)*(j+1)


            G_sim[i][start:end] = G[parent_selector[segment_selector]][start:end]
    return G_sim, ethnicity_sim



def sim_single_traits_lmm(G, pcs, var_genetic=0.5, var_std_snps=0.02, var_strat=0.10, ld_present=False, causal_num=5000, base_scale=10, bin_scale=0.5):
    # pdb.set_trace()
    
    # percentage of phenotypic variance explained by each component

    skip_size = 1000 # original simulation asks us to skip 2 centimorgan (cM) and 2 megabase (Mb) from the center of a strand in each chromosome

    var_std_snps = 0 # if we don't want polygenic signal
    var_env = 1 - (var_genetic+var_std_snps+var_strat)
    
    n_individuals = G.shape[0]
    n_causes = G.shape[1]

    epsilons = npr.normal(0, 0.5*2, size=n_individuals)
    betas = npr.normal(0, 0.5**2, size=n_causes)
    # betas_polygenic may help in power comparisons
    betas_polygenic = npr.normal(0, 0.5**2, size=n_causes)
    

    # if base SNP matrix contains LD, select causal_snps randomly from half of the chromosome and test null snps on the other half
    if not ld_present:
        causal_snps = int(causal_num)
        betas[causal_snps:] = 0.0
    else:
        # randomly select causal snps
        causal_snps = npr.choice(int(n_causes/2)-skip_size, size = causal_num, replace=False)
        # remaining non-causal snps until midline-skipsize
        non_causal_snps=list()
        for i in range(int(n_causes/2)-skip_size):
            if i not in causal_snps:
                non_causal_snps.append(i)
        non_causal_snps = np.array(non_causal_snps)
        betas[non_causal_snps] = 0.0

        betas[int(n_causes/2)-skip_size:] = 0.0
        
        
    
    raw_y = G.dot(betas)
    polygenic_y = G.dot(betas_polygenic)
    raw_y_std = raw_y.std()
    # pdb.set_trace()
    # lambdas = KMeans(n_clusters=10, random_state=123).fit(pcs).labels_ 
    lambdas = pcs


    y = raw_y + \
        np.sqrt(var_strat)*raw_y_std/np.sqrt(var_genetic)*(lambdas-lambdas.mean())/lambdas.std() + \
        np.sqrt(var_env)*raw_y_std/np.sqrt(var_genetic)*epsilons/epsilons.std() + \
        np.sqrt(var_std_snps)*raw_y_std/np.sqrt(var_genetic)*polygenic_y/polygenic_y.std()

    scale = raw_y_std / base_scale
    y = (y - y.mean()) / scale

    y_bin = npr.binomial(1, expit(y / bin_scale))

    transf_lambdas = np.sqrt(var_strat)*raw_y_std/np.sqrt(var_genetic)*(lambdas-lambdas.mean())/lambdas.std()
    # true_lambdas = np.sqrt(var_strat) * raw_y_std / np.sqrt(var_genetic) * lambdas / lambdas.std() / scale
    true_betas = betas / scale

    # print('confounding strength np.corrcoef(y, true_lambdas)', np.corrcoef(y, true_lambdas)[0, 1])

    return y, y_bin, true_betas, causal_snps, lambdas, transf_lambdas
