import os
import logging
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.stats import binom
from numba import jit
from cyvcf2 import VCF
from . import vcf, pedigree

logger = logging.getLogger(__name__)

def genotype_vector(genotypes):
    return np.array([vcf.geno2int(*g[:2]) for g in genotypes], dtype=int)

def mendel_errors(args):
    genotypes, pairs = args
    pairs = np.array(pairs)
    genotypes = np.array(genotypes)
    from_genotypes = np.array(genotypes)[pairs[:, 0], :2]
    to_genotypes = np.array(genotypes)[pairs[:, 1], :2]
    fg = genotype_vector(from_genotypes)
    tg = genotype_vector(to_genotypes)
    nobs = ((fg == 0) | (fg == 2)) & ((tg == 0) | (tg == 2))
    nerr = ((fg == 0) & (tg == 2)) | ((fg == 2) & (tg == 0))
    return nerr, nobs


def identify_bad_pairs(pairs, merr, fpr=1e-3, iterate = False):
    """Identify outliers for mendelian errors.

    Starting from all pairs, outliers are identified iteratively as follows:

    1. Compute overal error rate
    2. For each pair compute the p-value (from Binomial) that data comes
       from the binomial
    3. reject the null is p-value < fpr/len(pairs) (Bonferroni correction)
    4. Remove pairs for which the null is rejected from the pair list

    If iterate, these steps are iterated until no pair is removed from the set.

    Arguments
    ---------
    pairs : list of objects
        family links
    merr : array len(pairs) x 2
        array of number of errors (merr[:,0]) and number of
        informative loci(merr[:,1])
    fpr : float
        false positive rate to call outliers

    Returns
    -------
    list of objects
        List of pairs from the input list that are outliers
    """
    pair2rm = []
    pval_th = fpr / len(pairs)
    logger.info(f"Identifying links with p < {pval_th:.2g}")
    current_merr = merr.copy()
    current_pairs = pairs[:]
    while True:
        tx_err = np.sum(current_merr[:, 0]) / np.sum(current_merr[:, 1])
        logger.debug(f"te : {tx_err}")
        tmp_pairs = current_pairs[:]
        keepidx = np.ones(len(current_pairs), dtype=bool)
        for i, (p, e) in enumerate(zip(current_pairs, current_merr)):
            pval = binom.sf(n=e[1], p=tx_err, k=e[0])
            if pval < pval_th:
                logger.debug(f"{p} : pbinom({e[0]}, size={e[1]},  p={tx_err}) = {pval}")
                pair2rm.append(p)
                tmp_pairs.remove(p)
                keepidx[i] = False
        if not iterate or keepidx.all():
            break
        logger.debug(f"Start with {len(current_pairs)} -> {len(tmp_pairs)}")
        current_pairs = tmp_pairs[:]
        current_merr = current_merr[keepidx, :]
    return pair2rm


def main(args):
    prfx = args.prfx
    vcf_file = f"{prfx}.vcf.gz"
    fam_file = f"{prfx}.fam"
    myvcf = VCF(vcf_file, gts012=True, strict_gt=True, lazy=True)
    ped = pedigree.Pedigree.from_fam_file(fam_file)

    # Identify indices
    indiv_idx = defaultdict(lambda: -1)
    for i, v in enumerate(myvcf.samples):
        try:
            node = ped.nodes[v]
            indiv_idx[node.indiv] = i
        except KeyError:
            continue

    # Find parent -> offspring pairs of indices
    pairs = []
    for node in ped:
        if indiv_idx[node.indiv] < 0:
            continue
        for c in node.children:
            if indiv_idx[c.indiv] < 0:
                continue
            pairs.append((indiv_idx[node.indiv], indiv_idx[c.indiv]))

    logger.info(f"Found {len(pairs)} offspring-parents pairs to check")
    # pairs = np.array(pairs, dtype=int)

    geno_getter = ((s.genotypes, pairs) for s in myvcf)
    merr = np.zeros((len(pairs), 2), dtype=int)
    with Pool(args.c) as workers:
        for nerr, nobs in workers.imap_unordered(
            mendel_errors, geno_getter#, chunksize=1000
        ):
            merr[:, 0] += nerr
            merr[:, 1] += nobs
    tx_err = np.sum(merr[:, 0]) / np.sum(merr[:, 1])
    # pval_th = 0.001/pairs.shape[0]

    logger.info(f"Global Mendel Error rate : {tx_err:.2g}")
    # unset_links=[]
    unset_links = identify_bad_pairs(pairs, merr)
    with open(f"{prfx}_yapp_mendel.err", "w") as fout:
        print("parent offspring nerr nobs err.rate pvalue removed", file=fout)
        for p, e in zip(pairs, merr):
            pval = binom.sf(n=e[1], p=tx_err, k=e[0])
            print(
                f"{myvcf.samples[p[0]]} "
                f"{myvcf.samples[p[1]]} "
                f"{e[0]} {e[1]} {e[0]/e[1]:.2g} {pval:.2g} "
                f"{p in unset_links}",
                file=fout,
            )

    if len(unset_links) > 0:
        logger.info(f"Saving original fam file to {prfx}.fam.orig")
        os.replace(f"{prfx}.fam", f"{prfx}.fam.orig")
        logger.info("Correcting pedigree errors in fam file")
        with open(f"{prfx}.fam.orig") as fam:
            with open(f"{prfx}.fam.new", "w") as nfam:
                for ligne in fam:
                    buf = ligne.split()
                    try:
                        pat_p = (indiv_idx[buf[2]], indiv_idx[buf[1]])
                    except KeyError:
                        pass
                    else:
                        if pat_p in unset_links:
                            logger.info(f"removing paternal link {buf[2]} -> {buf[1]}")
                            buf[2] = "0"
                    try:
                        mat_p = (indiv_idx[buf[3]], indiv_idx[buf[1]])
                    except KeyError:
                        pass
                    else:
                        if mat_p in unset_links:
                            logger.info(f"removing maternal link {buf[3]} -> {buf[1]}")
                            buf[3] = "0"
                    print(" ".join(buf), file=nfam)
        logger.info(f"new fam file is : {prfx}.fam.new")
