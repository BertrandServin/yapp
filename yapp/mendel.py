import sys
import os
import logging
from collections import defaultdict
from multiprocessing import Pool
import bz2
import numpy as np
from scipy.stats import binom
from cyvcf2 import VCF
from . import vcf, gamete, pedigree, MALE, FEMALE

logger = logging.getLogger(__name__)

def genotype_vector( genotypes ):
    return np.array([vcf.geno2int(*g[:2]) for g in genotypes],dtype=np.int)

def mendel_errors( args):
    genotypes, pairs= args
    from_genotypes = np.array(genotypes)[pairs[:,0],:2]
    to_genotypes = np.array(genotypes)[pairs[:,1],:2]
    fg = genotype_vector(from_genotypes)
    tg = genotype_vector(to_genotypes)
    nobs =  (fg>0)&(tg>0)
    nerr =  ((fg==0)&(tg==2))|((fg==2)&(tg==0))
    return nerr,nobs

def main(args):
    prfx=args.prfx
    vcf_file = f"{prfx}.vcf.gz"
    fam_file = f"{prfx}.fam"
    myvcf=VCF(vcf_file, gts012=True, strict_gt=True, lazy=True)
    ped = pedigree.Pedigree.from_fam_file(fam_file)

    logger = logging.getLogger('yapp')
    
    ## Identify indices
    indiv_idx=defaultdict( lambda : -1)
    for i,v in enumerate(myvcf.samples):
        try:
            node = ped.nodes[v]
            indiv_idx[node.indiv]=i
        except KeyError:
            continue

    ## Find parent -> offspring pairs of indices
    pairs=[]
    for node in ped:
        if indiv_idx[node.indiv]<0:
            continue
        for c in node.children:
            if indiv_idx[c.indiv]<0:
                continue
            pairs.append((indiv_idx[node.indiv], indiv_idx[c.indiv]))
    pairs = np.array(pairs, dtype=np.int)

    geno_getter = ( (s.genotypes, pairs) for s in myvcf)
    merr = np.full_like(pairs, 0,dtype=np.int)
    with Pool(args.c) as workers:
        for nerr,nobs in workers.imap_unordered(mendel_errors, geno_getter,chunksize=10000):
            merr[:,0]+=nerr
            merr[:,1]+=nobs
    tx_err=np.sum(merr[:,0])/np.sum(merr[:,1])
    pval_th = 0.001/pairs.shape[0]

    logger.info(f"Global Mendel Error rate : {tx_err:.2g}")
    unset_links=[]
    with open(f"{prfx}.mendel.err","w") as fout:
        print("parent offspring nerr nobs err.rate pvalue",file=fout)
        for (p, e) in zip(pairs, merr):
            pval = binom.sf(n=e[1],p=tx_err, k=e[0])
            print(f"{myvcf.samples[p[0]]} {myvcf.samples[p[1]]} {e[0]} {e[1]} {e[0]/e[1]:.2g} {pval:.2g}",file=fout)
            if pval<pval_th:
                unset_links.append(tuple(p))
    if len(unset_links)>0:
        logger.info(f"Saving original fam file to {prfx}.fam.orig")
        os.replace(f"{prfx}.fam",f"{prfx}.fam.orig")
        logger.info(f"Correcting pedigree errors in fam file")
        with open(f"{prfx}.fam.orig") as fam:
            with open(f"{prfx}.fam.new",'w') as nfam:
                for ligne in fam:
                    buf=ligne.split()
                    try:
                        pat_p = (indiv_idx[buf[2]], indiv_idx[buf[1]])
                    except KeyError:
                        pass
                    else:
                        if pat_p in unset_links:
                            logger.info(f'removing paternal link {buf[2]} -> {buf[1]}')
                            buf[2]='0'
                    try:
                        mat_p = (indiv_idx[buf[3]], indiv_idx[buf[1]])
                    except KeyError:
                        pass
                    else:
                        if mat_p in unset_links:
                            logger.info(f'removing maternal link {buf[3]} -> {buf[1]}')
                            buf[3]='0'
                    print(' '.join(buf),file=nfam)
        logger.info(f"new fam file is : {prfx}.fam.new")
                        
if __name__=='__main__':
    main(sys.argv[1:])
