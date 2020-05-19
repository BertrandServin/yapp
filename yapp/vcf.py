# -*- coding: utf-8 -*-
"""Module vcf.py from yapp

This module exports functions to parse VCF files into various input 
for other modules in yapp or other packages. VCF files are read using cyvcf2.
"""
from collections import defaultdict
import numpy as np
from cyvcf2 import VCF

modes_avail=['genotype','inbred','phased','likelihood']
default_mode=modes_avail[0]

def geno2int(a1,a2):
    if a1 == -1 or a2 == -1:
        return -1
    else:
        return a1+a2

def geno2hap(a1,a2):
    if a1==-1 or a2==-1 or (a1!=a2):
        return -1
    else:
        return a1

def phred2ln(x):
    """Transforms a PHRED score into a natural logarithm"""
    return -0.1*np.log(10)*x
        
def vcf2fph(fname, mode='genotype', samples=None, reg=None,maf=0.01, varids=None):
    """Parses a VCF file and returns data arrays usable with 
    the fastphase package

    Parameters
    ----------
    fname : str
        The name of the vcf(.gz) file
    mode : str
        One of 'genotype', 'inbred', 'phased' or 'likelihood'.

        In 'genotype' mode, the returning 1D arrays are genotypes 
        (0,1,2 and -1 for missing data)

        In 'inbred' mode, the resulting 1D arrays are haplotypes (0,1 and -1).
        Heterozygote calls are treated as missing data.
        
        In 'phased' mode, two haplotype 1D arrays are returned per sample, using phased 
        information in the VCF (eg. 0|1). Any non phased genotype is considered missing.

        In 'likelihood' mode, the resulting 2D arrays contain the genotype log-likelihoods 
        extracted from the 'PL' field.
    samples : list
        List of samples to extract from full set in file
    reg : str 
        query variants in region reg only in bed format ("chr:begin-end"). VCF file must be indexed.
    maf : double
        Minimum allele frequency to include variants
    varids : list
        List of variants IDs to extract. Only variants found in vcf are returned, irrespective of MAF.
    Returns
    -------
    dict
        {
         regions : list of regions (str)
         samples : list of samples (str)
         variants : dict { region : list of variants (ID, CHROM, POS, REF, ALT) }
         data : dict { region : dict { sample ID: data (numpy array, data type depends on mode. see Parameters.)}
        }
    
    """

    v = VCF(fname, gts012=True, strict_gt=True, samples=samples)
    smp = v.samples ## might differ if some requested are not found

    if reg is None:
        regions=v.seqnames
    else:
        regions=[reg]
    if varids is None:
        keepvar=defaultdict(lambda: True)
    else:
        maf=0
        keepvar=defaultdict(lambda: False)
        for s in varids:
            keepvar[s]=True
    variants={}
    trueregions=[]
    variants_summary={}
    fphdata={}
    for r in regions:
        snps = []
        for s in v(r):
            if len(s.ALT)>1 or s.aaf<maf or not keepvar[s.ID]:
                continue
            snps.append(s)
        if len(snps) == 0:
            continue
        variants[r] = sorted(snps, key = lambda x: x.POS)
        variants_summary[r]=[(s.ID,s.CHROM,s.POS,s.REF,s.ALT[0]) for s in variants[r]]
        fphdata[r] = {}
        trueregions.append(r)
        for i, sid in enumerate(smp):
            if mode == 'likelihood':
                fphdata[r][sid] = phred2ln( np.array([ [s.gt_phred_ll_homref[i],
                                                        s.gt_phred_ll_het[i],
                                                        s.gt_phred_ll_homalt[i]] for s in variants[r]],
                                                     dtype=np.float))
            else:
                geno = [s.genotypes[i] for s in variants[r]]
                if mode =='genotype':
                    fphdata[r][sid] = np.array([geno2int(*g[:2]) for g in geno], dtype=np.int)
                elif mode == 'inbred':
                    fphdata[r][sid] = np.array([geno2hap(*g[:2]) for g in geno], dtype=np.int)
                elif mode == 'phased':
                    h1=[]
                    h2=[]
                    for g in geno:
                        if g[2]:
                            h1.append(g[0])
                            h2.append(g[1])
                        else:
                            h1.append(-1)
                            h2.append(-1)
                    fphdata[r][sid+'.h1']=np.array(h1,dtype=np.int)
                    fphdata[r][sid+'.h2']=np.array(h2,dtype=np.int)
    return {
        'regions' : trueregions,
        'samples' : smp,
        'variants' : variants_summary,
        'data' : fphdata
        }
            
