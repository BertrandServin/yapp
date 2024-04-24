# -*- coding: utf-8 -*-
"""Module vcf.py from yapp

This module exports functions to parse VCF files into various input
for other modules in yapp or other packages. VCF files are read using cyvcf2.
"""
import sys
import logging
from collections import defaultdict
import numpy as np
import zarr
from cyvcf2 import VCF

logger = logging.getLogger(__name__)

modes_avail = ["genotype", "inbred", "phased", "likelihood"]
default_mode = modes_avail[0]


def geno2int(a1, a2):
    if a1 == -1 or a2 == -1:
        return -1
    else:
        return a1 + a2


def geno2hap(a1, a2):
    if a1 == -1 or a2 == -1 or (a1 != a2):
        return -1
    else:
        return a1


def phred2ln(x):
    """Transforms a PHRED score into a natural logarithm"""
    return -0.1 * np.log(10) * x


def chunk_region(reg, chunksize=-1):
    """Decompose reg into smaller regions
    Parameters
    ----------
    reg : str
       region in bed format ("chr:begin-end").
       begin and end may contain commas.
    chunksize : float
       size of chunks in basepairs

    Returns
    -------
    list of str
        list of sub-regions in bed format.
        If chunksize is <0, returns a list with the original region
    """
    if chunksize < 0:
        return [reg]
    chrom, coords = reg.split(":")
    beg, end = coords.split("-")
    beg = int(beg.replace(",", ""))
    end = int(end.replace(",", ""))
    starts = np.arange(beg, end, step=chunksize, dtype=int)
    ends = [int(x + chunksize - 1) for x in starts]
    ends[-1] = end
    reglist = []
    for i, s in enumerate(starts):
        newreg = chrom + ":" + str(s) + "-" + str(ends[i])
        reglist.append(newreg)
    return reglist


def vcf_chunk_regions(fname, chunksize=-1):
    """Creates regions of chunksize in a VCF file from contig header information
    Parameters
    ----------

    fname : str
        The name of the vcf(.gz) file
    chunksize : int
        Size of chunks

    Returns
    -------
    list of str
        List of regions in bed format. If chunksize is <0, regions are contigs.
    """
    v = VCF(fname, lazy=True)
    regions = [name + ":" + "0-" + str(l) for name, l in zip(v.seqnames, v.seqlens)]
    if chunksize > 0:
        new_regions = []
        for r in regions:
            new_regions += chunk_region(r, chunksize)
        regions = new_regions
    return regions


def vcf2fph(
    fname,
    mode=default_mode,
    samples=None,
    reg=None,
    maf=0.01,
    varids=None,
    genotypes=True,
):
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

        In 'phased' mode, two haplotype 1D arrays are returned per
        sample, using phased information in the VCF (eg. 0|1). Any non
        phased genotype is considered missing.

        In 'likelihood' mode, the resulting 2D arrays contain the
        genotype log-likelihoods extracted from the 'PL' field.

    samples : list
        List of samples to extract from full set in file
    reg : str
        query variants in region reg only in bed format ("chr:begin-end").
        VCF file must be indexed.
    maf : double
        Minimum allele frequency to include variants
    varids : list
        List of variants IDs to extract. Only variants found in vcf are
        returned, irrespective of MAF.
    genotypes : bool
        Return Genotype data. For large files this can be quite big.
    Returns
    -------
    dict
        {
         regions : list of regions (str)
         samples : list of samples (str)
         variants : dict {region : list of variants (ID, CHROM, POS, REF, ALT)}
         data : dict {region : dict {
                                  sample ID: data (numpy array)
                                  data type depends on mode. see Parameters.)}
        }

    """

    v = VCF(fname, gts012=True, strict_gt=True, samples=samples, lazy=True)
    smp = v.samples  # might differ if some requested are not found

    if reg is None:
        regions = v.seqnames
    else:
        regions = [reg]

    if varids is None:
        keepvar = defaultdict(lambda: True)
    else:
        maf = 0
        keepvar = defaultdict(lambda: False)
        for s in varids:
            keepvar[s] = True
    variants = {}
    trueregions = []
    variants_summary = {}
    fphdata = {}
    for r in regions:
        snps = []
        for s in v(r):
            if len(s.ALT) > 1 or s.aaf < maf or not keepvar[s.ID]:
                continue
            snps.append(s)
        if len(snps) == 0:
            continue
        variants[r] = sorted(snps, key=lambda x: x.POS)
        variants_summary[r] = [
            (s.ID, s.CHROM, s.POS, s.REF, s.ALT[0]) for s in variants[r]
        ]
        fphdata[r] = {}
        trueregions.append(r)
        if genotypes:
            for i, sid in enumerate(smp):
                if mode == "likelihood":
                    fphdata[r][sid] = phred2ln(
                        np.array(
                            [
                                [
                                    s.gt_phred_ll_homref[i],
                                    s.gt_phred_ll_het[i],
                                    s.gt_phred_ll_homalt[i],
                                ]
                                for s in variants[r]
                            ],
                            dtype=float,
                        )
                    )
                else:
                    geno = [s.genotypes[i] for s in variants[r]]
                    if mode == "genotype":
                        fphdata[r][sid] = np.array(
                            [geno2int(*g[:2]) for g in geno], dtype=int
                        )
                    elif mode == "inbred":
                        fphdata[r][sid] = np.array(
                            [geno2hap(*g[:2]) for g in geno], dtype=int
                        )
                    elif mode == "phased":
                        h1 = []
                        h2 = []
                        for g in geno:
                            if g[2]:
                                h1.append(g[0])
                                h2.append(g[1])
                            else:
                                h1.append(-1)
                                h2.append(-1)
                        fphdata[r][sid + ".h1"] = np.array(h1, dtype=int)
                        fphdata[r][sid + ".h2"] = np.array(h2, dtype=int)
    return {
        "regions": trueregions,
        "samples": smp,
        "variants": variants_summary,
        "data": fphdata,
    }


def vcf2zarr(
    fname,
    output_prefix=None,
    mode=default_mode,
    samples=None,
    reg=None,
    maf=0.01,
    varids=None,
):
    """Parses a VCF file and creates a ZARR file with its contents

    Parameters
    ----------
    fname : str
        The name of the vcf(.gz) file
    output_prefix: str
        The prefix for zarr file name (prefix.zarr)
    mode : str
        One of 'genotype', 'inbred', 'phased' or 'likelihood'.

        In 'genotype' mode, the returning 1D arrays are genotypes
        (0,1,2 and -1 for missing data)

        In 'inbred' mode, the resulting 1D arrays are haplotypes (0,1 and -1).
        Heterozygote calls are treated as missing data.

        In 'phased' mode, two haplotype 1D arrays are returned per
        sample, using phased information in the VCF (eg. 0|1). Any non
        phased genotype is considered missing.

        In 'likelihood' mode, the resulting 2D arrays contain the
        genotype log-likelihoods extracted from the 'PL' field.

    samples : list
        List of samples to extract from full set in file
    reg : str
        query variants in region reg only in bed format
        ("chr:begin-end"). VCF file must be indexed.
    maf : double
        Minimum allele frequency to include variants
    varids : list
        List of variants IDs to extract. Only variants found in vcf
        are returned, irrespective of MAF.
    genotypes : bool
        Return Genotype data. For large files this can be quite big.
    Returns
    -------
    str
       handler to the zarr root group

    """

    v = VCF(fname, gts012=True, strict_gt=True, samples=samples, lazy=True)
    if not output_prefix:
        output_prefix = fname.split(".")[0]
    fzname = output_prefix + ".zarr"
    fz = zarr.open(fzname, "w")
    fz.attrs["archive"] = fzname
    smp = v.samples  # might differ if some requested are not found
    if reg is None:
        _regions = v.seqnames
    else:
        _regions = [reg]

    regions = []
    for r in _regions:
        ns = len([s for s in v(r)])
        if ns > 0:
            regions.append(r)
    if len(regions) < 1:
        raise IOError(f"No SNP found in {fname}")

    logger.info("Creating Archive structure")
    fz["samples"] = zarr.array(smp, dtype="U50")
    fz["regions"] = zarr.array(regions, dtype="U50")
    for i, reg in enumerate(regions):
        fz.create_group(f"variants/{reg}")
    fz.create_group("genotypes")

    if varids is None:
        keepvar = defaultdict(lambda: True)
    else:
        maf = 0
        keepvar = defaultdict(lambda: False)
        for s in varids:
            keepvar[s] = True

    for r in regions:
        logger.info(f"Getting data for region {r}")
        snps = []
        for s in v(r):
            if len(s.ALT) > 1 or s.aaf < maf or not keepvar[s.ID]:
                continue
            snps.append(s)
        if len(snps) == 0:
            continue
        variants = sorted(snps, key=lambda x: x.POS)
        fz[f"variants/{r}/ID"] = np.array([s.ID for s in variants], dtype="U50")
        fz[f"variants/{r}/CHROM"] = np.array([s.CHROM for s in variants], dtype="U50")
        fz[f"variants/{r}/POS"] = np.array([s.POS for s in variants], dtype=np.uint32)
        fz[f"variants/{r}/REF"] = np.array([s.REF for s in variants], dtype="S20")
        fz[f"variants/{r}/ALT"] = np.array([s.ALT[0] for s in variants], dtype="S20")

        data = []
        for i, sid in enumerate(smp):
            sys.stdout.write(f"{sid:<20} {100*(i+1)/len(smp):.2f} %\r")
            sys.stdout.flush()
            if mode == "likelihood":
                gind = phred2ln(
                    np.array(
                        [
                            [
                                s.gt_phred_ll_homref[i],
                                s.gt_phred_ll_het[i],
                                s.gt_phred_ll_homalt[i],
                            ]
                            for s in variants[r]
                        ],
                        dtype=float,
                    )
                )
            else:
                geno = [s.genotypes[i] for s in variants]
                if mode == "genotype":
                    gind = np.array([geno2int(*g[:2]) for g in geno], dtype=np.int8)
                elif mode == "inbred":
                    gind = np.array([geno2hap(*g[:2]) for g in geno], dtype=np.int8)
                elif mode == "phased":
                    gind = np.array(
                        [
                            [g[0] if g[2] else -1 for g in geno],
                            [g[1] if g[2] else -1 for g in geno],
                        ],
                        dtype=np.int8,
                    )
            data.append(gind)
        fz[f"genotypes/{r}"] = zarr.array(data, chunks=False)
    return fz
