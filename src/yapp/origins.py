# -*- coding: utf-8 -*-
"""
Yapp module origins.py

Trace ancestral origins down a pedigree.
"""
import logging
from collections import defaultdict, namedtuple
import numpy as np
from zarr.errors import ContainsGroupError
from numba import njit, prange
from . import family_phaser

logger = logging.getLogger(__name__)


@njit(parallel=True)
def compute_grm(nindiv, nsnp, origins, outgrm):
    # pylint: disable=not-an-iterable
    for i in range(nindiv):
        for j in prange(i, nindiv):
            val = 0
            for ell in range(nsnp):
                val += (
                    (origins[i, 0, ell] == origins[j, 0, ell])
                    + (origins[i, 1, ell] == origins[j, 0, ell])
                    + (origins[i, 0, ell] == origins[j, 1, ell])
                    + (origins[i, 1, ell] == origins[j, 1, ell])
                )
            outgrm[i, j] += val


#### IBD matches code
IBD_track = namedtuple("IBD_track", "id1, id2, start, end")


def build_pbwt(H):
    """Returns the Positional Burrows-Wheeler Transform (Durbin, 2014)
    of a haplotype collection H in the form of a positional-prefix array
    and a divergence array. H must me castable to a numpy array of short,
    first dimenstion is samples, second is sites. 0,1 are alleles ; negative
    numbers are missing data.
    """
    H = np.asarray(H, dtype=np.short)
    ## M haplotypes over N markers
    M, N = H.shape

    ppa = -1 * np.ones_like(H, dtype=np.short)
    div = 0 * np.ones_like(H, dtype=np.short)

    ## initialize positional prefix array (sort on first allele)
    ## initialize divergence array (not in the paper but makes a difference imho)
    ## the first "0" or first "1" allele seen has no match, this leads to
    ## div > k in these situations, which is ok.
    y = H[:, 0]
    a = np.flatnonzero(y == 0)
    b = np.flatnonzero(y != 0)
    ppa[:, 0] = np.hstack((a, b))
    div[0, 0] = 1
    div[len(a), 0] = 1

    for k in range(N - 1):
        y = H[ppa[:, k], k + 1]
        p = q = k + 2
        a = []
        b = []
        d = []
        e = []
        for i in range(M):
            if y[i] < 0:  ## missing data
                if i > 0:
                    y[i] = y[i - 1]  ## "impute" from closest match
                else:
                    y[i] = 0
            if div[i, k] > p:
                p = div[i, k]
            if div[i, k] > q:
                q = div[i, k]
            if y[i] == 0:  ## note y[i] is 0/1
                a.append(ppa[i, k])
                d.append(p)
                p = 0
            else:
                b.append(ppa[i, k])
                e.append(q)
                q = 0
        ppa[:, k + 1] = np.array(a + b)
        div[:, k + 1] = np.array(d + e)
    return ppa, div


def report_long_matches(H, L, ppa=None, div=None):
    """
    Report matches longer than L in a collection of haplotypes H.
    ppa,div are a PBWT of the data. If None, it will be computed.
    """
    ## TODO : if ppa and div are None, do it on the fly
    if ppa is None or div is None:
        ppa, div = build_pbwt(H)

    H = np.asarray(H, dtype=np.short)
    ## M haplotypes over N markers
    M, N = H.shape
    matches = []

    for k in range(N):
        na = nb = 0
        i0 = 0
        if k < (N - 1):
            y = H[ppa[:, k], k + 1]
        else:
            y = np.ones(M, dtype=np.short)  # placeholder not used
        for i in range(0, M):
            if y[i] < 0:  ## missing data
                if i > 0:
                    y[i] = y[i - 1]  ## "impute" from closest match
                else:
                    y[i] = 0
            if div[i, k] > k:  # first 0 and first 1 seen, not in Durbin2014
                na = nb = 0
                i0 = i
            elif div[i, k] > (k - L) > 0:  # second condition is not in Durbin2014
                if (na > 0) and (nb > 0):
                    for ia in range(i0, i):
                        dmin = 0
                        for ib in range(ia + 1, i):
                            if div[ib, k] > dmin:
                                dmin = div[ib, k]
                            if (k == N - 1) or y[ib] != y[ia]:
                                id1, id2 = (
                                    (ppa[ia, k] < ppa[ib, k])
                                    and (ppa[ia, k], ppa[ib, k])
                                    or (ppa[ib, k], ppa[ia, k])
                                )
                                # report match, we could yield here
                                assert (k - dmin + 1) > L
                                matches.append(IBD_track(id1, id2, dmin, k))
                na = nb = 0
                i0 = i
            if y[i] == 0:
                na += 1
            else:
                nb += 1
            if k == N - 1:  ## deal with last position not in Durbin2014
                na += 1
                nb += 1
            ## last haplotype block is not treated in Durbin2014
            if i == M - 1:
                if (na > 0) and (nb > 0):
                    for ia in range(i0, i):
                        dmin = 0
                        for ib in range(ia + 1, i + 1):
                            if div[ib, k] > dmin:
                                dmin = div[ib, k]
                            if dmin < k - L + 1:  ## match
                                id1, id2 = (
                                    (ppa[ia, k] < ppa[ib, k])
                                    and (ppa[ia, k], ppa[ib, k])
                                    or (ppa[ib, k], ppa[ia, k])
                                )
                                assert (k - dmin + 1) > L
                                # report match, we could yield here
                                matches.append(IBD_track(id1, id2, dmin, k))
    return matches


class OriginTracer:
    """Class Implementing routines to trace origins down a pedigree"""

    def __init__(self):
        self.norigins = 0
        self.origins = defaultdict(self._new_origin)
        self.Lmatch = 5

    def _new_origin(self):
        """Return a new origin code, increments number of origins"""
        self.norigins += 1
        return self.norigins

    def run(self, phaser_db, L=None):
        """Run an analysis on phaser object"""
        if L is not None:
            self.Lmatch = int(L)
        phaser = family_phaser.Phaser.load(phaser_db)
        self.trace_origins(phaser)
        self.ped_grm(phaser)
        ## LD based analysis
        self.get_founder_origins(phaser)
        self.trace_ancestral_origins(phaser)
        self.grm(phaser)

    def get_founder_origins(self, phaser):
        logger.info("Reconstructing founder haplotypes")
        smpidx = {}
        for i, name in enumerate(phaser.data["samples"]):
            smpidx[name] = i
        phaser.data.create_group("founders", overwrite=True)
        for reg in phaser.regions:
            or_z = phaser.data["founders"].create_group(reg)
            logger.info(f"Working on region {reg}")
            logger.info("Grabbing data")
            origins = np.array(phaser.data[f"linkage/{reg}/origins"])
            gametes = np.array(phaser.data[f"phases/{reg}/gametes"])
            founder_haps = -1 * np.ones(
                (self.norigins, origins.shape[-1]), dtype=np.int8
            )
            hapcounts = np.zeros_like(founder_haps)
            hapdepths = np.zeros_like(hapcounts)
            logger.info("Diving in")
            ## We loop through all individuals to impute missing data
            ## at founding haplotypes
            for node in phaser.pedigree:
                # fmt: off
                node_idx = smpidx[node.indiv]
                orig = origins[node_idx]
                gam = gametes[node_idx]
                ## first gamete
                cols = np.where(gam[0,] > -1)[0]
                rows = np.array(orig[0,] - 1)[cols]
                hapcounts[rows, cols] += gam[0, cols]
                hapdepths[rows, cols] += 1
                ## second gamete
                cols = np.where(gam[1,] > -1)[0]
                rows = np.array(orig[1,] - 1)[cols]
                hapcounts[rows, cols] += gam[1, cols]
                hapdepths[rows, cols] += 1
                # fmt: on
            sub = np.where(hapdepths > 0)
            founder_haps[sub] = np.where((hapcounts[sub] / hapdepths[sub]) > 0.5, 1, 0)
            or_z["haplotypes"] = founder_haps
            logger.info(f"Done reconstructing founder haplotypes for region {reg}")
            logger.info(f"Getting IBD tracks  of Length >= {self.Lmatch} SNPs")
            ## TODO : set L depending on marker density and cM length
            ibd = report_long_matches(founder_haps, L=self.Lmatch)
            logger.info(f"Reconstructing ancestral origins from IBD tracks")
            anc_origins = np.repeat(range(self.norigins), founder_haps.shape[1])
            anc_origins = anc_origins.reshape(self.norigins, founder_haps.shape[1])
            ## note that by construction id2>id1 and tracks are sorted by
            ## id1, id2, and track.start
            for track in sorted(ibd):
                anc_origins[track.id2, track.start : (track.end + 1)] = anc_origins[
                    track.id1, track.start : (track.end + 1)
                ]
            or_z["origins"] = anc_origins

    def trace_ancestral_origins(self, phaser):
        logger.info("Tracing Ancestral Origins down the pedigree")
        smpidx = {}
        for i, name in enumerate(phaser.data["samples"]):
            smpidx[name] = i
        for reg in phaser.regions:
            logger.info(f"Working on region {reg}")
            segregations = np.array(phaser.data[f"phases/{reg}/segregations"])
            origins = np.zeros_like(segregations, dtype=int)
            anc_origins = np.array(phaser.data[f"founders/{reg}/origins"])
            for node in phaser.pedigree:
                node_idx = smpidx[node.indiv]
                if node.father is None:
                    orig_0 = self.origins[node.indiv + "_p"]
                    origins[
                        node_idx,
                        0,
                    ] = anc_origins[orig_0 - 1]
                else:
                    fa_idx = smpidx[node.father.indiv]
                    rows = segregations[
                        node_idx,
                        0,
                    ]
                    cols = np.arange(rows.shape[0])
                    origins[
                        node_idx,
                        0,
                    ] = origins[fa_idx, rows, cols]
                if node.mother is None:
                    orig_1 = self.origins[node.indiv + "_m"]
                    origins[
                        node_idx,
                        1,
                    ] = anc_origins[orig_1 - 1]
                else:
                    mo_idx = smpidx[node.mother.indiv]
                    rows = segregations[
                        node_idx,
                        1,
                    ]
                    cols = np.arange(rows.shape[0])
                    origins[
                        node_idx,
                        1,
                    ] = origins[mo_idx, rows, cols]
            phaser.data[f"linkage/{reg}/ancestral_origins"] = origins
        logger.info("Origins done")

    def trace_origins(self, phaser):
        logger.info("Tracing Origins down the pedigree")
        smpidx = {}
        for i, name in enumerate(phaser.data["samples"]):
            smpidx[name] = i
        phaser.data.create_group("linkage", overwrite=True)
        for reg in phaser.regions:
            logger.info(f"Working on region {reg}")
            logger.info("Grabbing data")
            segregations = np.array(phaser.data[f"phases/{reg}/segregations"])
            logger.info("Diving in")
            or_z = phaser.data["linkage"].create_group(reg)
            origins = np.zeros_like(segregations, dtype=int)
            for node in phaser.pedigree:
                node_idx = smpidx[node.indiv]
                if node.father is None:
                    orig_0 = self.origins[node.indiv + "_p"]
                    origins[
                        node_idx,
                        0,
                    ] = orig_0
                else:
                    fa_idx = smpidx[node.father.indiv]
                    rows = segregations[
                        node_idx,
                        0,
                    ]
                    cols = np.arange(rows.shape[0])
                    origins[
                        node_idx,
                        0,
                    ] = origins[fa_idx, rows, cols]
                if node.mother is None:
                    orig_1 = self.origins[node.indiv + "_m"]
                    origins[
                        node_idx,
                        1,
                    ] = orig_1
                else:
                    mo_idx = smpidx[node.mother.indiv]
                    rows = segregations[
                        node_idx,
                        1,
                    ]
                    cols = np.arange(rows.shape[0])
                    origins[
                        node_idx,
                        1,
                    ] = origins[mo_idx, rows, cols]
            or_z["origins"] = origins
        logger.info("Origins done")
        # Output global results to txt files
        with open(phaser.prefix + "_yapp_ancestors.txt", "w") as fout:
            print("ancestor\tgamete\tcode", file=fout)
            for k in self.origins:
                name, o = k.split("_")
                print(f"{name}\t{o}\t{self.origins[k]}", file=fout)
        with open(phaser.prefix + "_yapp_ancestral_props.txt", "w") as fout:
            ancprop = defaultdict(lambda: defaultdict(int))
            Atot = 0
            for reg in phaser.regions:
                origins = np.array(phaser.data[f"linkage/{reg}/origins"])
                Atot += 2 * origins.shape[2]
                for node in phaser.pedigree:
                    node_idx = smpidx[node.indiv]
                    for anc_all in origins[
                        node_idx,
                        0,
                    ]:
                        ancprop[node.indiv][anc_all] += 1
                    for anc_all in origins[
                        node_idx,
                        1,
                    ]:
                        ancprop[node.indiv][anc_all] += 1
            print("individual\tcode\tproportion", file=fout)
            for node in phaser.pedigree:
                for anc_all in self.origins.values():
                    if ancprop[node.indiv][anc_all] > 0:
                        print(
                            f"{node.indiv}\t{anc_all}\t{ancprop[node.indiv][anc_all]/Atot}",  # noqa
                            file=fout,
                        )

    def ped_grm(self, phaser):
        """Compute Genomic Relationship Matrix based on ancestral allele
        transmissions in the pedigree.
        """
        logger.info("Computing linkage-based GRM")
        samples = list(phaser.data["samples"])
        N = len(samples)
        Ltot = 0
        GRM = np.zeros((N, N), dtype=float)
        for reg in phaser.regions:
            logger.info(f"\tAccumulate region {reg}")
            origins = np.array(phaser.data[f"linkage/{reg}/origins"])
            L = origins.shape[2]
            compute_grm(N, L, origins, GRM)
            Ltot += L
        GRM /= 2 * Ltot

        logger.info("writing GRM to disk")
        with open(phaser.prefix + "_yapp_pedGRM.txt", "w") as fout:
            print("id1 id2 kinship", file=fout)
            for i in range(N):
                for j in range(i, N):
                    if GRM[i, j] > 0:
                        print(samples[i], samples[j], GRM[i, j], file=fout)
        for i in range(2, N):
            for j in range(1, i):
                GRM[i, j] = GRM[j, i]
        phaser.data["linkage/pedGRM"] = GRM

    def grm(self, phaser):
        """Compute GRM based on ancestral origins and pedigree"""
        logger.info("Computing LDLA-based GRM")
        samples = list(phaser.data["samples"])
        N = len(samples)
        Ltot = 0
        GRM = np.zeros((N, N), dtype=float)
        for reg in phaser.regions:
            logger.info(f"\tAccumulate region {reg}")
            origins = np.array(phaser.data[f"linkage/{reg}/ancestral_origins"])
            L = origins.shape[2]
            compute_grm(N, L, origins, GRM)
            Ltot += L
        GRM /= 2 * Ltot

        logger.info("writing GRM to disk")
        with open(phaser.prefix + "_yapp_GRM.txt", "w") as fout:
            print("id1 id2 kinship", file=fout)
            for i in range(N):
                for j in range(i, N):
                    if GRM[i, j] > 0:
                        print(samples[i], samples[j], GRM[i, j], file=fout)
        for i in range(2, N):
            for j in range(1, i):
                GRM[i, j] = GRM[j, i]
        phaser.data["linkage/GRM"] = GRM


def main(args):
    prfx = args.prfx
    phaser_db = prfx + "_yapp.db"
    tracer = OriginTracer()
    tracer.run(phaser_db)
