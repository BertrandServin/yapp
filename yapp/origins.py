# -*- coding: utf-8 -*-
"""
Yapp module origins.py

Trace ancestral origins down a pedigree.
"""
import logging
from collections import defaultdict
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
            # outgrm[i, j] += (origins[i, 0, ell] == origins[j, 0, ell])
            # outgrm[i, j] += (origins[i, 1, ell] == origins[j, 0, ell])
            # outgrm[i, j] += (origins[i, 0, ell] == origins[j, 1, ell])
            # outgrm[i, j] += (origins[i, 1, ell] == origins[j, 1, ell])


class OriginTracer:
    """Class Implementing routines to trace origins down a pedigree"""

    def __init__(self):
        self.norigins = 0
        self.origins = defaultdict(self._new_origin)

    def _new_origin(self):
        """Return a new origin code, increments number of origins"""
        self.norigins += 1
        return self.norigins

    def run(self, phaser_db):
        """Run an analysis on phaser object"""
        phaser = family_phaser.Phaser.load(phaser_db)
        self.trace_origins(phaser)
        self.ped_grm(phaser)

    def trace_origins(self, phaser):
        logger.info("Tracing Origins down the pedigree")
        smpidx = {}
        for i, name in enumerate(phaser.data["samples"]):
            smpidx[name] = i
        try:
            phaser.data.create_group("linkage")
        except ContainsGroupError:
            logger.warning(
                "Seems Zarr archive already has linkage information: will use this"
            )  # noqa
            logger.warning(
                "If rebuilding linkage info is needed, delete the 'linkage' directory"
            )  # noqa
            return
        for reg in phaser.regions:
            logger.info(f"Working on region {reg}")
            logger.info("Grabbing data")
            segregations = np.array(phaser.data[f"phases/{reg}/segregations"])
            logger.info("Diving in")
            or_z = phaser.data["linkage"].create_group(reg)
            origins = np.zeros_like(segregations, dtype=np.int)
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
        GRM = np.zeros((N, N), dtype=np.float)
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


def main(args):
    prfx = args.prfx
    phaser_db = prfx + ".db"
    tracer = OriginTracer()
    tracer.run(phaser_db)
