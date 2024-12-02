# -*- coding: utf-8 -*-
# pylint: disable=C0301,C0103,C0116
"""
Yapp module family_phaser.py

Infer phases and segregation indicators from high density SNP data
in pedigrees.
"""
import sys
import logging
from collections import defaultdict
import time
from multiprocessing import Pool, cpu_count, TimeoutError  # noqa pylint: disable=W0622
import bz2
import pickle
import numpy as np
from scipy.stats import binom
import cyvcf2
from . import vcf, gamete, pedigree

logger = logging.getLogger(__name__)


class PhaseError(Exception):
    """Base class for Phase Errors"""


def qerr(n, p, q=0.001):
    """(1-q) Quantile of the binomial(n,p)"""
    return binom(p=p, n=n).isf(q)


def wcsp_phase(args):
    """parallel WCSP function"""
    indiv, p, children_gametes, recpos = args
    wcsp_gam = gamete.Gamete.from_wcsp_solver(p.g, children_gametes, mkpos=recpos)
    return (indiv, wcsp_gam)


def run_segregation_task(args):
    """parallele segregation function"""
    name, chpair_parent, gam, recmap, err = args
    return name, chpair_parent.get_segregation_indicators(gam, recmap, err)


class ChromosomePair:
    """A class to manipulare a pair of chromosomes"""

    def __init__(self, genotype):
        self.g = gamete.Gamete.valid_genotype(genotype)
        self.H = [
            gamete.Gamete.from_genotype(self.g),
            gamete.Gamete.from_genotype(self.g),
        ]
        self._h_pat = 0
        self.si_pat = [(0, 0.5) for _ in range(len(self.g))]
        self.si_mat = [(0, 0.5) for _ in range(len(self.g))]

    @classmethod
    def from_phase_data(
        cls, genotype, paternal_hap, maternal_hap, paternal_si, maternal_si
    ):
        obj = cls(genotype)
        paternal_gamete = gamete.Gamete(paternal_hap)
        obj.update_paternal_gamete(paternal_gamete)
        maternal_gamete = gamete.Gamete(maternal_hap)
        obj.update_maternal_gamete(maternal_gamete)
        assert len(paternal_si) == obj.len
        assert len(maternal_si) == obj.len
        obj.si_pat = np.array(paternal_si, dtype=int)
        obj.si_mat = np.array(maternal_si, dtype=int)
        return obj

    @property
    def len(self):
        return len(self.g)

    @property
    def het_mks(self):
        "Heterozygous markers"
        return self.g == 1

    @property
    def nhet(self):
        return np.sum(self.het_mks)

    @property
    def phased(self):
        """Phased markers"""
        return (self.paternal_gamete.haplotype > -1) & (
            self.maternal_gamete.haplotype > -1
        )  # noqa

    @property
    def resolved(self):
        """Resolved markers = heterozygotes and phased"""
        return self.het_mks & self.phased

    @property
    def nresolved(self):
        return np.sum(self.resolved)

    @property
    def h_pat(self):
        """Index (0,1) of the paternal haplotype"""
        return self._h_pat

    @h_pat.setter
    def h_pat(self, value):
        assert value == 0 or value == 1
        self._h_pat = value

    @property
    def h_mat(self):
        """Index (0,1) of the maternal haplotype"""
        return 1 - self._h_pat

    @h_mat.setter
    def h_mat(self, value):
        self.h_pat = 1 - value

    @property
    def paternal_gamete(self):
        return self.H[self.h_pat]

    @property
    def maternal_gamete(self):
        return self.H[self.h_mat]

    def get_phase_info(self, origin, call=0.999):
        """Return informative phase in parent origin (0:father, 1:
        mother) = segregation indicator truncated at uninformative
        boundaries.
        """
        assert (origin == 0) or (origin == 1)
        segind = self.si_pat if (origin == 0) else self.si_mat
        phase_info = np.full_like(self.g, -1, dtype=np.int8)
        for i, (a, p) in enumerate(segind):
            if self.phased[i] and p > call:
                phase_info[i] = a
        return phase_info

    def get_transmitted_from_segregation(self, si, call=0.999):
        """
        Returns the gamete transmitted from a segregation indicator vector.
        Inform haplotype only if posterior probability > call
        """
        gam = gamete.Gamete.from_genotype(self.g)
        for i, (orig, prob) in enumerate(si):
            assert orig == 1 or orig == 0
            if prob < call:
                continue
            if orig == 0:
                allele = self.h_pat
            elif orig == 1:
                allele = self.h_mat
            else:
                continue
            gam.haplotype[i] = self.H[allele].haplotype[i]
        return gam

    def get_segregation_indicators(self, gam, recmap, err=1e-2):
        """Infer origin of alleles in the gamete gam using a simple HMM.

        Arguments
        ---------
        gam : gamete.Gamete
           the gamete transmitted
        recmap : array of floats
           the recombination rates between adjacent markers
        err : float
           Genotyping error rate

        Transition probabilities are:
        [ [ (1-r)   r   ]
          [   r   (1-r) ] ... ] for r in recmap
          between all marker pairs

        Returns
        -------
        List( tuple( int, float) )
        segregation indicators for all markers with associated probability
        """
        assert type(gam) == gamete.Gamete
        assert len(gam.haplotype) == self.len
        assert len(recmap) == self.len - 1

        # 1. HMM Setup
        transitions = np.array(
            [np.identity(2) * (1 - x) + (1 - np.identity(2)) * x for x in recmap]
        )
        # [ P(gam[m]| S==pat[m]), P(gam[m]| S==mat[m])]
        emissions = np.ones((self.len, 2), dtype=float)
        for m in range(self.len):
            if (
                (gam.haplotype[m] > -1)
                and self.g[m] > -1
                and self.H[0].haplotype[m] > -1
                and self.H[1].haplotype[m] > -1
            ):
                # ignore mendelian errors
                if (gam.haplotype[m] == 0 and self.g[m] == 2) or (
                    gam.haplotype[m] == 1 and self.g[m] == 0
                ):
                    emissions[m, 0] = 1
                    emissions[m, 1] = 1
                else:
                    emissions[m, 0] = (
                        (gam.haplotype[m] == self.paternal_gamete.haplotype[m])
                        and (1 - err)
                        or err
                    )  # noqa
                    emissions[m, 1] = (
                        (gam.haplotype[m] == self.maternal_gamete.haplotype[m])
                        and (1 - err)
                        or err
                    )  # noqa
        # 2. Forward-Backward algorithm
        fwd = np.ones((self.len, 2), dtype=float)  # Forward
        rew = np.ones((self.len, 2), dtype=float)  # Backward
        sca = np.ones(self.len, dtype=float)  # scaling

        # Compute forward probabilities
        fwd[0, 0] = 0.5 * emissions[0, 0]
        fwd[0, 1] = 0.5 * emissions[0, 1]
        sca[0] = 1.0 / (fwd[0, 0] + fwd[0, 1])
        fwd[
            0,
        ] *= sca[0]
        for m in range(1, self.len):
            fwd[m, 0] = (
                fwd[m - 1, 0] * transitions[m - 1, 0, 0]
                + fwd[m - 1, 1] * transitions[m - 1, 1, 0]
            ) * emissions[m, 0]
            fwd[m, 1] = (
                fwd[m - 1, 0] * transitions[m - 1, 0, 1]
                + fwd[m - 1, 1] * transitions[m - 1, 1, 1]
            ) * emissions[m, 1]
            sca[m] = 1.0 / (fwd[m, 0] + fwd[m, 1])
            fwd[
                m,
            ] *= sca[m]
        # Compute backward probabilities
        rew[
            self.len - 1,
        ] /= sca[m - 1]
        for m in range(self.len - 1, 0, -1):
            rew[m - 1, 0] = (
                transitions[m - 1, 0, 0] * emissions[m, 0] * rew[m, 0]
                + transitions[m - 1, 0, 1] * emissions[m, 1] * rew[m, 1]
            )
            rew[m - 1, 1] = (
                transitions[m - 1, 1, 0] * emissions[m, 0] * rew[m, 0]
                + transitions[m - 1, 1, 1] * emissions[m, 1] * rew[m, 1]
            )
            rew[
                m - 1,
            ] *= sca[m - 1]
        # loglik = -np.sum(np.log(sca))
        # Compute Posterior probabilities
        post_si = fwd * rew
        post_si /= np.sum(post_si, axis=1, keepdims=True)  # required ?

        # 3. Viterbi Algorithm
        # viterbi variables
        delta = np.zeros((self.len, 2), dtype=float)
        psi = np.zeros((self.len, 2), dtype=int)
        soluce = np.empty(self.len, dtype=int)
        # init
        delta[0, 0] = np.log(0.5 * emissions[0, 0])
        delta[0, 1] = np.log(0.5 * emissions[0, 1])
        # recursion
        for m in range(1, self.len):
            val_0 = [
                delta[m - 1, 0] + np.log(transitions[m - 1, 0, 0]),
                delta[m - 1, 1] + np.log(transitions[m - 1, 1, 0]),
            ]
            val_1 = [
                delta[m - 1, 0] + np.log(transitions[m - 1, 0, 1]),
                delta[m - 1, 1] + np.log(transitions[m - 1, 1, 1]),
            ]
            psi[m, 0] = np.argmax(val_0)
            psi[m, 1] = np.argmax(val_1)
            delta[m, 0] = val_0[psi[m, 0]] + np.log(emissions[m, 0])
            delta[m, 1] = val_1[psi[m, 1]] + np.log(emissions[m, 1])
        # termination / backtracking
        soluce[-1] = np.argmax(
            delta[
                -1,
            ]
        )
        for m in range(self.len - 1, 0, -1):
            soluce[m - 1] = psi[m, soluce[m]]
        result = [(x, post_si[i, x]) for i, x in enumerate(soluce)]
        return result

    def update_paternal_gamete(self, prop_gam):
        return self.update_gamete(prop_gam, 0)

    def update_maternal_gamete(self, prop_gam):
        return self.update_gamete(prop_gam, 1)

    def update_unknown_gamete(self, prop_gam, err, q=1e-3):
        """Update the phase when the origin of the prop_gam is not known.
        Tries to update paternal and maternal and keeps the one that leads
        to the fewest # of mismatches.

        """
        misses_pat, new_gam_p = gamete.Gamete.combine(self.paternal_gamete, prop_gam)
        misses_mat, new_gam_m = gamete.Gamete.combine(self.maternal_gamete, prop_gam)
        if min([len(misses_pat), len(misses_mat)]) > qerr(self.nhet, err, q):
            logger.debug(
                f"[update_unknown_gamete] "
                f"{len(misses_pat)} "
                f"{len(misses_mat)} "
                f"{self.nhet}"
            )
            raise PhaseError("Could not resolve origin of proposed gamete")
        if len(misses_mat) < len(misses_pat):
            return self.update_maternal_gamete(prop_gam)
        else:
            return self.update_paternal_gamete(prop_gam)

    def update_gamete(self, prop_gam, parent):
        """Update the gamete of a parent with new_gam.
        Arguments
        ---------
        - prop_gam : Gamete object
           the proposed gamete information
        - parent : int
           0 : update paternal, 1: update maternal
        Returns
        -------
        - [ int, int]
          number of mismatches found in [paternal, maternal] gametes
        """
        nmiss = [0, 0]
        if parent == 0:
            origin = self.h_pat
        elif parent == 1:
            origin = self.h_mat
        else:
            raise ValueError(f"{type(self)}:[update_gamete]: parent must be 0 or 1")
        misses_origin, new_gam = gamete.Gamete.combine(self.H[origin], prop_gam)
        nmiss[origin] = len(misses_origin)
        self.H[origin] = new_gam
        prop_gam_o = gamete.Gamete.complement(self.H[origin], self.g)
        misses_other, new_gam = gamete.Gamete.combine(self.H[1 - origin], prop_gam_o)
        nmiss[1 - origin] = len(misses_other)
        self.H[1 - origin] = new_gam
        return nmiss


class Phaser:
    def __init__(
        self, vcf_file, ped_file, out_prfx, region=None, focalID=None, err=1e-3, rho=1
    ):
        self.prefix = out_prfx
        self.err = err
        self.rho = rho
        if region is not None:
            self.prefix += "_" + region
        if focalID is not None:
            self.prefix += "_" + focalID
        # individuals
        ped = pedigree.Pedigree.from_fam_file(ped_file)
        ped.del_unrelated()
        if focalID is not None:
            ped = ped.get_family(focalID)
        self.ped_file_name = ped_file
        pedindivs = [x.indiv for x in ped]
        # genotype data
        self.data = vcf.vcf2zarr(
            vcf_file, output_prefix=self.prefix, reg=region, samples=pedindivs
        )
        self.vcf_out_file_name = f"{self.prefix}_phased.vcf.gz"
        self.vcf_file_name = f"{out_prfx}.vcf.gz"
        genosmp = defaultdict(lambda: False)
        for g in self.genotyped_samples:
            genosmp[g] = True
        self.ignored_indivs = []
        for indiv in pedindivs:
            if not genosmp[indiv]:
                rm_node = ped.del_indiv(indiv)
                logger.debug(f"Ignoring {indiv} as it is not genotyped")
                self.ignored_indivs.append(rm_node)
        self.pedigree = ped
        self.relations = self.pedigree.to_tuples()
        if len(self.relations) > sys.getrecursionlimit():
            sys.setrecursionlimit(len(self.relations)*2)
        # Run parameters
        self.timeout_delay = 30

    @classmethod
    def from_prefix(cls, prfx, region=None, focalID=None, err=1e-3, rho=1):
        """
        Create a phaser object from files with the same prefix :
        prfx.vcf.gz and prfx.fam

        Arguments
        ---------

        prfx : str
           Prefix of input files
        **kwargs : arguments
           Optional arguments to pass to yapp.vcf.vcf2fph
        """
        vcf_file = f"{prfx}.vcf.gz"
        fam_file = f"{prfx}.fam"
        obj = cls(
            vcf_file, fam_file, prfx, region=region, focalID=focalID, err=err, rho=rho
        )
        print(obj.data.tree())
        return obj

    @property
    def regions(self):
        return self.data["regions"]

    @property
    def genotyped_samples(self):
        return list(self.data["samples"])

    @property
    def pedigree_samples(self):
        return [x for x in self.pedigree.nodes]

    @property
    def families(self):
        return self.pedigree.families

    def recmap(self):
        """Computes a recombination map assuming a rate of recrate (in cM/Mb)
        along regions"""
        res = {}
        for reg in self.regions:
            pos = self.data["variants"][reg]["POS"]
            distances = np.array(pos[1:] - pos[:-1])
            distances[distances < 100] = 100
            res[reg] = distances * self.rho * 1e-8
        return res

    def write_phased_vcf(self):
        """Write phase information in a VCF file"""
        pedindivs = [x.indiv for x in self.pedigree]
        vcf_tmpl = cyvcf2.VCF(self.vcf_file_name, lazy=True, samples=pedindivs)
        w = cyvcf2.Writer(self.vcf_out_file_name, vcf_tmpl)
        snp_mapping = {}
        for s in vcf_tmpl:
            snp_mapping[s.ID] = s
        nidx = np.array([vcf_tmpl.samples.index(n.indiv) for n in self.pedigree])
        for reg in self.regions:
            try:
                snps = self.data["variants"][reg]["ID"]
            except KeyError:
                continue
            variants = [snp_mapping[x] for x in snps]
            ph = np.array(self.data["phases"][reg]["gametes"])
            sel = np.zeros((ph.shape[0], ph.shape[2]), dtype=bool)
            sel = np.where(ph[:, 0, :] > -1, True, False)
            for i, v in enumerate(variants):
                ph_genotypes = v.genotypes[:]
                loc_nidx = nidx[sel[nidx, i]]
                for idx in loc_nidx:
                    ph_genotypes[idx] = [ph[idx, 0, i], ph[idx, 1, i], 1]
                v.genotypes = [list(x) for x in ph_genotypes]
                w.write_record(v)

    ## Make the class pickable even in case of pedigree loops
    def __getstate__(self):
        ## deal with unpickable pedigree
        state = self.__dict__.copy()
        del state["pedigree"]
        return state

    def __setstate__(self, state):
        ## recover pedigree after pickling
        self.__dict__.update(state)
        self.pedigree = pedigree.Pedigree.from_tuples(self.relations)

    def save(self, fname=None):
        """Save Phaser internal state to file"""
        if fname is None:
            fname = f"{self.prefix}_yapp.db"
        dbfile = bz2.BZ2File(fname, "wb")
        try:
            pickle.dump(self, dbfile)
        except RecursionError:
            print(f"Error with recursion limit:{sys.getrecursionlimit()}")
            raise
        dbfile.close()

    @classmethod
    def load(cls, fname):
        with bz2.BZ2File(fname, "rb") as f:
            obj = pickle.load(f)
        obj.__class__ = cls
        return obj

    def get_genotypes(self, region):
        data = {}
        genotypes = np.array(self.data["genotypes"][region])
        for i, name in enumerate(self.data["samples"]):
            data[name] = genotypes[
                i,
            ]
        return data

    def get_mendelian_genotypes(self, region):
        genotypes = self.get_genotypes(region)
        for node in self.pedigree:
            pg = genotypes[node.indiv]
            for child in node.children:
                cg = genotypes[child.indiv]
                errors = ((pg == 2) & (cg == 0)) | ((pg == 0) & (cg == 2))
                pg[errors] = -1
                cg[errors] = -1
        return genotypes

    def run(self, ncpu=cpu_count()):
        self.data.create_group("phases")
        with open(self.prefix + "_yapp_phasestats.txt", "w") as fstat:
            print(
                "Individual Sex Generation Nparents \
                Nchildren region het resolved",
                file=fstat,
            )
        for reg in self.regions:
            logger.info(f"Working on region {reg}")
            try:
                phases, ignored = self.phase_samples_from_genotypes(reg, ncpu=ncpu)
            except KeyError:
                logger.info(f"No data on region {reg}")
                continue
            nhet = 0
            nresolved = 0
            for p in phases.values():
                nhet += p.nhet
                nresolved += p.nresolved
            logger.info(
                f"[{reg}] Resolved {nresolved} out of {nhet} phases : "
                f"{100*nresolved/nhet:.1f}%"
            )
            sys.stdout.flush()
            phases = self.phase_samples_from_segregations(
                reg, ncpu=ncpu, phases=phases, ignore_child=ignored
            )
            nhet = 0
            nresolved = 0
            for p in phases.values():
                nhet += p.nhet
                nresolved += p.nresolved
            logger.info(
                f"[{reg}] Resolved {nresolved} out of {nhet} phases : "
                f"{100*nresolved/nhet:.1f}%"
            )
            # Final estimation of segregation indicators
            # once phases are inferred
            _ = self.compute_segregations(reg, phases=phases)

            logger.info("Storing results")
            with open(self.prefix + "_yapp_phasestats.txt", "a") as fstat:
                for node in self.pedigree:
                    p = phases[node.indiv]
                    print(
                        f"{node.indiv}  "
                        f"{node.sex} "
                        f"{node.gen} "
                        f"{(node.father!=None)+(node.mother!=None)} "
                        f"{len(node.children)} "
                        f"{reg} "
                        f"{p.nhet} "
                        f"{p.nresolved}",
                        file=fstat,
                    )

            nind, nsnp = self.data["genotypes"][reg].shape
            ph_g = self.data["phases"].create_group(reg)
            gametes = np.full((nind, 2, nsnp), dtype="int8", fill_value=-2)
            segregations = np.full((nind, 2, nsnp), dtype="int8", fill_value=0)
            seg_probs = np.full(shape=(nind, 2, nsnp), dtype="float", fill_value=0.5)
            for i, name in enumerate(self.data["samples"]):
                try:
                    chpair = phases[name]
                except KeyError:
                    continue
                gametes[i, 0] = chpair.paternal_gamete.haplotype
                gametes[i, 1] = chpair.maternal_gamete.haplotype
                segregations[i, 0] = np.array([x[0] for x in chpair.si_pat])
                segregations[i, 1] = np.array([x[0] for x in chpair.si_mat])
                seg_probs[i, 0] = np.array([x[1] for x in chpair.si_pat])
                seg_probs[i, 1] = np.array([x[1] for x in chpair.si_mat])
            ph_g["gametes"] = gametes
            ph_g["segregations"] = segregations
            ph_g["seg_probs"] = seg_probs
        print(self.data.tree())

    def compute_segregations(self, region, phases=None):
        """Compute segregations for all individuals in the pedigree"""
        logger.info(f"Computing segregations on region {region}")
        recmaps = self.recmap()
        if phases is None:
            chrom_pairs = self.phase_samples_from_genotypes(region)
        else:
            chrom_pairs = phases
        recmap = recmaps[region]
        pat_seg_tasks = []
        mat_seg_tasks = []
        for node in self.pedigree:
            chpair = chrom_pairs[node.indiv]
            if node.father:
                pat_seg_tasks.append(
                    (
                        node.indiv,
                        chrom_pairs[node.father.indiv],
                        chpair.paternal_gamete,
                        recmap,
                        self.err,
                    )
                )
            if node.mother:
                mat_seg_tasks.append(
                    (
                        node.indiv,
                        chrom_pairs[node.mother.indiv],
                        chpair.maternal_gamete,
                        recmap,
                        self.err,
                    )
                )
        with Pool(cpu_count()) as workers:
            for indiv, segind in workers.imap_unordered(
                run_segregation_task, pat_seg_tasks
            ):
                chpair = chrom_pairs[node.indiv]
                chpair.si_pat = segind
            for indiv, segind in workers.imap_unordered(
                run_segregation_task, mat_seg_tasks
            ):
                chpair = chrom_pairs[node.indiv]
                chpair.si_mat = segind
        return phases

    def phase_samples_from_segregations(
        self,
        region,
        ncpu=1,
        phases=None,
        ignore_child=defaultdict(lambda: False),  # noqa
    ):
        """Infer segregation indicators in the pedigree.

        Arguments
        ---------
        recrate : recombination rate in cM/Mb

        Returns
        -------
        update phases
        """
        logger.info("Phasing samples from segregations")
        recmaps = self.recmap()
        if phases is None:
            chrom_pairs, ignore_child = self.phase_samples_from_genotypes(region)
        else:
            chrom_pairs = phases

        recmap = recmaps[region]
        recpos = np.cumsum([0] + list(recmap)) * 100
        logger.info("\t1. Infer Segregation Indicators")
        pat_seg_tasks = []
        mat_seg_tasks = []
        for node in self.pedigree:
            if ignore_child[node.indiv]:
                logger.debug(f"Ignoring {node.indiv} as requested")
                continue
            chpair = chrom_pairs[node.indiv]
            if node.father:
                pat_seg_tasks.append(
                    (
                        node.indiv,
                        chrom_pairs[node.father.indiv],
                        chpair.paternal_gamete,
                        recmap,
                        self.err,
                    )
                )
            if node.mother:
                mat_seg_tasks.append(
                    (
                        node.indiv,
                        chrom_pairs[node.mother.indiv],
                        chpair.maternal_gamete,
                        recmap,
                        self.err,
                    )
                )

        nmm = 0
        with Pool(cpu_count()) as workers:
            for indiv, segind in workers.imap_unordered(
                run_segregation_task, pat_seg_tasks
            ):
                node = self.pedigree.nodes[indiv]
                chpair_p = chrom_pairs[node.father.indiv]
                chpair = chrom_pairs[node.indiv]
                chpair.si_pat = segind
                maxprob = max([x[1] for x in segind])
                if maxprob < 0.999:
                    logger.debug(
                        f"Ignoring {node.indiv} "
                        f"as segregations cannot be resolved confidently"
                    )
                    ignore_child[node.indiv] = True
                else:
                    old_gam = chpair.paternal_gamete
                    new_gam = chpair_p.get_transmitted_from_segregation(chpair.si_pat)
                    nmiss = chpair.update_paternal_gamete(new_gam)
                    if (nmiss[0] + nmiss[1]) > qerr(
                        chpair.nhet * 2,
                        self.err,
                        q=1e-3 / (len(pat_seg_tasks) + len(mat_seg_tasks)),
                    ):  # noqa
                        logger.warning(
                            f"[from_seg] {node.father.indiv}[pat] -> {node.indiv} :"  # noqa
                            f"{50*(nmiss[0]+nmiss[1])/chpair.nhet:.1g} % mismatch"
                        )  # noqa
                        ignore_child[node.indiv] = True
                        chpair.update_paternal_gamete(old_gam)
                        nmm += 1
            for indiv, segind in workers.imap_unordered(
                run_segregation_task, mat_seg_tasks
            ):
                node = self.pedigree.nodes[indiv]
                chpair_m = chrom_pairs[node.mother.indiv]
                chpair = chrom_pairs[node.indiv]
                chpair.si_mat = segind
                maxprob = max([x[1] for x in segind])
                if maxprob < 0.999:
                    logger.debug(
                        f"Ignoring {node.indiv} "
                        f"as segregations cannot be resolved confidently"
                    )
                    ignore_child[node.indiv] = True
                else:
                    old_gam = chpair.maternal_gamete
                    new_gam = chpair_m.get_transmitted_from_segregation(chpair.si_mat)
                    nmiss = chpair.update_maternal_gamete(new_gam)
                    if (nmiss[0] + nmiss[1]) > qerr(
                        chpair.nhet * 2,
                        self.err,
                        q=1e-3 / (len(pat_seg_tasks) + len(mat_seg_tasks)),
                    ):  # noqa
                        logger.warning(
                            f"[from_seg] {node.mother.indiv}[mat] -> {node.indiv} :"  # noqa
                            f"{50*(nmiss[0]+nmiss[1])/chpair.nhet:.1g} % mismatch"
                        )  # noqa
                        ignore_child[node.indiv] = True
                        chpair.update_maternal_gamete(old_gam)
                        nmm += 1
        logger.info(
            f"\t\tFound {nmm} mismatches out of "
            f"{len(pat_seg_tasks)+len(mat_seg_tasks)} tasks"
        )
        logger.info("\t2. Update parental phases")
        wcsp_tasks = []
        for node in self.pedigree:
            p = chrom_pairs[node.indiv]

            children_gametes = {}
            for child in node.children:
                if ignore_child[child.indiv]:
                    logger.debug(f"Ignoring {child.indiv} as requested")
                    continue
                chpair = chrom_pairs[child.indiv]
                if child.father is node:
                    children_gametes[child.indiv] = chpair.paternal_gamete
                elif child.mother is node:
                    children_gametes[child.indiv] = chpair.maternal_gamete
            if len(children_gametes) > 0:
                wcsp_tasks.append((node.indiv, p, children_gametes, recpos))

        logger.info(f"\t\t Phasing {len(wcsp_tasks)} parents with  WCSP")
        remaining_guys = [x[0] for x in wcsp_tasks]
        with Pool(ncpu) as workers:
            results = workers.imap(wcsp_phase, wcsp_tasks)
            ntimeout = 0
            while True:
                try:
                    indiv, new_gam = results.next(timeout=self.timeout_delay)
                    p = chrom_pairs[indiv]
                    try:
                        p.update_unknown_gamete(new_gam, self.err, 1e-2)
                    except PhaseError:
                        logger.debug(
                            f"Inconsistency in phase data for individual"
                            f"{indiv} on region {region}. Will try to fix it."
                        )
                        si = p.get_segregation_indicators(new_gam, recmap, err=self.err)
                        si_gam = gamete.Gamete.from_offspring_segregation(
                            p.g, new_gam, [x[0] for x in si]
                        )
                        try:
                            chrom_pairs[indiv].update_unknown_gamete(
                                si_gam, self.err, 1e-3
                            )
                        except PhaseError:
                            logger.warning(
                                f"[from_seg] Unresolved Phasing inconsistency "
                                f"for individual {indiv} on region {region}. "
                            )
                        else:
                            logger.debug("Fixed it.")
                    remaining_guys.remove(indiv)
                except StopIteration:
                    break
                except TimeoutError:
                    if ntimeout > 5:
                        logger.warning(
                            f"[WCSP] Max delay attained :"
                            f"aborting {len(remaining_guys)} remaining tasks"
                        )
                        for guy in remaining_guys:
                            logger.info(f"WCSP : {guy} not analyzed")
                        break
                    else:
                        logger.warning(
                            f"[WCSP] Result time out : {len(remaining_guys)}"
                            f"tasks remaining, will wait a bit more ..."
                        )
                        ntimeout += 1
        for node in self.pedigree:
            p = chrom_pairs[node.indiv]
            logger.debug(
                f"[from_seg] {node.indiv} -- [ "
                f"sex:{node.sex} "
                f"gen:{node.gen} "
                f"par:{(node.father is not None)+(node.mother is not None)} "  # noqa
                f"off:{len(node.children)} "
                f"res: {p.nresolved}/{p.nhet}]"
            )

        return chrom_pairs

    def phase_samples_from_genotypes(self, region, ncpu=1, phases=None):
        """Reconstruct paternal and maternal gametes in the pedigree
        based on observed genotypes.
        Returns
        -------
        Dict[ str, Dict[ str, ChromosomePair ]]
            regions as keys, dict( name : ChromosomePair) as value
        """
        logger.info("Phasing samples from genotypes")
        logger.info("\t1. Loading Genotypes")
        genotypes = self.get_mendelian_genotypes(region)
        recmaps = self.recmap()
        recmap = recmaps[region]
        recpos = np.cumsum([0] + list(recmap)) * 100
        if phases is None:
            chrom_pairs = {}
            for node in self.pedigree:
                chrom_pairs[node.indiv] = ChromosomePair(genotypes[node.indiv])
        else:
            chrom_pairs = phases
        ignore_child = defaultdict(lambda: False)
        wcsp_tasks = []
        logger.info("\t2. Constructing gametes")
        for node in self.pedigree:
            name = node.indiv
            p = chrom_pairs[name]

            if node.father is not None:
                geno_p = genotypes[node.father.indiv]
                gam_p = gamete.Gamete.from_genotype(geno_p)
                nmiss = p.update_paternal_gamete(gam_p)
                if (nmiss[0] + nmiss[1]) > qerr(p.nhet * 2, self.err):
                    logger.warning(
                        f"[from_geno] {node.father.indiv}[pat] -> {node.indiv}:"  # noqa
                        f"{50*(nmiss[0]+nmiss[1])/p.nhet:.1g}% mismatches"
                    )
                elif nmiss[0] + nmiss[1] > 0:
                    logger.debug(
                        f"[from_geno] {node.father.indiv}[pat] -> {node.indiv}:"  # noqa
                        f"{50*(nmiss[0]+nmiss[1])/p.nhet:.1g}% mismatches"
                    )
            if node.mother is not None:
                geno_m = genotypes[node.mother.indiv]
                gam_m = gamete.Gamete.from_genotype(geno_m)
                nmiss = p.update_maternal_gamete(gam_m)
                if (nmiss[0] + nmiss[1]) > qerr(p.nhet * 2, self.err):
                    logger.warning(
                        f"[from_geno] {node.mother.indiv}[mat] -> {node.indiv}:"  # noqa
                        f"{50*(nmiss[0]+nmiss[1])/p.nhet:.1g}% mismatches"
                    )
                elif (nmiss[0] + nmiss[1]) > 0:
                    logger.debug(
                        f"[from_geno] {node.mother.indiv}[pat] -> {node.indiv}:"  # noqa
                        f"{50*(nmiss[0]+nmiss[1])/p.nhet:.1g}% mismatches"
                    )

            children_gametes = {}
            for child in node.children:
                geno_off = genotypes[child.indiv]
                if child.father is node:
                    if child.mother is not None:
                        geno_other = genotypes[child.mother.indiv]
                    else:
                        geno_other = None
                elif child.mother is node:
                    if child.father is not None:
                        geno_other = genotypes[child.father.indiv]
                    else:
                        geno_other = None
                gam_off = gamete.Gamete.from_offspring_genotype(
                    geno_off, other_geno=geno_other
                )
                children_gametes[child.indiv] = gam_off
            if len(node.children) > 0:
                wcsp_tasks.append((node.indiv, p, children_gametes, recpos))

        logger.info(f"\t\tPhasing {len(wcsp_tasks)} parents with WCSP")
        start_time = time.time()
        with Pool(ncpu) as workers:
            for indiv, new_gam in workers.imap(wcsp_phase, wcsp_tasks):
                p = chrom_pairs[indiv]
                try:
                    # be liberal in calling switch errors
                    p.update_unknown_gamete(new_gam, self.err, 1e-2)
                except PhaseError:
                    logger.debug(
                        f"Inconsistency in phase data "
                        f"for individual {indiv} on region {region}. "
                        f"I Will try to fix it. "
                    )
                    # correct switch errors
                    si = p.get_segregation_indicators(new_gam, recmap, err=self.err)
                    si_gam = gamete.Gamete.from_offspring_segregation(
                        p.g, new_gam, [x[0] for x in si]
                    )
                    try:
                        # admit a bit more inconsistencies to solve gamete
                        chrom_pairs[indiv].update_unknown_gamete(si_gam, self.err, 1e-3)
                    except PhaseError:
                        logger.warning(
                            f"Unresolved phasing inconsistency "
                            f"for individual {indiv} on region {region}. "
                        )
                    else:
                        logger.debug("Fixed it.")
        elapsed_time = time.time() - start_time
        tot_time = elapsed_time * ncpu
        if len(wcsp_tasks) > 0:
            time_per_task = tot_time / len(wcsp_tasks)
            self.timeout_delay = round(time_per_task) + 1
            logger.debug(f"Set timeout delay to {self.timeout_delay} seconds")
        for node in self.pedigree:
            p = chrom_pairs[node.indiv]
            logger.debug(
                f"[from_geno] {node.indiv} -- [ "
                f"sex:{node.sex} "
                f"gen:{node.gen} "
                f"par:{(node.father is not None)+(node.mother is not None)} "  # noqa
                f"off:{len(node.children)} "
                f"res: {p.nresolved}/{p.nhet}]"
            )

        return chrom_pairs, ignore_child


def main(args):
    prfx = args.prfx

    logger.info("Loading Data")
    phaser = Phaser.from_prefix(
        prfx, region=args.reg, focalID=args.focalID, err=args.err, rho=args.rho
    )
    phaser.run(ncpu=args.c)
    logger.info(
        f"Exporting results to : "
        f"{phaser.vcf_out_file_name} and {phaser.prefix}_yapp.db"
    )
    phaser.write_phased_vcf()
    phaser.save()
