# -*- coding: utf-8 -*-
"""
Module gamete.py from yapp

This module exports classes and functions to build and manipulate gametes
and their transmissions from genotype data.

This module works with genotype and haplotype data that are expected to be
(castable to) 1-D numpy arrays of integers.

For genotypes valid values (seel Gamete.valid_genotype static method) are:
    - -1 : missing data
    -  0 : homozygote AA
    -  1 : heterozygote (Aa)
    -  2 : homozygote aa

For haplotypes:
    - -2 : unknown allele
    - -1 : unresolved allele
    -  0 : for allele A
    -  1 : for allele a
"""
import logging
import numpy as np
from yapp import wcsp

logger = logging.getLogger(__name__)


class Gamete:
    """A gamete = an haplotype transmitted during meiosis"""

    def __init__(self, hap=None):
        self.haplotype = hap

    @staticmethod
    def valid_genotype(genotype):
        try:
            geno = np.array(genotype, dtype=int)
        except ValueError:
            print("Genotype must be castable to a numpy array of integers")
            raise
        else:
            try:
                assert len(geno.shape) == 1
            except AssertionError:
                raise ValueError("Genotype must be (castable to) a 1-D numpy array.")
        try:
            assert np.all((geno > -2) & (geno < 3))
        except AssertionError:
            raise AssertionError("Genotype values must be in [-1,0,1,2]")
        return geno

    def __str__(self):
        return " ".join([f"{x<0 and ' -' or x:2}" for x in self.haplotype])

    @classmethod
    def from_genotype(cls, genotype):
        """
        Creates a gamete from a genotype.

        Arguments
        ---------
        - genotype : sequence
           A diploid genotype with values

        Returns
        -------
        A Gamete object with haplotype of the same length as genotype.
        Haplotype values are :
        """

        gam = cls()
        geno = cls.valid_genotype(genotype)
        gam.haplotype = np.full_like(geno, -2)
        gam.haplotype[geno == 1] = -1
        gam.haplotype[geno == 0] = 0
        gam.haplotype[geno == 2] = 1
        return gam

    @classmethod
    def from_offspring_genotype(cls, off_geno, other_geno=None):
        """
        Creates the gamete transmitted to an offspring from its
        genotype, possibly using information from its other
        parent's genotype.

        Arguments
        ---------
        - off_geno : genotype
            genotype of the offspring
        - other_geno : genotype
            genotype of the other parent
        """
        gam = cls.from_genotype(off_geno)
        if other_geno is not None:
            gam_other = cls.from_genotype(other_geno)
            try:
                assert len(gam_other.haplotype) == len(gam.haplotype)
            except AssertionError:
                print("Genotypes have different lengths")
                raise
            new_info_mk = (gam.haplotype == -1) & (gam_other.haplotype > -1)
            gam.haplotype[new_info_mk] = 1 - gam_other.haplotype[new_info_mk]
        return gam

    @classmethod
    def from_offspring_segregation(cls, par_geno, off_gam, off_seg):
        """
        Creates an inferred gamete for the parent given the gamete transmitted
        to an offspring and the segregation indicator of the meiosis.

        Arguments
        ---------
        - off_gam : gamete
           gamete received by the offspring
        - off_seg : array of int
           Segregation indicator of the grand-parental origin of the gamete
        """
        gam = cls.from_genotype(par_geno)
        # gam=cls()
        # gam.haplotype = np.full_like(off_gam.haplotype,-2)
        for i, a in enumerate(off_gam.haplotype):
            if a > -1 and off_seg[i] > -1:
                if par_geno[i] == 1:
                    gam.haplotype[i] = a if off_seg[i] == 1 else (1 - a)
                elif par_geno[i] < 0:
                    gam.haplotype[i] = a
        return gam

    @classmethod
    def from_wcsp_solver(cls, par_geno, dict_child_gam, mkpos=None):
        """Infer the gamete of the parent from transmitted gametes using a
        Weighted Constraint Satisfaction Problem

        Arguments
        ---------
        - par_geno : array of int
           genotype of the parent
        - dict_child_gam : dictionary of Gamete objects
           collection of gametes with key = offspring names

        """
        pg = Gamete.valid_genotype(par_geno)
        het_mk = np.where(pg == 1)[0]
        hap_data = {}
        new_gam = Gamete.from_genotype(pg)
        for k, v in dict_child_gam.items():
            hap_data[k] = list(dict_child_gam[k].haplotype[het_mk])
        if mkpos is not None:
            mkpos = mkpos[het_mk]
        phase_data = wcsp.PhaseData(hap_data, mkpos)
        if len(phase_data.info_mk) > 0:
            resolved_mk = [het_mk[i] for i in phase_data.info_mk]
            S = wcsp.PhaseSolver(
                phase_data.info_mk, phase_data.info_pairs, phase_data.recombination
            )
            S.add_constraints()
            try:
                par_phase = S.solve()
                new_gam.haplotype[resolved_mk] = par_phase
            except RuntimeError:
                logger.warning("WCSP solver failure")
        return new_gam

    @classmethod
    def combine(cls, first, second):
        """Combine two gametes into a new one
        Argument
        --------
        first, second : Gametes
            Gametes to combine
        Returns
        -------
        ( int, Gamete )

            Number of mismatches and resulting gamete. Any mismatch is
        resolved as -1.
        """
        assert type(first) == Gamete
        assert type(second) == Gamete
        assert len(first.haplotype) == len(second.haplotype)
        gam = cls()
        gam.haplotype = np.full_like(first.haplotype, -1)
        misses = []
        for i, alleles in enumerate(zip(first.haplotype, second.haplotype)):
            if alleles[0] < 0 and alleles[1] < 0:
                continue
            elif alleles[0] < 0:
                gam.haplotype[i] = alleles[1]
            elif alleles[1] < 0:
                gam.haplotype[i] = alleles[0]
            else:
                if alleles[0] != alleles[1]:
                    misses.append(i)
                    gam.haplotype[i] = -1
                else:
                    gam.haplotype[i] = alleles[0]
        return misses, gam

    def add(self, other):
        """Add two gametes to form a genotype

        Argument
        --------
        other : object of type Gamete
           the other gamete to add
        Returns
        -------
        numpy 1D array of integers : the genotype obtained
        """
        assert type(other) == Gamete
        assert len(self.haplotype) == len(other.haplotype)
        genotype = np.full_like(self.haplotype, -1)
        for i, alleles in enumerate(zip(self.haplotype, other.haplotype)):
            if alleles[0] < 0 or alleles[1] < 0:
                continue
            g = alleles[0] + alleles[1]
            assert g < 3
            genotype[i] = g
        return g

    @classmethod
    def complement(cls, gam, genotype):
        """Return the complementary gamete needed to form the given genotype.
        In case of incompatibility set both gametes to -1.
        """
        assert type(gam) == Gamete
        geno = Gamete.valid_genotype(genotype)
        newgam = cls()
        newgam.haplotype = np.full_like(gam.haplotype, -1)
        for i, (a, g) in enumerate(zip(gam.haplotype, geno)):
            if g < 0 or a < 0:
                continue
            if g == 1:
                newgam.haplotype[i] = g - a
            else:  # 0 -> (0,0) or 2->(1,1)
                try:
                    assert ((g == 0) and (a == 0)) or ((g == 2) and (a == 1))
                except AssertionError:
                    gam.haplotype[i] = -1
                else:
                    newgam.haplotype[i] = a
        return newgam
