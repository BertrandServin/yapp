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
import numpy as np
from yapp import wcsp

class Gamete():
    """A gamete = an haplotype transmitted during meiosis
    """
    def __init__(self):
        self.haplotype = None
        self.origin = None

    @staticmethod
    def valid_genotype(genotype):
        try:
            geno = np.array(genotype, dtype=np.int)
        except ValueError:
            print("Genotype must be castable to a numpy array of integers")
            raise
        else:
            try:
                assert len(geno.shape)==1
            except AssertionError:
                raise ValueError("Genotype must be (castable to) a 1-D numpy array.")
        try:
            assert np.all( (geno>-2) & (geno<3))
        except AssertionError:
            raise AssertionError("Genotype values must be in [-1,0,1,2]")
        return geno

    def __str__(self):
        return ' '.join([ f"{x<0 and ' -' or x:2}" for x in self.haplotype])
    
    @classmethod
    def from_genotype(cls, genotype, origin=None):
        """
        Creates a gamete from a genotype.

        Arguments
        ---------
        - genotype : sequence
           A diploid genotype with values
        - origin : obj
           The origin of the gamete. Default to None (Unknown).

        Returns
        -------
        A Gamete object with haplotype of the same length as genotype. 
        Haplotype values are :
       """
        
        gam = cls()
        geno = cls.valid_genotype(genotype)
        gam.haplotype=np.full_like(geno, -2)
        gam.haplotype[geno==1]=-1
        gam.haplotype[geno==0]=0
        gam.haplotype[geno==2]=1
        gam.origin=origin
        return gam

    @classmethod
    def from_offspring_genotype(cls, off_geno, other_geno=None, origin=None):
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
        gam = cls.from_genotype(off_geno,origin)
        if other_geno is not None:
            gam_other = cls.from_genotype(other_geno)
            try:
                assert len(gam_other.haplotype) == len(gam.haplotype)
            except AssertionError:
                print("Genotypes have different lengths")
                raise
            new_info_mk = (gam.haplotype==-1) & (gam_other.haplotype > -1)
            gam.haplotype[new_info_mk] = 1 - gam_other.haplotype[new_info_mk]
        return gam

    @classmethod
    def from_wcsp_solver(cls, par_geno, dict_child_gam):
        """
        Infer the gamete of the parent from transmitted gametes using a Weighted Constraint Satisfaction Problem
        
        Arguments
        ---------
        - par_geno : array of int
           genotype of the parent
        - dict_child_gam : dictionary of Gamete objects
           collection of gametes with key = offspring names
        """
        pg = Gamete.valid_genotype(par_geno)
        het_mk = [ i for i,x in enumerate(pg) if x ==1]
        hap_data = {}
        new_gam = Gamete()
        new_gam.haplotype = np.full_like(pg, -1)
        for k,v in dict_child_gam.items():
            hap_data[k] = list(dict_child_gam[k].haplotype[het_mk])
        phase_data = wcsp.PhaseData(hap_data)
        resolved_mk = [het_mk[i] for i in phase_data.info_mk]
        S=wcsp.PhaseSolver(phase_data.info_mk,phase_data.info_pairs,phase_data.recrate)
        S.add_constraints()
        par_phase=S.solve()
        new_gam.haplotype[resolved_mk]=par_phase
        return new_gam
    
    @classmethod
    def combine(cls,first, second):
        '''Combine two gametes into a new noe
        Argument
        --------
        first, second : Gametes
            Gametes to combine
        Returns
        -------
        ( int, Gamete )
            Number of mismatches and resulting gamete. Any mismatch is resolved as -1.
        '''
        assert type(first)==Gamete
        assert type(second)==Gamete
        assert len(first.haplotype) == len(second.haplotype)
        gam = cls()
        gam.haplotype = np.full_like(first.haplotype, -1)
        nmiss=0
        for i, alleles in enumerate(zip(first.haplotype,second.haplotype)):
            if alleles[0]<0 and alleles[1]<0:
                continue
            elif alleles[0]<0:
                gam.haplotype[i]=alleles[1]
            elif alleles[1]<0:
                gam.haplotype[i]=alleles[0]
            else:
                if alleles[0]!=alleles[1]:
                    gam.haplotype[i]=-1
                    nmiss+=1
                else:
                    gam.haplotype[i]=alleles[0]
        return nmiss,gam
    
    def add(self,other):
        """Add two gametes to form a genotype

        Argument
        --------
        other : object of type Gamete
           the other gamete to add
        Returns
        -------
        numpy 1D array of integers : the genotype obtained
        """
        assert type(other)==Gamete
        assert len(self.haplotype)==len(other.haplotype)
        genotype = np.full_like(self.haplotype, -1)
        for i,alleles in enumerate(zip(self.haplotype,other.haplotype)):
            if alleles[0]<0 or alleles[1]<0:
                continue
            g = alleles[0]+alleles[1]
            assert g < 3
            genotype[i]=g
        return g

    @classmethod
    def complement(cls,gam,genotype,origin=None):
        """Return the complementary gamete needed to form the given genotype
        """
        assert type(gam)==Gamete
        geno = Gamete.valid_genotype(genotype)
        newgam = cls()
        newgam.haplotype=np.full_like(gam.haplotype, -1)
        for i, (a, g) in enumerate(zip(gam.haplotype,geno)):
            if g<0 or a<0:
                continue
            if g==1:
                newgam.haplotype[i]=g-a
            else: ## 0 -> (0,0) or 2->(1,1)
                try:
                    assert ((g==0) and (a==0)) or ((g==2) and (a==1))
                except AssertionError:
                    gam.haplotype[i]=-1
                else:
                    newgam.haplotype[i]=a
        return newgam