# -*- coding: utf-8 -*-
"""
Module gamete.py from yapp

This module exports classes and functions to build and manipulate gametes
and their transmissions from pedigree and genotype data.

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
from yapp import pedigree as ped

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
        for i,alleles in enumerate(zip(self.haplotype)):
            if alleles[0]<0 or alleles[1]<0:
                continue
            g = alleles[0]+alleles[1]
            assert g < 3
            genotype[i]=g
        return g

    def complement(self,genotype,origin=None):
        """Return the complementary gamete needed to form the given genotype
        """
        print("Not implemented")
        pass
