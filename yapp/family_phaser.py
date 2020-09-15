# -*- coding: utf-8 -*-
"""
Module family_phaser.py

Infer phases and segregation indicators from high density SNP data
in pedigrees.
"""
import warnings
from yapp import vcf
from yapp import gamete
from yapp import pedigree

class ChromosomePair():
    def __init__(self, genotype):
        self.g = gamete.Gamete.valid_genotype(genotype)
        self.H= [ gamete.Gamete.from_genotype(self.g),
                  gamete.Gamete.from_genotype(self.g)]
        self._h_pat = 0
        self.si_pat = None
        self.si_mat = None

    @property
    def het_mks(self):
        "Heterozygous markers"
        return self.g==1

    @property
    def h_pat(self):
        """Index (0,1) of the paternal haplotype """
        return self._h_pat
    @h_pat.setter
    def h_pat(self,value):
        assert value==0 or value==1
        self._h_pat=value
        
    @property
    def h_mat(self):
        """Index (0,1) of the maternal haplotype """
        return 1-self._h_pat
    @h_mat.setter
    def h_mat(self,value):
        self.h_pat = 1-value
    
class Phaser():
    def __init__(self,vcf,ped):
        self.vcf = vcf
        pedindivs = [x for x in ped.nodes]
        for indiv in pedindivs:
            if indiv not in self.genotyped_samples:
                rm_node = ped.del_indiv(indiv)
        self.pedigree=ped

    @classmethod
    def from_prefix(cls,prfx,**kwargs):
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
        vcf_file=f"{prfx}.vcf.gz"
        fam_file=f"{prfx}.fam"
        my_ped=pedigree.Pedigree.from_fam_file(fam_file)
        my_vcf=vcf.vcf2fph(vcf_file,**kwargs)
        return cls(my_vcf,my_ped)
    
    @property
    def regions(self):
        return self.vcf['regions']

    @property
    def genotyped_samples(self):
        return self.vcf['samples']

    @property
    def pedigree_samples(self):
        return [x for x in self.pedigree.nodes]

    @property
    def families(self):
        return self.pedigree.families

    def phase_samples(self, dict_pairs=None):
        for reg in self.vcf['regions']:
            print(f"Working on : {reg}")
            genotypes = self.vcf['data'][reg]
            chrom_pairs = {}
            for node in self.pedigree:
                name = node.indiv
                print(f"\n{name} -- {node.gen}")
                if dict_pairs == None:
                    chrom_pairs[name]=ChromosomePair(genotypes[name])
                else:
                    chrom_pairs=dict_pairs
                p = chrom_pairs[name]

                print("1. At Init")
                print(f"geno :",*[f"{x:2}" for x in p.g])
                print(f".pat : {p.H[p.h_pat]}")
                print(f".mat : {p.H[p.h_mat]}")

                print("2. From Parents")
                if node.father != None:
                    geno_p = genotypes[node.father.indiv]
                    gam_p = gamete.Gamete.from_genotype(geno_p)
                    nmiss,new_gam_p=gamete.Gamete.combine(p.H[p.h_pat],gam_p)
                    p.H[p.h_pat]=new_gam_p
                    if nmiss>0:
                        warnings.warn(f"{region}{name}.pat: {nmiss} mismatches")
                    prop_gam_m=gamete.Gamete.complement(p.H[p.h_pat],p.g)
                    nmiss,new_gam_m=gamete.Gamete.combine(p.H[p.h_mat],prop_gam_m)
                    if nmiss>0:
                        warnings.warn(f"{region}{name}.mat: {nmiss} mismatches")
                    p.H[p.h_mat]=new_gam_m
                if node.mother != None:
                    geno_m = genotypes[node.mother.indiv]
                    gam_m = gamete.Gamete.from_genotype(geno_m)
                    nmiss,new_gam_m=gamete.Gamete.combine(p.H[p.h_mat],gam_m)
                    p.H[p.h_mat]=new_gam_m
                    if nmiss>0:
                        warnings.warn(f"{region}{name}.mat: {nmiss} mismatches")
                    prop_gam_p=gamete.Gamete.complement(p.H[p.h_mat],p.g)
                    nmiss,new_gam_p=gamete.Gamete.combine(p.H[p.h_pat],prop_gam_p)
                    if nmiss>0:
                        warnings.warn(f"{region}{name}.pat: {nmiss} mismatches")
                    p.H[p.h_pat]=new_gam_p
                print(f".pat : {p.H[p.h_pat]}")
                print(f".mat : {p.H[p.h_mat]}")
 
                print(f"3. from {len(node.children)} Offsprings")
                children_gametes = {}
                for child in node.children:
                    geno_off = genotypes[child.indiv]
                    if child.father is node:
                        if child.mother is not None:
                            geno_other=genotypes[child.mother.indiv]
                        else:
                            geno_other=None
                    elif child.mother is node:
                        if child.father is not None:
                            geno_other = genotypes[child.father.indiv]
                        else:
                            geno_other=None
                    gam_off = gamete.Gamete.from_offspring_genotype(geno_off,other_geno=geno_other)
                    ##print(f".off : {gam_off}")
                    children_gametes[child.indiv]=gam_off
                if len(node.children)>0:
                    try:
                        wcsp_gam = gamete.Gamete.from_wcsp_solver(p.g, children_gametes)
                    except:
                        warnings.warn("Could not run wcsp solver")
                    else:
                        ## combine with current gametes
                        n_miss_pat,new_gam_p = gamete.Gamete.combine(p.H[p.h_pat],wcsp_gam)
                        n_miss_mat,new_gam_m = gamete.Gamete.combine(p.H[p.h_mat],wcsp_gam)
                        if n_miss_mat < n_miss_pat:
                            p.H[p.h_mat]=new_gam_m
                            prop_gam_p=gamete.Gamete.complement(p.H[p.h_mat],p.g)
                            nmiss,new_gam_p=gamete.Gamete.combine(p.H[p.h_pat],prop_gam_p)
                            if nmiss>0:
                                warnings.warn(f"{region}{name}.pat: {nmiss} mismatches")
                            p.H[p.h_pat]=new_gam_p
                        else:
                            p.H[p.h_pat]=new_gam_p
                            prop_gam_m=gamete.Gamete.complement(p.H[p.h_pat],p.g)
                            nmiss,new_gam_m=gamete.Gamete.combine(p.H[p.h_mat],prop_gam_m)
                            if nmiss>0:
                                warnings.warn(f"{region}{name}.mat: {nmiss} mismatches")
                            p.H[p.h_mat]=new_gam_m
                print(f".pat : {p.H[p.h_pat]}")
                print(f".mat : {p.H[p.h_mat]}")
