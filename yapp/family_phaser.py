# -*- coding: utf-8 -*-
"""
Yapp module family_phaser.py

Infer phases and segregation indicators from high density SNP data
in pedigrees.
"""
import sys
import logging
import warnings
from collections import defaultdict
import tempfile
from multiprocessing import Pool, cpu_count
import bz2
import pickle
import numpy as np
from scipy.stats import binom
import cyvcf2 
from . import vcf, gamete, pedigree, MALE, FEMALE
try:
    from tqdm import tqdm
    has_tqdm=True
except:
    has_tqdm=False

logger = logging.getLogger(__name__)

def qerr( n, p, q=0.001):
    '''(1-q) Quantile of the binomial(n,p)'''
    return binom(p=p,n=n).isf(q)

def wcsp_phase( args):
    node,p,children_gametes = args
    try:
        wcsp_gam = gamete.Gamete.from_wcsp_solver(p.g, children_gametes)
    except:
        logger.error("Could not run wcsp solver")
        raise
    else:
        p.update_unknown_gamete(wcsp_gam)
        logger.debug(f"{node.indiv} -- [ "
                     f"sex:{node.sex} "
                     f"gen:{node.gen} "
                     f"par:{(node.father!=None)+(node.mother!=None)} "
                     f"off:{len(node.children)} "
                     f"nresolved/nhet : {p.nresolved}/{p.nhet}]")
    return (node.indiv,p)

def run_segregation_task(args):
    name, chpair_parent, gam, recmap = args
    return name, chpair_parent.get_segregation_indicators( gam, recmap)

class ChromosomePair():
    def __init__(self, genotype):
        self.g = gamete.Gamete.valid_genotype(genotype)
        self.H= [ gamete.Gamete.from_genotype(self.g),
                  gamete.Gamete.from_genotype(self.g)]
        self._h_pat = 0
        self.si_pat = [ (0,0.5) for _ in range(len(self.g))]
        self.si_mat = [ (0,0.5) for _ in range(len(self.g))]

    @property
    def len(self):
        return len(self.g)

    @property
    def het_mks(self):
        "Heterozygous markers"
        return self.g==1

    @property
    def nhet(self):
        return np.sum(self.het_mks)

    @property
    def phased(self):
        """Phased markers"""
        return (self.paternal_gamete.haplotype>0)|(self.maternal_gamete.haplotype>0)

    @property
    def resolved(self):
        """Resolved markers = heterozygotes and phased """
        return self.het_mks&self.phased
    
    @property
    def nresolved(self):
        return np.sum(self.resolved)
    
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

    @property
    def paternal_gamete(self):
        return self.H[self.h_pat]

    @property
    def maternal_gamete(self):
        return self.H[self.h_mat]
    
        
    def get_transmitted_from_segregation(self, si, call=0.99):
        """ 
        Returns the gamete transmitted from a segregation indicator vector.
        Inform haplotype only if posterior probability > 0.99
        """
        gam=gamete.Gamete.from_genotype(self.g)
        for i,(orig,prob) in enumerate(si):
            assert orig==1 or orig==0
            if prob<call:
                continue
            if orig ==0 :
                allele = self.h_pat
            else:
                allele = self.h_mat
            gam.haplotype[i]=self.H[allele].haplotype[i]
        return gam
    
    def get_segregation_indicators(self, gam, recmap, err=1e-3):
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
          bewteen all marker pairs

        Returns
        -------
        List( tuple( int, float) )
        segregation indicators for all markers with associated probability
        """
        assert type(gam) == gamete.Gamete
        assert len(gam.haplotype) == self.len
        assert len(recmap) == self.len-1

        ### 1. HMM Setup
        transitions = np.array( [ np.identity(2)*(1-x)+(1-np.identity(2))*x for x in recmap])
        emissions   = np.ones( (self.len,2), dtype=np.float) ## [ P(gam[m]| S==pat[m]), P(gam[m]| S==mat[m])]
        for m in range(self.len):
            if gam.haplotype[m] > 0 and self.g[m]>0 and self.H[0].haplotype[m]>-1:
                emissions[m,0]= (gam.haplotype[m] == self.paternal_gamete.haplotype[m]) and (1-err) or err 
                emissions[m,1]= (gam.haplotype[m] == self.maternal_gamete.haplotype[m]) and (1-err) or err
        ### 2. Forward-Backward algorithm
        fwd = np.ones( (self.len, 2), dtype=np.float) ## Forward
        rew = np.ones( (self.len, 2), dtype=np.float) ## Backward
        sca = np.ones( self.len, dtype=np.float)      ## scaling
        
        ## Compute forward probabilities
        fwd[0,0] = 0.5*emissions[0,0]
        fwd[0,1] = 0.5*emissions[0,1]
        sca[0]   = 1.0/(fwd[0,0]+fwd[0,1])
        fwd[0,] *= sca[0]
        for m in range(1,self.len):
            fwd[m,0]=( fwd[m-1,0]*transitions[m-1,0,0] + fwd[m-1,1]*transitions[m-1,1,0] )*emissions[m,0]
            fwd[m,1]=( fwd[m-1,0]*transitions[m-1,0,1] + fwd[m-1,1]*transitions[m-1,1,1] )*emissions[m,1]
            sca[m]  = 1.0/(fwd[m,0]+fwd[m,1])
            fwd[m,] *= sca[m]
        ## Compute backward probabilities
        rew[self.len-1,] /= sca[m-1]
        for m in range(self.len-1,0,-1):
            rew[m-1,0]= transitions[m-1,0,0]*emissions[m,0]*rew[m,0] + transitions[m-1,0,1]*emissions[m,1]*rew[m,1]
            rew[m-1,1]= transitions[m-1,1,0]*emissions[m,0]*rew[m,0] + transitions[m-1,1,1]*emissions[m,1]*rew[m,1]
            rew[m-1,]*=sca[m-1]
        loglik = -np.sum( np.log(sca))
        ## Compute Posterior probabilities
        post_si = fwd*rew
        post_si /= np.sum(post_si, axis=1, keepdims=True) ## required ?

        ### 3. Viterbi Algorithm
        ## viterbi variables
        delta = np.zeros((self.len,2), dtype=np.float)
        psi = np.zeros((self.len,2), dtype=np.int)
        soluce = np.empty(self.len, dtype=np.int)
        ## init
        delta[0,0] = np.log( 0.5*emissions[0,0])
        delta[0,1] = np.log( 0.5*emissions[0,1])
        ## recursion
        for m in range(1, self.len):
            val_0 = [ delta[m-1,0]+np.log(transitions[m-1,0,0]), delta[m-1,1]+np.log(transitions[m-1,1,0])]
            val_1 = [ delta[m-1,0]+np.log(transitions[m-1,0,1]), delta[m-1,1]+np.log(transitions[m-1,1,1])]
            psi[m,0]=np.argmax(val_0)
            psi[m,1]=np.argmax(val_1)
            delta[m,0]=val_0[psi[m,0]]+np.log(emissions[m,0])
            delta[m,1]=val_1[psi[m,1]]+np.log(emissions[m,1])
        ## termination / backtracking
        soluce[-1]=np.argmax(delta[-1,])
        for m in range(self.len-1,0,-1):
            soluce[m-1]=psi[m, soluce[m]]
        result = [ (x, post_si[i,x]) for i,x in enumerate(soluce) ]
        return result
        
    def update_paternal_gamete(self, prop_gam):
        return self.update_gamete( prop_gam, 0)

    def update_maternal_gamete(self, prop_gam):
        return self.update_gamete( prop_gam, 1)

    def update_unknown_gamete(self, prop_gam):
        """Update the phase when the origin of the prop_gam is not known.
        Tries to update paternal and maternal and keeps the one that leads 
        to the fewest # of mismatches.

        """
        n_miss_pat,new_gam_p = gamete.Gamete.combine(self.paternal_gamete,prop_gam)
        n_miss_mat,new_gam_m = gamete.Gamete.combine(self.maternal_gamete,prop_gam)
        if n_miss_mat < n_miss_pat:
            return self.update_maternal_gamete(prop_gam)
        else:
            return self.update_paternal_gamete(prop_gam)
        
    def update_gamete( self, prop_gam, parent):
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
        nmiss=[0,0]
        if parent == 0:
            origin = self.h_pat
        elif parent == 1:
            origin = self.h_mat
        else:
            raise ValueError(f"{type(self)}:[update_gamete]: parent must be 0 or 1")
        nmiss[origin],new_gam=gamete.Gamete.combine(self.H[origin],prop_gam)
        self.H[origin]=new_gam
        prop_gam_o=gamete.Gamete.complement(self.H[origin],self.g)
        nmiss[1-origin],new_gam=gamete.Gamete.combine(self.H[1-origin],prop_gam_o)
        self.H[1-origin]=new_gam
        return nmiss

class Phaser():
    def __init__(self,vcf_file,ped_file, out_prfx):
        ped=pedigree.Pedigree.from_fam_file(ped_file)
        self.vcf=vcf.vcf2fph(vcf_file, genotypes=False)
        self.vcf_out_file_name = f"{out_prfx}_phased.vcf.gz"
        v = cyvcf2.VCF(vcf_file, lazy=True)
        w = cyvcf2.Writer(self.vcf_out_file_name, v)
        w.write_header()
        w.close()
        self.vcf_file_name=vcf_file
        self.ped_file_name=ped_file
        self.err=1e-3
        ## individuals
        pedindivs = [x for x in ped.nodes]
        self.ignored_indivs=[]
        for indiv in pedindivs:
            if indiv not in self.genotyped_samples:
                rm_node = ped.del_indiv(indiv)
                self.ignored_indivs.append(rm_node)
        self.pedigree=ped
        ## Init phases
        # self.phases=defaultdict(dict)
        # for reg in self.regions:
        #     genotypes=self.vcf['data'][reg]
        #     for node in self.pedigree:
        #         self.phases[reg][node.indiv]=ChromosomePair(genotypes[node.indiv])
                
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
        return cls(vcf_file,fam_file,prfx)
    
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

    def recmap(self, recrate=1):
        """Computes a recombination map assuming a rate of recrate (in cM/Mb) along regions """
        res = {}
        for reg in self.regions:
            snps = self.vcf['variants'][reg]
            pos = np.array([ x[2] for x in snps])
            distances = pos[1:]-pos[:-1]
            distances[distances<100]=100
            res[reg] = distances * recrate * 1e-8
        return res


    def write_phased_vcf(self,fname):
        """Write phase information in a VCF file
        """
        pass
        ##vcf_tmpl = cyvcf2.VCF(self.vcf_file_name, lazy=True)
        ## create a temporary file to dump results
        # fd,tmpfile_name=tempfile.mkstemp(suffix='.vcf')
        # w = cyvcf2.Writer(tmpfile_name, vcf_tmpl)
        # w.header_written=True
        # snp_mapping = {}
        # for s in vcf_tmpl:
        #     snp_mapping[(s.ID,s.CHROM,s.POS)]=s
        # smp_mapping = {}
        # for n in self.pedigree:
        #     smp_mapping[n.indiv]=vcf_tmpl.samples.index(n.indiv)
        # for reg in self.regions:
        #     snps = self.vcf['variants'][reg]
        #     variants = [snp_mapping[tuple(x[:3])] for x in snps]
        #     for i,v in enumerate(variants):
        #         ph_genotypes=v.genotypes[:]
        #         for node in self.pedigree:
        #             ph = self.phases[reg][node.indiv]
        #             nidx=smp_mapping[node.indiv]
        #             if ph.H[0].haplotype[i]>-1:
        #                 ph_genotypes[nidx]=[ph.paternal_gamete.haplotype[i], ph.maternal_gamete.haplotype[i], 1]
        #         v.genotypes=ph_genotypes
        #         w.write_record(v)
    
    def save(self,fname):
        """Save Phaser internal state to file"""
        dbfile=bz2.BZ2File(fname,'wb')
        pickle.dump(self,dbfile)
        dbfile.close()

    @classmethod
    def load(cls, fname):
        with bz2.BZ2File(fname,'rb') as f:
            obj=pickle.load(f)
        obj.__class__=cls
        return obj
        

    def get_genotypes(self, region):
        tempvcf_data = vcf.vcf2fph(self.vcf_file_name, reg=region, genotypes=True)
        return tempvcf_data['data'][region]
        
    def run(self):
        for reg in self.regions:
            logger.info(f"Working on region {reg}")
            phases = self.phase_samples_from_genotypes(reg)
            nhet = 0
            nresolved = 0
            for p in phases.values():
                nhet += p.nhet
                nresolved += p.nresolved
            logger.info(f"Resolved {nresolved} out of {nhet} phases : {100*nresolved/nhet:.1f}%")
            sys.stdout.flush()
            phases = self.phase_samples_from_segregations(reg, phases=phases)
            nhet = 0
            nresolved = 0
            for p in phases.values():
                nhet += p.nhet
                nresolved += p.nresolved
            logger.info(f"Resolved {nresolved} out of {nhet} phases : {100*nresolved/nhet:.1f}%")
        # print("***** Looking for crossovers *****")
        # recombinations = self.identify_crossovers(self.phases)
        ##self.write_phased_vcf('test.vcf.gz')
        # return { 'phases' : phases,
        #          'recombinations' : recombinations }
    def phase_samples_from_segregations(self, region, phases=None, recrate=1):
        """Infer segregation indicators in the pedigree.

        Arguments
        ---------
        recrate : recombination rate in cM/Mb

        Returns
        -------
        update phases
        """
        logger.info("Phasing samples from segregations")
        recmaps = self.recmap(recrate)
        if phases == None:
            chrom_pairs = self.phases_from_genotypes(region)
        else:
            chrom_pairs = phases
        recmap=recmaps[region]
        logger.debug("1. Infer Segregation Indicators")
        pat_seg_tasks=[]
        mat_seg_tasks=[]
        for node in self.pedigree:
            chpair = chrom_pairs[node.indiv]
            logger.debug(f"{node.indiv} -- [ "
                          f"sex:{node.sex} "
                          f"gen:{node.gen} "
                          f"par:{(node.father!=None)+(node.mother!=None)} "
                          f"off:{len(node.children)} "
                          f"nhet : {chpair.nhet}"
                          f"nresolved : {chpair.nresolved} ]")
            if node.father:
                pat_seg_tasks.append( ( node.indiv, chrom_pairs[node.father.indiv], chpair.paternal_gamete, recmap))
            if node.mother:
                mat_seg_tasks.append( ( node.indiv, chrom_pairs[node.mother.indiv], chpair.maternal_gamete, recmap))

        if has_tqdm:
            pbar = tqdm(total= len(pat_seg_tasks)+len(mat_seg_tasks))
            pbar.set_description("Building segregation indicators")
        with Pool(cpu_count()) as workers:
            for indiv, segind in workers.imap(run_segregation_task, pat_seg_tasks):
                node = self.pedigree.nodes[indiv]
                chpair_p = chrom_pairs[node.father.indiv]
                chpair = chrom_pairs[node.indiv]
                chpair.si_pat=segind
                new_gam = chpair_p.get_transmitted_from_segregation(chpair.si_pat)
                nmiss=chpair.update_paternal_gamete(new_gam)
                if (nmiss[0]+nmiss[1]) > qerr(chpair.nhet*2, self.err, q=1e-3/(len(pat_seg_tasks)+len(mat_seg_tasks))):
                    logger.warning(f"{node.father.indiv}[pat] -> {node.indiv} :{50*(nmiss[0]+nmiss[1])/chpair.nhet:.1g} % mismatch : possible pedigree error")
                if has_tqdm:
                    pbar.update(1)
            for indiv, segind in workers.imap(run_segregation_task, mat_seg_tasks):
                node = self.pedigree.nodes[indiv]
                chpair_m = chrom_pairs[node.mother.indiv]
                chpair = chrom_pairs[node.indiv]
                chpair.si_mat=segind
                new_gam = chpair_m.get_transmitted_from_segregation(chpair.si_mat)
                nmiss=chpair.update_maternal_gamete(new_gam)
                if (nmiss[0]+nmiss[1]) > qerr(chpair.nhet*2, self.err, q=1e-3/(len(pat_seg_tasks)+len(mat_seg_tasks))):
                    logger.warning(f"{node.mother.indiv}[mat] -> {node.indiv} :{50*(nmiss[0]+nmiss[1])/chpair.nhet:.1g} % mismatch : possible pedigree error")
                if has_tqdm:
                    pbar.update(1)

        logger.debug("2. Update parental phases")
        wcsp_tasks=[]
        for node in self.pedigree:
            p=chrom_pairs[node.indiv]
            logger.debug(f"{node.indiv} -- [ "
                  f"sex:{node.sex} "
                  f"gen:{node.gen} "
                  f"par:{(node.father!=None)+(node.mother!=None)} "
                  f"off:{len(node.children)} "
                  f"nhet : {p.nhet} ]")
            children_gametes = {}
            for child in node.children:
                chpair = chrom_pairs[child.indiv]
                if child.father is node:
                    children_gametes[child.indiv]=chpair.paternal_gamete
                elif child.mother is node:
                    children_gametes[child.indiv]=chpair.maternal_gamete
            if len(node.children)>0:
                wcsp_tasks.append((node,p,children_gametes))

        logger.info(f"Phasing {len(wcsp_tasks)} with  WCSP")
        if has_tqdm:
            pbar = tqdm(total=len(wcsp_tasks))
            pbar.set_description("WCSP phasing")
        with Pool(cpu_count()) as workers:
            for indiv,pair in workers.imap(wcsp_phase,wcsp_tasks):
                del chrom_pairs[indiv]
                chrom_pairs[indiv]=pair
                pbar.update(1)
        return chrom_pairs

    def phase_samples_from_genotypes(self, region, phases=None):
        """Reconstruct paternal and maternal gametes in the pedigree 
        based on observed genotypes.
        Returns
        -------
        Dict[ str, Dict[ str, ChromosomePair ]]
            regions as keys, dict( name : ChromosomePair) as value
        """
        logger.info("Phasing samples from genotypes")
        genotypes = self.get_genotypes(region)
        if phases == None:
            chrom_pairs = {}
            for node in self.pedigree:
                chrom_pairs[node.indiv]=ChromosomePair(genotypes[node.indiv])
        else:
            chrom_pairs = phases
        wcsp_tasks=[]
        if has_tqdm:
            pbar = tqdm(total=len(self.pedigree.nodes))
        for node in self.pedigree:
            name = node.indiv
            if has_tqdm:
                pbar.set_description(f"Processing {name:<20}")
            logger.debug(f"{name} -- [ "
                          f"sex:{node.sex} "
                          f"gen:{node.gen} "
                          f"par:{(node.father!=None)+(node.mother!=None)} "
                          f"off:{len(node.children)} ]")
            p = chrom_pairs[name]

            logger.debug("1. Initialize gametes")
            logger.debug(f"nhet : {p.nhet}")
            logger.debug(f"geno :",*[f"{x:2}" for x in p.g])
            logger.debug(f".pat : {p.paternal_gamete}")
            logger.debug(f".mat : {p.maternal_gamete}")
            logger.debug("2. Update gametes from Parents")
            if node.father != None:
                geno_p = genotypes[node.father.indiv]
                gam_p = gamete.Gamete.from_genotype(geno_p)
                nmiss=p.update_paternal_gamete(gam_p)
                if (nmiss[0]+nmiss[1]) > qerr(p.nhet*2, self.err):
                    logger.warning(f"{node.father.indiv}[pat] -> {node.indiv} :{50*(nmiss[0]+nmiss[1])/p.nhet}% mismatches : possible pedigree error")

            if node.mother != None:
                geno_m = genotypes[node.mother.indiv]
                gam_m = gamete.Gamete.from_genotype(geno_m)
                nmiss = p.update_maternal_gamete(gam_m)
                if (nmiss[0]+nmiss[1]) > qerr(p.nhet*2, self.err):
                    logger.warning(f"{node.mother.indiv}[mat] -> {node.indiv} :{50*(nmiss[0]+nmiss[1])/p.nhet}% mismatches : possible pedigree error")

            logger.debug(f".pat : {p.paternal_gamete}")
            logger.debug(f".mat : {p.maternal_gamete}")
            logger.debug(f"nresolved : {p.nresolved}")
            sys.stdout.flush()


            logger.debug(f"3. Update gametes from {len(node.children)} Offsprings")
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
                logger.debug(f".off : {gam_off}")
                children_gametes[child.indiv]=gam_off
            if len(node.children)>0:
                wcsp_tasks.append((node,p,children_gametes))
            if has_tqdm:
                pbar.update(1)

        logger.info(f"Phasing {len(wcsp_tasks)} parents with WCSP")
        if has_tqdm:
            pbar = tqdm(total=len(wcsp_tasks))
            pbar.set_description("WCSP phasing")
        with Pool(cpu_count()) as workers:
            for indiv,pair in workers.imap(wcsp_phase,wcsp_tasks):
                del chrom_pairs[indiv]
                chrom_pairs[indiv]=pair
                pbar.update(1)
        return chrom_pairs


def main(args):
    prfx=args.prfx
    # if len(args)<1:
    #     print("usage: yapp phase <prfx>")
    #     sys.exit(1)
    # logging.basicConfig(format='%(asctime)s  %(levelname)s: %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p',
    #                     filename=prfx+'_yapp_phase.log',filemode='w',level=logging.INFO)
    phaser=Phaser.from_prefix(prfx)
    phaser.run()
    # logger.info(f"Exporting results to : {prfx}_phased.vcf.gz and {prfx}_yapp.db")
    # phaser.write_phased_vcf(prfx+'_phased.vcf.gz')
    # phaser.save(prfx+'_yapp.db')
    logger.info("YAPP PHASE analysis done")
if __name__=='__main__':
    main(sys.argv)
