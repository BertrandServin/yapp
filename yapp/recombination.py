# -*-coding:utf-8 -*
''' 
Yapp module recombination.py

Yapp module to analyse recombination information in pedigrees
'''
import sys
import logging
import random
import collections.abc
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma,binom, poisson
try:
    from tqdm import tqdm
    has_tqdm=True
except:
    has_tqdm=False

from . import family_phaser, MALE, FEMALE

logger = logging.getLogger(__name__)

##import matplotlib.pyplot as plt

def reg2chr(bedstr):
    """Returns chromosome value from  bed string"""
    return bedstr.split(':')[0]

class RecombAnalyser():
    def __init__(self, phaser_db):
        self.phaser = family_phaser.Phaser.load(phaser_db)
        self.parents = {}
        for node in self.phaser.pedigree:
            if len(node.children)>1: ## need at least 2 offspring to detect recomb 
                self.add_parent(node)
        self.size_covered = defaultdict( lambda : defaultdict(lambda : 0.0)) ## size_covered[sex][chrom]
        self._gsize=None
    @property
    def crossovers(self):
        """Iterator over crossing overs"""
        return (co for co in ind.meioses.values() for ind in self.parents)

    @property
    def male_crossovers(self):
        """Iterator over male crossing overs"""
        return (co for co in ind.meioses.values() for ind in self.parents if ind.sex==MALE)

    @property
    def female_crossovers(self):
        """Iterator over female crossing overs"""
        return (co for co in ind.meioses.values() for ind in self.parents if ind.sex==FEMALE)
    
    def add_parent(self, pednode):
        try:
            par = self.parents[pednode.indiv]
        except KeyError:
            par = Parent(pednode.indiv, pednode.sex)
            self.parents[pednode.indiv]=par
        else:
            logger.warning(f"Trying to create the parent {pednode.indiv} multiple times")

    @property
    def genome_size(self):
        res = 0
        for reg in self.phaser.regions:
            res += self.phaser.vcf['variants'][reg][-1][2]
        return res
        
    @staticmethod
    def get_crossovers(si, call=0.99):
        """Find crossover boundaries from segregation indicators
        
        Arguments
        ---------
        - si : np.array( (2,L) )
           Segragation indicator. si[0,] is the best phase solution, si[1,] is the 
           probability that the phase is correct. At each phase switch, determines the 
           boundaries such that the phase probabilty is > call
        - call : float
           Minimum probability to consider a phase resolved.
        
        Returns
        -------
        
        list[ [l,r] ]
        list of crossover boundaries found (left, right) = Indices in the array.
        
        """
        best_guess = np.array([x[0] for x in si])
        phase_prob = np.array([x[1] for x in si])
        co_loc = np.asarray( (best_guess[1:]-best_guess[:-1])!=0).nonzero()[0]
        nco = len(co_loc)
        res=[]
        if nco==0:
            return res
        for l in co_loc:
            left=l
            while phase_prob[left]<call:
                if left==0:
                    break
                left -=1
            right=l+1
            while phase_prob[right]<call:
                if right == len(best_guess)-1:
                    break
                right +=1
            res.append([left,right])
        return res

    @staticmethod
    def min_interval_size( nmeio, recrate=1, alpha=0.05):
        """Determine minimal interval size to estimate recombination rates
        
        Assuming a Poisson distributed number of crossovers in an interval of size L.
        The probability of sampling 0 crossovers among nmeio meioses is:

        p_0 = exp( - recrate * L * nmeio)

        The minimal interval size is the one that satisfies p_0 = alpha, or:

        L = -log(alpha)/(recrate*nmeio)

        Arguments
        ---------
        nmeio : int
          number of meioses
        recracte : float
          recombination rate (in cM/Mb)
        alpha : float
          precision parameter ( P(k=0)<alpha)
          
        Returns
        -------
        float
           minimum interval size
        """
        return 1e8*np.log(alpha)/(recrate*nmeio)
    
    def run(self):
        """ Run the recombination analysis """
        print("Calculating number of informative meioses for each parent")
        self.set_informative_meioses()
        print("Finding recombinations")
        self.identify_crossovers()
            
    def set_informative_meioses(self):
        """ Identify informative meioses for each parent """
        logger.info("Set Informative meioses")
        if has_tqdm:
            pbar= tqdm(total=len(self.phaser.regions)*len(self.parents))
        for reg in self.phaser.regions:
            if has_tqdm:
                pbar.set_description(f"Processing {reg}")
            snps = self.phaser.vcf['variants'][reg]
            pos = np.array([ x[2] for x in snps])
            mids = np.array( [ 0.5*(x+y) for x,y in zip(pos[:-1],pos[1:])])
            chrom = reg2chr(reg)
            chrom_pairs = self.phaser.phases[reg]
            for indiv,par in self.parents.items():
                chpair = chrom_pairs[indiv]
                node = self.phaser.pedigree.nodes[indiv]
                n_meio_info = np.zeros(len(snps)-1, dtype=np.int)
                for c in node.children:
                    par.meioses[c.indiv]=[]
                    chpair_c = chrom_pairs[c.indiv]
                    combin_info = chpair.resolved&chpair_c.phased
                    infomk = combin_info.nonzero()[0]
                    if len(infomk)>0:
                        infomk_l = min(infomk)
                        infomk_r = max(infomk)
                        n_meio_info[infomk_l:infomk_r]+=1
                        self.size_covered[node.sex][chrom]+=(pos[infomk_r]-pos[infomk_l])*1e-6
                par.set_n_info_meioses(chrom, mids, n_meio_info)
                for pp in range(0,max(pos), 1000000):
                    logging.debug(f"{indiv}:{chrom} {pp} {par.n_info_meioses(chrom,pp)}")
                if has_tqdm:
                    pbar.update(1)
        for sex in self.size_covered:
            for chrom in self.size_covered[sex]:
                logging.info(f"sex:{sex} chrom:{chrom} size:{self.size_covered[sex][chrom]} Mb")

    def identify_crossovers(self, recrate = 1):
        logging.info("Gathering crossovers")
        recmaps = self.phaser.recmap(recrate)
        if has_tqdm:
            pbar= tqdm(total=len(self.phaser.regions)*len(self.phaser.pedigree.nodes))
   
        for reg in self.phaser.regions:
            if has_tqdm:
                pbar.set_description(f"Processing {reg}")
            logging.info(f"Working on : {reg}")
            chrom_pairs = self.phaser.phases[reg]
            recmap=recmaps[reg]
            chrom = reg2chr(reg)
            snps = self.phaser.vcf['variants'][reg]
            pos = np.array([ x[2] for x in snps])
            for node in self.phaser.pedigree:
                logging.debug(f"{node.indiv} -- [ "
                      f"sex:{node.sex} "
                      f"gen:{node.gen} "
                      f"par:{(node.father!=None)+(node.mother!=None)} "
                      f"off:{len(node.children)} ]")
                chpair = chrom_pairs[node.indiv]
                if node.father:
                    try:
                        par = self.parents[node.father.indiv]
                    except KeyError:
                        continue
                    else:
                        chpair_p = chrom_pairs[node.father.indiv]
                        chpair.si_pat = chpair_p.get_segregation_indicators(chpair.paternal_gamete,recmap)
                        cos = self.get_crossovers(chpair.si_pat)
                        for x,y in cos:
                            par.add_offspring_CO(node.indiv, chrom, pos[x],pos[y])
                if node.mother:
                    try:
                        par = self.parents[node.mother.indiv]
                    except KeyError:
                        continue
                    else:
                        chpair_m = chrom_pairs[node.mother.indiv]
                        chpair.si_mat = chpair_m.get_segregation_indicators(chpair.maternal_gamete,recmap)
                        cos=self.get_crossovers(chpair.si_mat)
                        for x,y in cos:
                            par.add_offspring_CO(node.indiv, chrom, pos[x],pos[y])
                if has_tqdm:
                    pbar.update(1)
 
        for name,par in self.parents.items():
            to_rm=[]
            for off in par.meioses:
                pval = poisson(recrate*self.genome_size*1e-8).sf(len(par.meioses[off]))
                if pval < 1e-6:
                    logging.warning(f"par:{name} off:{off} sex:{par.sex} nco:{len(par.meioses[off])} "
                                    f"pval:{pval:.3g} -> very high number of COs. Meiosis will be ignored")
                    to_rm.append(off)
                elif pval < 1e-3:
                    logging.info(f"par:{name} off:{off} sex:{par.sex} nco:{len(par.meioses[off])} "
                                    f"pval:{pval:.3g} -> Number of COs seems somewhat unlikely.")
            for off in to_rm:
                del par.meioses[off]
    
    def write_crossovers(self, prefix):
        with open(prefix+'_yapp_recombinations.txt','w') as fout:
            print("parent sex offspring chrom left right",file=fout)
            for name, par in self.parents.items():
                for off in par.meioses:
                    for co in par.meioses[off]:
                        print(f"{name} {((par.sex==None) and 'U') or ((par.sex==MALE) and 'M' or 'F')} {off} "
                              f"{co.chrom} {co.left} {co.right}",file=fout)
class Parent():
    '''
    Class for storing information for each parent

    Attributes
    ----------
    - name : str
        identifier for the parent
    - sex : int or None
        sex of the parent ( 0: Male, 1: Female)
    - meioses : dict( str : list of CrossingOver objects)
        meioses of the individual.
    - nmeio_info : function: x(chrom,pos) -> int
        a function that returns the number of meiosis at a given genomic coordinate
    '''
    
    def __init__(self,name,sex=None): 
        self.name = name
        self.sex = sex
        self.meioses = defaultdict(list)
        self._nmeio_info = None
        
    @property
    def nmeioses(self):
        return len(self.meioses)
    
    @property
    def nb_CO_meioses(self):
        ''' 
        List of the number of crossovers for each meiosis
        '''
        return [len(v) for v in self.meioses.values()]

    @property
    def nb_CO_tot(self):
        '''
        Total number of crossing over in all meioses
        '''
        return np.sum(self.nb_CO_meioses)

    def get_offspring_CO(self, name):
        '''
        Get the list of CO for offspring with name *name*
        '''
        return self.meioses[name]

    def add_offspring_CO(self, name, chro, left, right):
        '''
        Add a crossing over in offspring *name* on chromosome chro between *left* and *right*
        '''
        myco = CrossingOver(chro, left, right)
        self.meioses[name].append(myco)

    def n_info_meioses(self, chrom, pos):
        '''Get the number of informative meioses for the parent 
        at position pos on chromosome chrom.
        
        Arguments
        ---------
        - chrom : int
            Chromosome
        - pos : int
            Position

        Returns
        -------
        int
           number of informative meioses
        '''
        try:
            return self._nmeio_info( chrom, pos)
        except TypeError:
            return self.nmeioses

    def set_n_info_meioses(self, chrom, positions, values):
        '''Enter information on the number of informative meioses on
        a chromosome, at a set of positions.

        Arguments
        ---------
        - chrom : int
            chromosome
        - positions : array of int
            Positions at which the number of informative meioses is known
        - values : array of int
            Number of informative meioses at each position
        '''
        if self._nmeio_info == None:
            self._nmeio_info = Nmeioses(self.nmeioses)
        self._nmeio_info.add_predictor( chrom, positions, values)

    def n_info_meioses_seg(self, chrom, left, right):
        return max(self.n_info_meioses(chrom,left), self.n_info_mesioses(chrom,right))
        
    
    def oi_xi(self, chrom, w_left, w_right):
        '''
        Computes probabilities that each crossing over in the parent occurs in
        genomic region on chromosome *chrom* between positions *w_left* and *w_right*.
        
        Returns a tuple with entries:
        -- list of contributions for each CO
        -- number of informative meioses for the parent in the region
        '''
        contrib = []
        for m in self.meioses.values():
            for co in m:
                if co.chro == chrom and not( (w_right < co.left ) or (w_left > co.right)):
                    contrib += [co.oi_xi(w_left, w_right)]
        return (contrib, self.n_info_meioses_seg(chrom, w_left, w_right))

class Nmeioses(collections.abc.Callable):
    '''Class offering a callable interface to interpolate the number of informative meioses 
    from observed data.

    Usage
    -----
    Nmeioses(chrom, pos) -> int

    Nmeioses.add_predictor(chrom, positions, values) : set up a 1D interpolator 
    for chromosome chrom from observed (positions , values) points.

    Returns
    -------
    int
       Interpolated number of meioses. 0 if outside training range
    '''
    def __init__(self, default_value):
        self.default = int(default_value)
        self.predictors = defaultdict( lambda : lambda x : self.default)
        
    def __call__(self, chrom, pos):
        try: 
            return int( np.ceil( self.predictors[chrom](pos)))
        except ValueError:
            return 0
    
    def add_predictor(self, chrom, positions, values):
        self.predictors[chrom]=interp1d(positions, values, fill_value=0)
        
class CrossingOver():
    '''
    Class to store crossing over information
    
    Attributes:
    -- chro : chromosome
    -- left : position of the marker on the left side
    -- right : position of the marker on the right side
    '''
    def __init__(self, chrom, left, right):
        assert right > left
        self.chrom = chrom
        self.left = left
        self.right = right

    @property
    def size(self):
        return self.right-self.left
                
    def oi_xi(self, w_left, w_right):
        '''
        Computes the probability that the crossing over occured n the window between w_left and w_right
        '''
        if (w_right < self.left ) or (w_left > self.right):
            ## no overlap
            ## wl------wr               wl----------wr
            ##             sl-------sr
            return 0
        elif (w_left <= self.left) and (self.right <= w_right):
            ## co included in window
            ## wl---------------------------wr
            ##      sl---------------sr
            return 1
        elif (self.left <= w_left) and (w_right <= self.right):
            ## window is included in co
            ##          wl------wr
            ## sl---------------------sr
            return float(w_right - w_left)/(self.right-self.left)
        elif (w_left < self.left):
            ## we know w_right<self.right as other case is treated above
            ## wl-----------------wr
            ##       sl------------------sr
            return float(w_right-self.left)/(self.right-self.left)
        else:
            ## only case left
            ##           wl-----------------wr
            ##    sl--------------sr
            try: 
                assert (self.left <= w_left) and (self.right < w_right)
            except AssertionError:
                print(self.right, w_right, self.left, w_left)
                raise
            return float(self.right-w_left)/(self.right-self.left)



def main(args):
    if len(args)<1:
        print("Usage: yapp recomb <prfx>")
        sys.exit(1)
    prfx=args[0]
    phaser_db = prfx+'_yapp.db'
    logging.basicConfig(format='%(asctime)s  %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=prfx+'_yapp_recomb.log',filemode='w',level=logging.INFO)
    logging.info("Starting YAPP RECOMB analysis")
    analyzer = RecombAnalyser(phaser_db)
    analyzer.run()
    analyzer.write_crossovers(prfx)
    logging.info("YAPP RECOMB analysis done")


if __name__ == '__main__':
    main(sys.argv)



