# -*-coding:utf-8 -*
"""
Yapp module recombination.py

Yapp module to analyse recombination information in pedigrees
"""
import sys
import logging
import collections.abc
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d

from . import family_phaser, MALE, FEMALE

logger = logging.getLogger(__name__)


def reg2chr(bedstr):
    """Returns chromosome value from  bed string"""
    return bedstr.split(":")[0]


class RecombAnalyser:
    def __init__(self, phaser_db):
        self.phaser = family_phaser.Phaser.load(phaser_db)
        self.parents = {}
        for node in self.phaser.pedigree:
            # need at least 2 offspring to detect recomb
            if len(node.children) > 0:
                self.add_parent(node)
        # size_covered[sex][chrom]
        self.size_covered = defaultdict(lambda: defaultdict(lambda: 0.0))
        self._gsize = None

    @property
    def crossovers(self):
        """Iterator over crossing overs"""
        return (co for co in ind.meioses.values() for ind in self.parents)  # noqa

    @property
    def male_crossovers(self):
        """Iterator over male crossing overs"""
        return (
            co
            for co in ind.meioses.values()  # noqa
            for ind in self.parents
            if ind.sex == MALE
        )

    @property
    def female_crossovers(self):
        """Iterator over female crossing overs"""
        return (
            co
            for co in ind.meioses.values()  # noqa
            for ind in self.parents
            if ind.sex == FEMALE
        )

    def add_parent(self, pednode):
        try:
            par = self.parents[pednode.indiv]
        except KeyError:
            par = Parent(pednode.indiv, pednode.sex)
            self.parents[pednode.indiv] = par
        else:
            logger.warning(
                f"Trying to create the parent {pednode.indiv} multiple times"
            )

    @property
    def genome_size(self):
        res = 0
        for reg in self.phaser.regions:
            res += self.phaser.data["variants"][reg]["POS"][-1]
        return res

    @staticmethod
    def get_crossovers(si, call=0.999):
        """Find crossover boundaries from segregation indicators

        Arguments
        ---------
        - si: np.array( (2,L) )
           Segragation indicator. si[0,] is the best phase solution,
           si[1,] is the probability that the phase is correct. At
           each phase switch, determines the boundaries such that the
           phase probabilty is > call
        - call: float
           Minimum probability to consider a phase resolved.

        Returns
        -------

        list[ [l,r] ]
           list of crossover boundaries found (left, right) = Indices in
           the array.

        """
        best_guess = np.array([x[0] for x in si])
        phase_prob = np.array([x[1] for x in si])
        co_loc = np.asarray((best_guess[1:] - best_guess[:-1]) != 0).nonzero()[
            0
        ]  # noqa
        nco = len(co_loc)
        logger.debug("Found {nco} candidate crossovers")
        res = []
        if nco == 0:
            return res
        for ell in co_loc:
            left = ell
            while phase_prob[left] < call:
                if left == 0:
                    break
                left -= 1
            right = ell + 1
            while phase_prob[right] < call:
                if right == len(best_guess) - 1:
                    break
                right += 1
            if best_guess[left] != best_guess[right]:
                res.append((left, right))
        #  returns a set to resolve local clusters of crossovers
        return set(res)

    @staticmethod
    def min_interval_size(nmeio, recrate=1, alpha=0.05):
        """Determine minimal interval size to estimate recombination rates

        Assuming a Poisson distributed number of crossovers in an
        interval of size L.  The probability of sampling 0 crossovers
        among nmeio meioses is:

        p_0 = exp( - recrate * L * nmeio)

        The minimal interval size is the one that satisfies p_0 =
        alpha, or:

        L = -log(alpha)/(recrate*nmeio)

        Arguments
        ---------
        nmeio: int
          number of meioses
        recracte: float
          recombination rate (in cM/Mb)
        alpha: float
          precision parameter ( P(k=0)<alpha)

        Returns
        -------
        float
           minimum interval size

        """
        return 1e8 * np.log(alpha) / (recrate * nmeio)

    def run(self, recrate, call):
        """Run the recombination analysis"""
        logger.info("Finding recombinations")
        self.identify_crossovers(recrate=recrate, call=call)

    @staticmethod
    def get_chromosome_pair(genotype, gametes, segregations, seg_probs):
        si_pat = list(zip(segregations[0], seg_probs[0]))
        si_mat = list(zip(segregations[1], seg_probs[1]))
        chp = family_phaser.ChromosomePair.from_phase_data(
            genotype, gametes[0], gametes[1], si_pat, si_mat
        )
        return chp

    def set_informative_meioses(self):
        """Identify informative meioses for each parent"""
        logger.info("Set Informative meioses")
        # sample indices
        smpidx = {}
        for i, name in enumerate(self.phaser.data["samples"]):
            smpidx[name] = i
        for reg in self.phaser.regions:
            logger.info(f"Working on region {reg}")
            logger.info("Grabbing data")
            pos = self.phaser.data["variants"][reg]["POS"]
            mids = np.array([0.5 * (x + y) for x, y in zip(pos[:-1], pos[1:])])
            chrom = reg2chr(reg)
            genotypes = np.array(self.phaser.data["genotypes"][reg])
            gametes = np.array(self.phaser.data["phases"][reg]["gametes"])

            for indiv, par in self.parents.items():
                logger.info(f"Parent {indiv}")
                idx = smpidx[indiv]
                par_resolved = (genotypes[idx] == 1) & (
                    (gametes[idx][0] > 0) | (gametes[idx][1] > 0)
                )  # noqa
                node = self.phaser.pedigree.nodes[indiv]
                n_meio_info = np.zeros(len(pos) - 1, dtype=int)
                for c in node.children:
                    idx = smpidx[c.indiv]
                    par.meioses[c.indiv] = []
                    child_phased = (gametes[idx][0] > 0) | (gametes[idx][1] > 0)
                    combin_info = par_resolved & child_phased
                    infomk = combin_info.nonzero()[0]
                    if len(infomk) > 0:
                        infomk_l = min(infomk)
                        infomk_r = max(infomk)
                        n_meio_info[infomk_l:infomk_r] += 1
                        self.size_covered[node.sex][chrom] += (
                            pos[infomk_r] - pos[infomk_l]
                        ) * 1e-6
                    logger.debug(
                        f"\t Meiosis -> {c.indiv:24} "
                        f"{round( (pos[infomk_r]-pos[infomk_l])*1e-3)} Kb"
                    )
                par.set_n_info_meioses(chrom, mids, n_meio_info)
            for sex in self.size_covered:
                logger.info(
                    f"sex:{sex} "
                    f"chrom:{chrom} "
                    f"size:{self.size_covered[sex][chrom]} Mb"
                )

    def identify_crossovers(self, recrate=1, call=0.99):
        logger.info("Gathering crossovers")
        smpidx = {}
        for i, name in enumerate(self.phaser.data["samples"]):
            smpidx[name] = i

        for reg in self.phaser.regions:
            logger.info(f"Working on: {reg}")
            chrom = reg2chr(reg)

            pos = self.phaser.data["variants"][reg]["POS"]
            genotypes = np.array(self.phaser.data["genotypes"][reg])
            gametes = np.array(self.phaser.data["phases"][reg]["gametes"])
            segregations = np.array(self.phaser.data["phases"][reg]["segregations"])
            segprobs = np.array(self.phaser.data["phases"][reg]["seg_probs"])

            for node in self.phaser.pedigree:
                sys.stdout.write(f" --> {node.indiv:>24}\r")
                sys.stdout.flush()
                logger.debug(
                    f"{node.indiv} -- [ "
                    f"sex:{node.sex} "
                    f"gen:{node.gen} "
                    f"par:{(node.father!=None)+(node.mother!=None)} "
                    f"off:{len(node.children)} ]"
                )
                idx = smpidx[node.indiv]

                if node.father:
                    try:
                        par = self.parents[node.father.indiv]
                        idx_pat = smpidx[node.father.indiv]
                    except KeyError:
                        logger.debug(f"Cannot find {node.father.indiv}")
                        pass
                    else:
                        # crossovers
                        logger.debug("Finding paternal crossovers")
                        si_pat = list(
                            zip(
                                segregations[
                                    idx,
                                    0,
                                ],
                                segprobs[
                                    idx,
                                    0,
                                ],
                            )
                        )
                        cos = self.get_crossovers(si_pat, call=call)
                        for x, y in cos:
                            par.add_offspring_CO(node.indiv, chrom, pos[x], pos[y])
                        # coverage
                        par_resolved = (genotypes[idx_pat] == 1) & (
                            (gametes[idx_pat][0] > -1) & (gametes[idx_pat][1] > -1)
                        )  # noqa
                        child_phased = gametes[idx][0] > -1
                        combin_info = par_resolved & child_phased
                        infomk = combin_info.nonzero()[0]
                        if len(infomk) > 0:
                            infomk_l = min(infomk)
                            infomk_r = max(infomk)
                            par.add_offspring_coverage(
                                node.indiv, chrom, pos[infomk_l], pos[infomk_r]
                            )
                if node.mother:
                    try:
                        par = self.parents[node.mother.indiv]
                        idx_mat = smpidx[node.mother.indiv]
                    except KeyError:
                        pass
                    else:
                        si_mat = list(
                            zip(
                                segregations[
                                    idx,
                                    1,
                                ],
                                segprobs[
                                    idx,
                                    1,
                                ],
                            )
                        )
                        cos = self.get_crossovers(si_mat)
                        for x, y in cos:
                            par.add_offspring_CO(node.indiv, chrom, pos[x], pos[y])
                        # coverage
                        par_resolved = (genotypes[idx_mat] == 1) & (
                            (gametes[idx_mat][0] > -1) & (gametes[idx_mat][1] > -1)
                        )  # noqa
                        child_phased = gametes[idx][1] > -1  # noqa
                        combin_info = par_resolved & child_phased
                        infomk = combin_info.nonzero()[0]
                        if len(infomk) > 0:
                            infomk_l = min(infomk)
                            infomk_r = max(infomk)
                            par.add_offspring_coverage(
                                node.indiv, chrom, pos[infomk_l], pos[infomk_r]
                            )

    def write_results(self):
        with open(self.phaser.prefix + "_yapp_recomb_coverage.txt", "w") as fout:
            print("parent sex offspring chrom left right", file=fout)
            for name, par in self.parents.items():
                for off in par.coverage:
                    for chrom in par.coverage[off]:
                        print(
                            f"{name} "
                            f"{((par.sex==None) and 'U') or ((par.sex==MALE) and 'M' or 'F')} "  # noqa
                            f"{off} "
                            f"{chrom} "
                            f"{par.coverage[off][chrom][0]} "
                            f"{par.coverage[off][chrom][1]}",
                            file=fout,
                        )
        with open(self.phaser.prefix + "_yapp_recombinations.txt", "w") as fout:
            print("parent sex offspring chrom left right", file=fout)
            for name, par in self.parents.items():
                for off in par.meioses:
                    for co in sorted(par.meioses[off]):
                        print(
                            f"{name} "
                            f"{((par.sex==None) and 'U') or ((par.sex==MALE) and 'M' or 'F')} "  # noqa
                            f"{off} "
                            f"{co.chrom} {co.left} {co.right}",
                            file=fout,
                        )


class Parent:
    """Class for storing information for each parent

    Attributes
    ----------
    - name: str
        identifier for the parent
    - sex: int or None
        sex of the parent ( 0: Male, 1: Female)
    - meioses: dict( str: list of CrossingOver objects)
        meioses of the individual.
    - coverage: dict( str: dict of ( int, int))
        informative genome of each offspring(key).
        Values are [chromosome]=(left,right)
    - nmeio_info: function: x(chrom,pos) -> int
        a function that returns the number of meiosis at a given
        genomic coordinate
    """

    def __init__(self, name, sex=None):
        self.name = name
        self.sex = sex
        self.meioses = defaultdict(list)
        self.coverage = defaultdict(dict)
        self._nmeio_info = None

    @property
    def nmeioses(self):
        return len(self.meioses)

    @property
    def nb_CO_meioses(self):
        """
        List of the number of crossovers for each meiosis
        """
        return [len(v) for v in self.meioses.values()]

    @property
    def nb_CO_tot(self):
        """
        Total number of crossing over in all meioses
        """
        return np.sum(self.nb_CO_meioses)

    def get_offspring_CO(self, name):
        """
        Get the list of CO for offspring with name *name*
        """
        return self.meioses[name]

    def add_offspring_CO(self, name, chro, left, right):
        """
        Add a crossing over in offspring *name*
        on chromosome *chro* between *left* and *right*
        """
        myco = CrossingOver(chro, left, right)
        self.meioses[name].append(myco)

    def add_offspring_coverage(self, name, chro, left, right):
        """
        Add coveage for offspring *name*
        on chromosome chro between left and right
        """
        self.coverage[name][chro] = (left, right)

    def n_info_meioses(self, chrom, pos):
        """Get the number of informative meioses for the parent
        at position pos on chromosome chrom.

        Arguments
        ---------
        - chrom: int
            Chromosome
        - pos: int
            Position

        Returns
        -------
        int
           number of informative meioses
        """
        try:
            return self._nmeio_info(chrom, pos)
        except TypeError:
            return self.nmeioses

    def set_n_info_meioses(self, chrom, positions, values):
        """Enter information on the number of informative meioses on
        a chromosome, at a set of positions.

        Arguments
        ---------
        - chrom: int
            chromosome
        - positions: array of int
            Positions at which the number of informative meioses is known
        - values: array of int
            Number of informative meioses at each position
        """
        if self._nmeio_info is None:
            self._nmeio_info = Nmeioses(self.nmeioses)
        self._nmeio_info.add_predictor(chrom, positions, values)

    def n_info_meioses_seg(self, chrom, left, right):
        return max(self.n_info_meioses(chrom, left), self.n_info_meioses(chrom, right))

    def oi_xi(self, chrom, w_left, w_right):
        """Computes probabilities that each crossing over in the parent
        occurs in genomic region on chromosome *chrom* between
        positions *w_left* and *w_right*.

        Returns a tuple with entries:
        -- list of contributions for each CO
        -- number of informative meioses for the parent in the region

        """
        contrib = []
        for m in self.meioses.values():
            for co in m:
                if co.chro == chrom and not (
                    (w_right < co.left) or (w_left > co.right)
                ):  # noqa
                    contrib += [co.oi_xi(w_left, w_right)]
        return (contrib, self.n_info_meioses_seg(chrom, w_left, w_right))


class Nmeioses(collections.abc.Callable):
    """Class offering a callable interface to interpolate the number of
    informative meioses from observed data.

    Usage
    -----
    Nmeioses(chrom, pos) -> int

    Nmeioses.add_predictor(chrom, positions, values): set up a 1D interpolator
    for chromosome chrom from observed (positions , values) points.

    Returns
    -------
    int
       Interpolated number of meioses. 0 if outside training range

    """

    def __init__(self, default_value):
        self.default = int(default_value)
        self.predictors = defaultdict(lambda: lambda x: self.default)

    def __call__(self, chrom, pos):
        try:
            return int(np.ceil(self.predictors[chrom](pos)))
        except ValueError:
            return 0

    def add_predictor(self, chrom, positions, values):
        self.predictors[chrom] = interp1d(positions, values, fill_value=0)


class CrossingOver:
    """
    Class to store crossing over information

    Attributes:
    -- chro: chromosome
    -- left: position of the marker on the left side
    -- right: position of the marker on the right side
    """

    def __init__(self, chrom, left, right):
        assert right > left
        self.chrom = chrom
        self.left = left
        self.right = right

    def __lt__(self, other):
        return (self.chrom, self.left, self.right) < (
            other.chrom,
            other.left,
            other.right,
        )  # noqa

    @property
    def size(self):
        return self.right - self.left

    def oi_xi(self, w_left, w_right):
        """Computes the probability that the crossing over occured n the
        window between w_left and w_right

        """
        if (w_right < self.left) or (w_left > self.right):
            # no overlap
            # wl------wr               wl----------wr
            #              sl-------sr
            return 0
        elif (w_left <= self.left) and (self.right <= w_right):
            # co included in window
            # wl---------------------------wr
            #       sl---------------sr
            return 1
        elif (self.left <= w_left) and (w_right <= self.right):
            # window is included in co
            #       wl------wr
            # sl---------------------sr
            return float(w_right - w_left) / (self.right - self.left)
        elif w_left < self.left:
            # we know w_right<self.right as other case is treated above
            # wl-----------------wr
            #     sl------------------sr
            return float(w_right - self.left) / (self.right - self.left)
        else:
            # only case left
            #     wl-----------------wr
            # sl--------------sr
            try:
                assert (self.left <= w_left) and (self.right < w_right)
            except AssertionError:
                print(self.right, w_right, self.left, w_left)
                raise
            return float(self.right - w_left) / (self.right - self.left)


def main(args):
    prfx = args.prfx

    phaser_db = prfx + "_yapp.db"
    analyzer = RecombAnalyser(phaser_db)
    analyzer.run(recrate=args.rho, call=args.minsp)
    analyzer.write_results()
