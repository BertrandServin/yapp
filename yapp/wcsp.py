''' Phasing Half-Sib families using the WCSP approach of Favier(2011)

Test data corresponds to phase information:

T1: 1 * * 0 1 * 1
T2: * * * 0 1 * 0
T3: 0 * * * 0 * 0

Informative pairs of markers:

T1: (0,3),(3,4),(4,6)
T2: (3,4),(4,6)
T3: (0,4),(4,6)

Genetic distance is assumed to be 0.1 cM between each marker pair

'''
from collections import defaultdict
import numpy as np
import Numberjack as nj

test_phase = {'T1': [1, -1, -1, 0, 1, -1, 1],
              'T2': [-1, -1, -1, 0, 1, -1, 0],
              'T3': [0, -1, -1, -1, 0, -1, 0],
              'T4': [-1, -1, -1, -1, -1, -1, -1]}


class PhaseData(object):
    '''
    Class to format phase information data into WCSP variables

    Parameters:
    -----------
     -- rawphase: dict of phase info, keys are offspring names, values
                  are phase vectors (0,1,-1)
     -- mkpos: list of marker positions in centiMorgans
               (optional, if not given assumes 0.01 cM between markers)

     Attributes:
     -----------
     -- rawphase: init data
     -- info_mk: numpy array of informative markers indices
     -- info_pairs: dict with pairs of informative markers
            markers as keys and [N+,N-] as values.

     TODO:
     -----
     -- genetic map

    '''

    def __init__(self, rawphase, mkpos=None):
        self.rawphase = rawphase
        if mkpos is None:
            self.mkpos = None
            self.recombination = self.recrate
        else:
            self.mkpos = mkpos
            self.recombination = self.recrate_from_pos
        # info_pairs stores pairs of inform. markers as keys
        # and [N+,N-] as values.
        self.info_pairs = {}
        # For each offspring
        for k, v in self.rawphase.items():
            # get informative markers ...
            # infomk=[i for i,x in enumerate(v) if x>=0]
            v = np.array(v)
            infomk = np.where(v >= 0)[0]
            if len(infomk) < 2:
                continue
            # and corresponding successive pairs
            mypairs = zip(infomk[:-1], infomk[1:])
            # for each pair
            for p in mypairs:
                # get phase information
                npair = int(v[p[0]] == v[p[1]])  # + or - phase
                # adds it to the pool
                try:
                    Nvec = self.info_pairs[p]
                except KeyError:
                    self.info_pairs[p] = [0, 0]
                    Nvec = self.info_pairs[p]
                Nvec[1-npair] += 1
        # informative markers are all mks seen in pairs
        info_mk = defaultdict(int)
        for k in self.info_pairs.keys():
            info_mk[k[0]] += 1
            info_mk[k[1]] += 1
        self.info_mk = sorted(info_mk.keys())

    def recrate_from_pos(self, i, j):
        dtot = abs(self.mkpos[i]-self.mkpos[j])
        return 0.5*(1-np.exp(-dtot/50))

    @staticmethod
    def recrate(i, j, d=0.01):
        '''
        returns rec. rate between marker indices i and j
        assuming each adj. markers are separeted by d (default 0.01) cM
        '''
        dtot = abs(i-j)*d
        return 0.5*(1-np.exp(-dtot/50))


class PhaseSolver(object):
    '''
    A class to Solve the phase of a parent.

    Parameters:
    -----------

    -- mk_list : list of markers
    -- mk_pairs : dict of pairs of markers (see PhaseData class)
    -- recmap_f : function that takes as input a pair of markers
                  and returns their rec. rate.
    '''

    def __init__(self, mk_list, mk_pairs, recmap_f):
        self.mk = list(mk_list)
        self.pairs = mk_pairs
        self.recf = recmap_f
        self.L = len(self.mk)
        try:
            assert self.L > 1
        except AssertionError:
            raise ValueError("WCSP problem must have size > 1")
        # Initialize Optim. model
        # Creates an array of phase indicators
        self.Variables = nj.VarArray(self.L)
        # Init model with simple constraint Var[0]==False
        self.init_constraint = self.Variables[0] == 0
        self.constraints = []

    def add_constraints(self):
        ''' Build constraints from marker pairs
        '''
        for p in sorted(self.pairs):
            Nkl = self.pairs[p]
            # gt mk indices
            k = self.mk.index(p[0])
            ell = self.mk.index(p[1])
            # ger recomb rate
            rkl = self.recf(p[0], p[1])
            assert rkl > 0
            # get cost
            Wkl = (Nkl[0]-Nkl[1])*np.log((1-rkl)/rkl)
            # create constraints
            hk = self.Variables[k]
            hl = self.Variables[ell]
            if Wkl < 0:
                cost = [-Wkl, 0, 0, -Wkl]
            else:
                cost = [0, Wkl, Wkl, 0]
            cost = [int(x) for x in cost]
            self.constraints.append(nj.PostBinary(hk, hl, cost))
        # print("Constraints:",*self.constraints,sep='\n')

    def solve(self, verbose=0):
        model = nj.Model()
        for c in self.constraints:
            model.add(c)
        solver = model.load('Toulbar2')
        solver.setVerbosity(verbose)
        solver.setOption('updateUb', str(1000000))
        solver.setOption('btdMode', 1)
        solver.solve()
        return [self.Variables[m].get_value() for m in range(self.L)]


if __name__ == '__main__':
    T = PhaseData(test_phase)
    print("Info pairs:", *T.info_pairs.items())
    print("Info mk:", T.info_mk)
    S = PhaseSolver(T.info_mk, T.info_pairs, T.recombination)
    S.add_constraints()
    phase = S.solve()
    print("Resolved phase:", *phase)
