# -*- coding: utf-8 -*-
"""
Yapp module lmm.py

Functions to fit linear mixed-models for linkage and LDL analysis.  The
models considered are multivariate gaussian models of the form:

Y = XB + Zu + a + e

- XB are fixed effects terms
- a are the animal random effect (possibly with design matrix if 
  repeated measures) with a ~N(0,G.va)
- u is an additional random effect, typically of ancestral/founder 
  haplotypes, with design matrix Z.

Although some functions can be quite general, the philosophy is to target
specific models used in yapp. This is *not* the most computationaly
efficient module to do general mixed-model analysis.
"""

import numpy as np

def relik_animal_model(h, Y, X, Gi):
    """
    Computes the restricted likelihood of the animal model:

    Y = XB + a + e

    a ~N( 0, G ve.h/(1-h))
    e ~N( 0, I ve)

    Arguments:
    ----------
    - h : float
        heritability (h in ]0,1[)
    - Y : vector
        Observations
    - Gi : matrix like
        Inverse of G
    - X : matrix like
        Design Matrix of Fixed Effects.
    """
    pass

def relik_Gdiag(h, Y, X, diagG):
    """Computes the restricted likelihood of the animal model when G
    is diagonal
    
    Y = XB + w + e

    w ~ N(0, diagG ve.h/(1-h))
    e ~ N(0, I ve)
    
    This is obtained from the full matrix model by diagonalization of
    G through eigen decomposition and projection of the model on the
    PCs. 

    V(Y) is diagonal (I + h/(1-h) diagG) and easy to
    invert. So the restricted likelihood can be obtained as :

    C = Xt.Vi.X
    CB = Xt.Vi.Y

    L = P(Y|X,h) = \int P(Y|X,h,ve)P(ve) with P(ve) \propto 1/ve
    logL = cste - 0.5 ( logdet(C) + N log(Yt.Vi.Y - Bt.C.B) )
    """
    N = Y.shape[0]
    Vi = np.diag(1 / (1+ diagG * h/(1-h)))
    C = X.T @ Vi @ X
    B = np.linalg.solve(C,X.T @ Vi @ Y)
    yPy = Y.T@Vi@Y - B.T@C@B
    _,ldetC = np.linalg.slogdet(C)
    ldetVi = np.sum(np.log(1 / (1+ diagG * h/(1-h))))
    return -0.5 * (ldetC - ldetVi + N * np.log(yPy))

def relik_lowrank(h, Y, X, eigVal, eigVec, valmin=1):
    """Computes the restricted likelihood of the animal model when G
    is appromixated by (eigVec.sqrt(lambda))(eigVec.sqrt(lambda))'
    where only a subset of eigenValues are kept.
    
    In this case, with W=eigVec, L=eigVal,

    Vi \approx I_N - vg.W(1/L + I)^{-1}W' 

    C = Xt.Vi.X
    CB = Xt.Vi.Y

    L = P(Y|X,h) = \int P(Y|X,h,ve)P(ve) with P(ve) \propto 1/ve
    logL = cste - 0.5 ( logdet(C) + logdet(V) + N log(Yt.Vi.Y - Bt.C.B) )

    References:
    -----------
    - Mischenko et al. 2007, Efficient implementation of the AI-REML
    iteration for variance component QTL analysis

    """
    N = Y.shape[0]
    vg = h/(1-h)
    L = eigVal[eigVal>valmin]
    W  = eigVec[:,eigVal>valmin]
    K = W.shape[1]
    print("*** K=", K)
    print(min(eigVal),max(eigVal))
    ## approximate inverse(V) using low rank matrix (Mischenko et al. 2007)
    LRD = np.diag(1/(vg + 1/L))
    Vi = np.eye(N) - vg * W@LRD@W.T
    ## then as usual
    C = X.T @ Vi @ X
    B = np.linalg.solve(C,X.T @ Vi @ Y)
    yPy = Y.T @ Vi @ Y - B.T @ C @ B
    _,ldetC = np.linalg.slogdet(C)
    ldetVi = - np.sum(np.log(vg*L+1)) ## should we approximate this
                                      ## too ?
    return -0.5 * (ldetC - ldetVi + N * np.log(yPy))

def relik_Gdiag_Z(p, h, Y, X, Z, diagG):
    """Computes the restricted likelihood of the animal model when G 
    is diagonal and a second random effect is included with design 
    matrix Z.
    
    Y = XB + Zu + w + e

    w ~ N(0, diagG ve.(1-p).h/(1-h))
    u ~ N(0, Ik ve.p.h/(1-h))
    e ~ N(0, I ve)
    
    We rewrite the model as:
    Y = XB + Zu + r
    
    with r ~ N(O,R.ve) and R is diagonal (I + (1-p).h/(1-h) diagG). 
    with u ~ N(0,K.ve) and K is diagonal Ik.p.h/(1-h). 
    The mixed model equations are:
    
    C = [ [ XtRiX    ,    XtRiZ ] ,
          [ ZtRiX    , ZtRiZ + Ki]]

    C[B:u]t = [X:Z]t.Ri.Y

    TODO : figure out the likelihood
    """
    pass

if __name__=='__main__':
    '''Simulate stuff to test funcs'''

    rng = np.random.default_rng()
    Nfam = 30
    Nind = 100
    N = Nfam * Nind
    h = 0.8
    genetic = True
    if genetic:
        ## 1. Nfam families of halfsibs
        G = np.kron(np.eye(Nfam),np.ones((Nind,Nind))*0.25)
        np.fill_diagonal(G,1)
        dG,U = np.linalg.eigh(G)
        Zu = rng.multivariate_normal(mean = np.zeros(N),cov=G*h/(1-h))
    else:
        ## 2. random vectors
        Z = rng.normal(size=(N,N))
        dG,U = np.linalg.eigh(Z@Z.T)
        u = rng.normal(size=N,scale=np.sqrt(h/(1-h)))
        Zu = Z@u

    ## Fit REML
    X = np.block([[np.ones(N)],[np.random.randint(0,2,N)]]).T
    beta = np.array([2,0.01])
    Y = X@beta + Zu + np.random.randn(N)
    hh = np.linspace(0.01,0.99)
    with open('../test/h.txt','w') as fout:
        print('hh ll.lr ll.diag',file=fout)
        Ytilde = U.T@Y
        print("problem size", Ytilde.shape)
        for x in hh:
            res = relik_lowrank(x, Y, X, dG, U)
            res2 = relik_Gdiag(x, U.T@Y, U.T@X, dG)
            print(np.round(x,2), res,res2,file=fout)
        
