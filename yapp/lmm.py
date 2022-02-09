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

from functools import partial
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.linalg import block_diag
from scipy.stats import f as fdist
from scipy.stats import chi2


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
    logL = cste - 0.5 ( logdet(C) + (N-P) log(Yt.Vi.Y - Bt.C.B) )
    """
    N = Y.shape[0]
    P = X.shape[1]
    Vi = np.diag(1 / (1 + diagG * h / (1 - h)))
    C = X.T @ Vi @ X
    B = np.linalg.solve(C, X.T @ Vi @ Y)
    yPy = Y.T @ Vi @ Y - B.T @ C @ B
    _, ldetC = np.linalg.slogdet(C)
    ldetVi = np.sum(np.log(1 / (1 + diagG * h / (1 - h))))
    return -0.5 * (ldetC - ldetVi + (N - P) * np.log(yPy))


def Wald_Gdiag(h, Y, X, diagG):
    """Computes the Wald statistic of fixed effects of the animal
    model when G is diagonal
    """
    N = Y.shape[0]
    P = X.shape[1]
    Vi = np.diag(1 / (1 + diagG * h / (1 - h)))
    C = X.T @ Vi @ X
    B = np.linalg.solve(C, X.T @ Vi @ Y)
    return (B.T @ C @ B, P - 1, N - P)


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
    vg = h / (1 - h)
    L = eigVal[eigVal > valmin]
    W = eigVec[:, eigVal > valmin]
    K = W.shape[1]
    # print("*** K=", K)
    # print(min(eigVal),max(eigVal))
    ## approximate inverse(V) using low rank matrix (Mischenko et al. 2007)
    LRD = np.diag(1 / (vg + 1 / L))
    Vi = np.eye(N) - vg * W @ LRD @ W.T
    ## then as usual
    C = X.T @ Vi @ X
    B = np.linalg.solve(C, X.T @ Vi @ Y)
    yPy = Y.T @ Vi @ Y - B.T @ C @ B
    _, ldetC = np.linalg.slogdet(C)
    ldetVi = -np.sum(np.log(vg * L + 1))  ## should we approximate this
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
    """
    N = Y.shape[0]
    P = X.shape[1]
    K = Z.shape[1]
    Vi = np.diag(1 / (1 + diagG * (1 - p) * h / (1 - h)))
    W = np.hstack([X, Z])
    C = W.T @ Vi @ W
    C[P:, P:] += np.eye(K) * (1 - h) / (p * h)
    B = np.linalg.solve(C, W.T @ Vi @ Y)
    yPy = Y.T @ Vi @ Y - B.T @ C @ B
    _, ldetC = np.linalg.slogdet(C)
    ldetVi = np.sum(np.log(1 / (1 + diagG * (1 - p) * h / (1 - h))))
    return -0.5 * (ldetC - ldetVi + (N - P) * np.log(yPy))


if __name__ == "__main__":
    """Simulate stuff to test funcs"""

    rng = np.random.default_rng()
    Nfam = 10
    Nind = 100
    N = Nfam * Nind
    K = 2 * Nfam
    nsim = 1
    ## pedigree kinship -> does not work for testing linkage effects
    ##G = np.kron(np.eye(Nfam),np.ones((Nind,Nind))*0.25)
    fG = []
    ## create pedGRM-like coefficients with gametic variance
    for ifam in range(Nfam):
        a = rng.normal(loc=0.25, scale=0.05, size=(Nind, Nind))
        fG.append(np.tril(a) + np.tril(a, k=-1).T)
    G = block_diag(*fG)
    print(np.round(G, 2))
    np.fill_diagonal(G, 1)
    dG, U = np.linalg.eigh(G)
    with open("../test/h.txt", "w") as fout:
        print("h hreml success ll se", file=fout)
        for isim in range(nsim):
            # h = rng.random()
            # p = rng.random()
            h = 0.5
            p = 0.00001
            print(f"Simulation {isim+1} with h={h} and p={p}")
            ## polygenic component
            w = rng.multivariate_normal(mean=np.zeros(N), cov=G * (1 - p) * h / (1 - h))
            ## Fixed Effects = intercept
            # beta = np.array([2,0.01])
            # X = np.block([[np.ones(N)],[np.random.randint(0,2,N)]]).T
            X = np.block([[np.ones(N)]]).reshape(N, 1)
            beta = np.array([0])
            ## QTL Effect
            Z = np.zeros((N, 2 * Nfam))
            irow = np.arange(N)
            fam, row_within_fam = divmod(irow, Nind)
            div = np.sum(divmod(Nind, 2))
            icol = 2 * fam + (row_within_fam // div)
            Z[irow, icol] = 1
            ##Z = rng.binomial(1,0.5, (N,2*Nfam))
            u = rng.normal(scale=np.sqrt(p * h / (1 - h)), size=Z.shape[1])
            Zu = Z @ u
            ## Phenotypes
            Ynull = X @ beta + w + np.random.randn(N)
            Y = Ynull + Zu
            ##Y = X@beta + Zu + w + np.random.randn(N)
            ## Null Model
            mylik_null = partial(relik_Gdiag, Y=U.T @ Y, X=U.T @ X, diagG=dG)
            optlik_null = lambda x: -mylik_null(x)
            hmax_null = minimize(
                optlik_null, [0.4], bounds=[(1e-3, 1 - 1e-3)], options={"disp": False}
            )
            print(
                f"*****   H: {hmax_null.x[0]} se {np.sqrt(hmax_null.hess_inv.todense()[0,0])}"
            )
            ## QTL model fixed effects
            ##W = np.hstack([X,Z])
            mylik_qtl = partial(
                relik_Gdiag, Y=U.T @ Y - U.T @ X @ beta, X=U.T @ Z, diagG=dG
            )
            optlik_qtl = lambda x: -mylik_qtl(x)
            hmax_qtl = minimize(
                optlik_qtl, [0.4], bounds=[(1e-3, 1 - 1e-3)], options={"disp": False}
            )
            print(
                f"*****   H: {hmax_qtl.x[0]} se {np.sqrt(hmax_qtl.hess_inv.todense()[0,0])}"
            )
            wald = Wald_Gdiag(hmax_qtl.x[0], Y=U.T @ Y, X=U.T @ Z, diagG=dG)
            print(
                f"Wald statistic : {wald}, p = {fdist.sf(wald[0]/wald[1],wald[1],wald[2])} "
            )
            continue
            ## QTL model random effects
            mylik_qtl_2 = partial(
                relik_Gdiag_Z,
                Y=U.T @ Y - U.T @ X @ beta,
                X=U.T @ X,
                Z=U.T @ Z,
                diagG=dG,
            )
            optlik_qtl_2 = lambda x: -mylik_qtl_2(x[0], x[1])
            hmax_qtl_2 = minimize(
                optlik_qtl_2,
                [0.1, 0.1],
                bounds=[(1e-3, 1 - 1e-3), (1e-3, 1 - 1e-3)],
                options={"disp": False},
            )
            H = hmax_qtl_2.hess_inv.todense()
            print(f"Hessian : {H}")
            print(f"*****  H2: {hmax_qtl_2.x[1]} se {H[1,1]}")
            print(f"***** RHO: {hmax_qtl_2.x[0]} se {H[0,0]}")
            #
            # print(np.round(h,3),
            #       np.round(hmax.x[0],3),
            #       hmax.success,
            #       -hmax.fun,
            #         np.sqrt(hmax.hess_inv.todense()[0,0]),
            #         file=fout)

    # h = 0.5
    # genetic = True
    # if genetic:
    #     ## 1. Nfam families of halfsibs
    #     G = np.kron(np.eye(Nfam),np.ones((Nind,Nind))*0.25)
    #     np.fill_diagonal(G,1)
    #     dG,U = np.linalg.eigh(G)
    #     Zu = rng.multivariate_normal(mean = np.zeros(N),cov=G*h/(1-h))
    # else:
    #     ## 2. random vectors
    #     Z = rng.normal(size=(N,N))
    #     dG,U = np.linalg.eigh(Z@Z.T)
    #     u = rng.normal(size=N,scale=np.sqrt(h/(1-h)))
    #     Zu = Z@u

    # ## Fit REML
    # X = np.block([[np.ones(N)],[np.random.randint(0,2,N)]]).T
    # beta = np.array([2,0.01])
    # Y = X@beta + Zu + np.random.randn(N)
    # hh = np.linspace(0.01,0.9999)
    # with open('../test/h.txt','w') as fout:
    #     print('hh ll.diag',file=fout)
    #     mylik = partial(relik_Gdiag_correct, Y=U.T@Y, X=U.T@X, diagG=dG)
    #     optlik = lambda x: -mylik(x)
    #     hmax = minimize(optlik,[0.4],bounds=[(1e-3,1-1e-3)], options={"disp":True})
    #     print(f" h2 = {hmax.x}, success = {hmax.success}")
    #     print(np.round(hmax.x[0],3), -hmax.fun, file=fout)
    #     for x in hh:
    #         res = relik_Gdiag_correct(x, U.T@Y, U.T@X, dG)
    #         print(np.round(x,2), res,file=fout)
