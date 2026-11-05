# How to LDLA with yapp

A classical GWAS tests the association between SNP genotypes and a phenotype (**LD approach**). The implicit assumption being that LD decreases rapidly around a SNP and associations signals don't spread too far. The problem is that population structure and kinship lead to false positives. These can be accomodated by using mixed-models that carry a "polygenic" component capturing the effect of other polymorphisms on the trait when testing a particular SNP. But, these approaches have typically been developed in the context of human population studies, with little kinship between participants. *When working in highly related individuals in pedigrees, the adequation of these models can be questioned*. 

An old approach to study genetic effects in pedigrees is to use Linkage Mapping (**LA approach**). The idea here is to follow transmission of parental haplotypes to offspring and test the effect of the transmitted haplotypes on the phenotype. Because there can be many parents, parental haplotypes are considered as random effects and so called "variance component" (VC) approaches have been proposed. The VC approaches have also a nice property in that the pedigree structure can be somewhat arbitrary, as long as parental haplotypes can be traced down the pedigree. The problem with the Linkage Mapping (LA) framework is that it doesn't model the fact that parental haplotypes can be partially Identical By Descent themselves. This is why combining the two approaches have been proposed for a long time but the practical implementation of such approaches have proven difficult.

In this document we will explain how one can perform LDLA mapping using a combination of the software`yapp` to model ancestral haplotypes and the R package `gaston` to fit linear mixed models (other packages for fitting LMMs might be used too)

The steps required to perform the LDLA analysis are:

1. run `yapp` to model haplotype diversity in a pedigreed population
2. read in the haplotype information in `R` and build matrices required to do the LDLA analysis.
3. Run LMM functions and automate it, run parallel jobs ... 

We will now adress these points

## 1. Model haplotype diversity in a pedigreed population with yapp

1. Run `yapp phase` to phase haplotypes and construct segregation indicators between parents and offspring
2. Run `yapp origins` with the latest version of yapp to 
   1. drop founder chromosomes down the pedigree
   2. model the IBD status of founder chromosomes using the Positional Burrows-Wheeler transform (Durbin 2014)

## 2. Read in haplotype and origins information in R

The script below shows how to perform an LDLA gwas after step 1 above have been done. It assumes that the phenotype is unique (no repeated measures) and that it is stored in the fam file. Repeated measures and phenotype files can be handled in principle by extending this script.

```R
library(tidyverse)
library(gaston)

## before running this script replace <prefix> with the prefix used for the 
## yapp IO files 

## GRM from yapp
grm = read_table('<prefix>_yapp_GRM.txt',col_types='ffd')

## fam file is the input file, we assume col 6 is the phenotype
## and is a double
fam = read_table('<prefix>.fam',
   col_names=c('FID','ID','PID','MID','sex','phe'),
   col_types=c('fffffd')
                ) %>% arrange(PID)

## do some data cleaning here
mydata = fam %>% drop_na(phe)
## you can add anything you want here ... 

## align the sub GRM with mydata
subgrm = grm %>% filter( (id1 %in% mydata$ID) & (id2%in%mydata$ID))
## fill in the symmetry
other_pair = subgrm %>% filter(id2!=id1) %>% mutate(tmp=id2,id2=id1,id1=tmp) %>% select(-tmp)
G = bind_rows(subgrm,other_pair) %>% pivot_wider(values_from=kinship,names_from=id2)
## G is a convenient data.frame / tibble, A is a good old matrix for computations
A = G %>% select(-id1) %>% as.matrix()
eigA = eigen(A) ## can take a while if you have many individuals here
## make sur mydata and G/A are aligned !!
mydata = mydata %>% mutate(ord = match(ID,G$id1)) %>% arrange(ord)

## LDLA per se

## we use reticulate to import the yapp results in R
library(reticulate)
## Here we assume yapp has been installed in a conda environment names "yapp"(duh !)
use_condaenv("yapp")
zarr = import("zarr")
np = import("numpy")
yapp = zarr$open("<prefix>.zarr")
yapp.samples=np$array(yapp['samples'])
yapp.sub = match(mydata$ID,yapp.samples)

## get origins
chromosome = '26'
key = paste0('linkage/',chromosome,'/ancestral_origins')
ao = np$array(yapp[key])
ao = ao[yapp.sub,,]

N = nrow(mydata)
## we only add one fiexed effect : the intercept 
X = as.matrix(rep(1,N)) 
Y = mydata$phe
## fit the null model
fit.H0 = lmm.aireml(Y,X,A)

########## LDLA Analysis per se. see below for Parallel computations ######
### NB: this can take a while depending on the data size, go grab a coffee.
result = list()
## dim(ao) is Nindiv x 2 x Nsnp
for (snp in 1:dim(ao)[3]) {
    hvec = ao[,,snp]
    ## levels of haplotypes
    orig = table(as.vector(hvec))
  	## we pool haplotypes that are only seen once in a single class
  	## You can adjust the minimum number of carriers here
    orig = orig[orig>1]
    h1 = factor(hvec[,1],levels=c(names(orig),'99999'))
    h2 = factor(hvec[,2],levels=c(names(orig),'99999'))
    h1[is.na(h1)] = '99999'
    h2[is.na(h2)] = '99999'
		## Build the design matrices
    Z1 = model.matrix( ~ 0+h1)
    Z2 = model.matrix( ~ 0+h2)
    Z = Z1 + Z2
    XX = cbind(X,Z)
  	
    ## fixed effect model is not good, use at your own risks
    ## fit.d = lmm.diago(Y,XX,eiA)
  
    ## two random effects model
    K2 = tcrossprod(Z)
    fit2 = lmm.aireml(Y,X,K = list(K2,A))
    result[[snp]] = fit2
}

rez = tibble(
        chr = chromosome,
        pos = np$array(yapp[paste0('variants/',chromosome,'/POS')]),
        logL  =sapply(result,"[[","logL"),
        lrt = 2*(logL-fit.H0$logL),
        pval = 0.5*(pchisq(lrt,df=1,lower.tail=F)),
        v.qtl  = sapply(result,"[[","tau")[1,],
        v.poly  = sapply(result,"[[","tau")[2,],
        v.e = sapply(result,"[[","sigma2")
    )

######## LDLA parallel calculations
## Run genome scan in parallel
library(iterators)
library(foreach)
library(doParallel)

ncpu = 40 ## |..|_
cl = parallel::makeForkCluster(ncpu)
doParallel::registerDoParallel(cl)

result = list()
nchrom = 26 ## bêê
for (chromosome in 1:nchrom) {
    key = paste0('linkage/',chromosome,'/ancestral_origins')
    ao = np$array(yapp[key])
    ao=ao[yapp.sub,,]
    print(paste("Working on chromosome ", chromosome))
    result[[chromosome]] = foreach(hvec = iapply(ao,3)) %dopar% {
        ## levels of haplotypes
        orig = table(as.vector(hvec))
        orig = orig[orig>1]
        h1 = factor(hvec[,1],levels=c(names(orig),'99999'))
        h2 = factor(hvec[,2],levels=c(names(orig),'99999'))
        h1[is.na(h1)] = '99999'
        h2[is.na(h2)] = '99999'

        Z1 = model.matrix( ~ 0+h1)
        Z2 = model.matrix(~ 0+h2)
        Z = Z1 + Z2

        XX = cbind(X,Z)
        ## fixed effect model
        ## fit.d = lmm.diago(Y,XX,eiA)
        ## two random effects model
        K2 = tcrossprod(Z)
        fit2 = lmm.aireml(Y,X,K = list(K2,A))
        fit2
    }
}
parallel::stopCluster(cl)

tbls = list()
for (chromosome in 1:nchrom) {
    tbls[[chromosome]] = tibble(
        chr = chromosome,
        pos = np$array(yapp[paste0('variants/',chromosome,'/POS')]),
        logL  =sapply(result[[chromosome]],"[[","logL"),
        lrt = 2*(logL-fit.H0$logL),
        pval = 0.5*(pchisq(lrt,df=1,lower.tail=F)),
        v.qtl  = sapply(result[[chromosome]],"[[","tau")[1,],
        v.poly  = sapply(result[[chromosome]],"[[","tau")[2,],
        v.e = sapply(result[[chromosome]],"[[","sigma2")
    )
}

bigtbl = map_dfr(tbls,bind_rows)
write.table(bigtbl, quote=F,row.names=F,file='GWAS-<prefix>.tsv')


```

