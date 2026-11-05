# Representation of haplotype diversity from yapp output



The script below should allow to plot a representation of the haplotype diversity from the ouput of the `yapp origins` command. 

It is still a tad rough.

```R
library(tidyverse)
library(Matrix)
library(RSpectra)

prefix = "<yapp_prefix>"
## fill in region information
chrom = ''
left = 0
right = 0
coordinates = c(left,right)

################################ SETUP DATA ############################

##integration du fichier sortie Yap_pedGRM.txt
pgrm = read_table(paste0(prefix,'_yapp_GRM.txt'),col_types='ffd')
##grm = read_table('GenoYappAutosome_yapp_GRM.txt',col_types='ffd')
##grm = read.table("GenoYappAutosome_yapp_pedGRM.txt",h = TRUE, sep = " ", colClasses = c("factor","factor","numeric"))


##integration du fam
fam = read_table(paste0(prefix,'.fam'),
   col_names=c('FID','ID','PID','MID','sex','phe'),
   col_types=c('fffffd')
                ) %>% arrange(PID) 


## haplotype diversity
library(reticulate)
zarr = import("zarr")
np = import("numpy")
yapp = zarr$open(paste0(prefix,".zarr"))
yapp.samples=np$array(yapp['samples'])
key = paste0('variants/',chrom,'/POS')
yapp.ps = np$array(yapp[key])
key = paste0('variants/',chrom,'/ID')
yapp.rs = np$array(yapp[key])

## get haplotypes
key = paste0('linkage/',chrom,'/ancestral_origins')
ao = np$array(yapp[key])
Ni = dim(ao)[1]
Ns = dim(ao)[3]
ao.h = ao
dim(ao.h) = c(2*Ni,Ns)

no = apply(ao.h, 2, function(x) { length(unique(x)) })
no.h = apply(ao.h, 1, function(x) {length(unique(x))})

## Number of ancestral haplotypes along the genome
plot(yapp.ps,no,type='s',lwd=2,xlab='Position', ylab="Number of ancestral haplotypes")
abline(v=coordinates,lwd=2,lty=3,col='gray')

coordinates.focal.idx =  which.min(abs(yapp.ps-mean(coordinates)))

span = 50
l = coordinates.focal.idx - span
r = coordinates.focal.idx + span

h.coordinates.covmat = ao.h[,l:r]

### This step is to get a good ordering of haplotypes so that the plot makes sens.
### It is a bit long to run if span is large
covmat = Matrix(data= 0,nrow=nrow(h.coordinates.covmat),ncol=nrow(h.coordinates.covmat), sparse=TRUE )
for (snp in 1:ncol(h.coordinates.covmat)) {
    print(snp)
    hvec = h.coordinates.covmat[,snp]
    ## levels of haplotypes
    print("Building table")
    orig = table(as.vector(hvec))
    orig = orig[orig>1]
    print("Relevel")
    h = factor(hvec,levels=c(names(orig),'-1'))
    h[is.na(h)] = '-1'
    print(table(h))
    print("Design Matrix")
    Z = sparse.model.matrix( ~ 0+h)
    print("ZZt")
    covmat = covmat + Z%*%t(Z)
    #covmat = covmat + tcrossprod(Z)
    print("Done")
}
covmat = covmat / ncol(h.coordinates.covmat)
## Extract the first to Principal Components of the local kinship matrix
res = eigs(covmat, 2)

h.coordinates = ao.h[,l:r]
colnames(h.coordinates) = yapp.rs[l:r]
h.coordinates = as_tibble(h.coordinates) %>% mutate (hap = factor(c(paste0(yapp.samples,"_1"),paste0(yapp.samples,"_2"))),
                                         pcpos = res$vectors[,1])
h.coordinates$hap = fct_reorder(h.coordinates$hap,h.coordinates$pcpos)
h.coordinates = pivot_longer(h.coordinates,cols=-(hap:pcpos))

snps = tibble(name = yapp.rs, pos = yapp.ps) %>% arrange(pos)
snps$idx = 1:nrow(snps)
snps = snps %>% mutate(right=lead(pos)-pos, left=pos-lag(pos),
                       w = case_when(
                           is.na(left) ~ 0.5*right,
                           is.na(right) ~ 0.5*left,
                           .default = 0.5*(left+right)),
                       l = case_when(
                           is.na(left) ~ pos,
                           .default = pos-0.5*left),
                       r = case_when(
                           is.na(right) ~ pos,
                           .default = pos+0.5*right))


h.coordinates =  left_join(h.coordinates, snps) %>% arrange(hap,pos)
h.coordinates$value = as.factor(h.coordinates$value)
h.coordinates = h.coordinates %>% mutate(ih = match(hap,levels(hap)))

library(pals)
col=as.vector(alphabet(26))
col=c(col,as.vector(alphabet2(26)))

plot.idx = h.coordinates %>% ggplot(aes(x=idx,y=hap,color=value,fill=value))+geom_tile() +
    scale_colour_manual(values=col,aesthetics = c("colour", "fill")) +
    xlab("Position (index)") + ylab("Haplotypes") + theme_minimal()

plot.pos = h.coordinates %>% ggplot(aes(x=pos,y=ih,color=value,fill=value)) +
    geom_rect(aes(xmin=l,xmax=r,
                  ymin=ih-0.5,
                  ymax=ih+0.5)) +
    xlab("Position (Mb)") + ylab("Haplotypes") +
    scale_colour_manual(values=col,aesthetics = c("colour", "fill")) +
##    scale_colour_viridis_d(aesthetics = c("colour", "fill")) +
    theme_minimal() + scale_x_continuous(labels=function(x){x*1e-6})# + guides(colour="none", fill="none")

plot.pos

library(patchwork)

plot.idx / plot.pos


```

