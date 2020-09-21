# yapp : Yet Another Phasing Program

yapp is a program that includes a set of utilities to work on high
density genetic data in pedigrees. It is similar to other existing
softwares such as LINKPHASE, FIMPUTE, AlphaPhase ... . I
developed it for my own research to have a tool that is opensource and
that I can tweak to my own needs. 

If you find it useful for you, drop me a note on twitter
@BertrandServin. If you want to contribute ideas or code, you are
welcome if you know python :) 

yapp and its utilities implement models and methods that have been
invented by other people. If you use yapp it is important that you
acknowledge the work of these authors by citing the appropriate
references. These will depend on the way you use the software. The
documentation below provides the relevant citations.

## Installation

The code includes a `setup.py` script that should take care of
installing yapp and *most* its dependencies. However some of them are
optional and must be installed if needed:

- ray : ray is an open source frame for building distributed
  applications. It is really only needed in yapp for parallelizing
  tasks on a computer cluster. If you don't plan to do that, or are
  satisfied using the multiprocessing approach you don't need to
  install it.
  
- numberjack : a python platform for combinatorial optimization. it is
  a python module so can be installed via pip. However the current
  setup script is buggy and installation must be tweaked. TODO: write
  instructions for installing numberjack properly with python 3.8.
  
I will upload a proper package on pypi when I have finished
implementing my initial ideas on the software.

## Usage

`yapp` has a command line interface that is used to launch different
commands. The basic syntax is :

```bash
yapp <command> <args1, args2, ..., argsN>
```

Available commands are:

### `phase`

The `phase` command is used to infer gametic phase and segregation
indicators in a genotyped pedigree. Its input consists of three files
a [VCF file](http://samtools.github.io/hts-specs/VCFv4.2.pdf), its
index obtained with [`tabix`](http://www.htslib.org/doc/tabix.html), and a [FAM
file](https://www.cog-genomics.org/plink/1.9/formats#fam) with family
information. The usage is:

```bash
yapp phase <prfx>
```

where `prfx` is the prefix of **all** input files :
`<path/to/prefix>.fam` , `<path/to/prefix>.vcf.gz` ,
`<path/to/prefix>vcf.gz.tbi`. Note that when exporting a `plink` bed
file to VCF, you must do so using `--recode vcf-iid` so that sample
names in the resulting VCF do no include the family-ID.

The events are logged in `<path/to/prefix>_yapp_phase.log`. The output
files produced are a phased VCF `<path/to/prefix>_phased.vcf.gz` and a
binary file `<path/to/prefix>_yapp.db`. This binary file is useful
for conducting analyses with other `yapp` commands.

#### Citation
`yapp phase` uses a Weighted Constraints Satisfaction Problem solver,
[https://miat.inrae.fr/toulbar2/](ToulBar2), to infer parental phase
from transmitted gametes. This idea was developped by Aurélie Favier
during her PhD with Simon de Givry and Andres Legarra.

[Favier, Aurélie (2011). Décompositions fonctionnelles et
structurelles dans les modèles graphiques probabilistes appliquées à
la reconstruction d'haplotypes.](http://thesesups.ups-tlse.fr/1527/)

### `recomb`
