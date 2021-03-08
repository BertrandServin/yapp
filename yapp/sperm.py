import logging
from . import vcf, gamete, pedigree, MALE, FEMALE

logger = logging.getLogger(__name__)

def main(args):
    prfx=args.prfx
    vcf_file = f"{prfx}.vcf.gz"
    fam_file = f"{prfx}.fam"
    
    myvcf=VCF(vcf_file, gts012=True, strict_gt=True, lazy=True)
    ped = pedigree.Pedigree.from_fam_file(fam_file)
 
    
    return

