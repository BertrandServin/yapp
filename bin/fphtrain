#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle

try:
    import ray

    ray_available = True
    from fastphase import fastphase_ray as fphray
except ImportError:
    ray_available = False
from fastphase import fastphase
from yapp import vcf


def get_parser():
    parser = argparse.ArgumentParser(description="Train a fastphase model")
    parser.add_argument("vcf", metavar="vcf_file", help="read data from vcf")
    parser.add_argument("K", metavar="K", help="number of haplotype clusters", type=int)
    parser.add_argument(
        "-o",
        metavar="PREFIX",
        help="Use PREFIX for output files",
        default="fph",
        dest="prefix",
    )
    parser.add_argument(
        "--vcf-mode",
        metavar="M",
        help="data type, one of : " + ",".join(vcf.modes_avail),
        default=vcf.default_mode,
        choices=vcf.modes_avail,
        dest="vmode",
    )
    parser.add_argument(
        "-r",
        "--reg",
        metavar="region",
        help="region in which the model is trained",
        default=None,
        dest="region",
    )
    parser.add_argument(
        "-c",
        "--chunks",
        metavar="chunk_length",
        dest="chunklen",
        help="fit model on chunks of provided lengths",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--samples",
        metavar="IDs",
        help="comma separated list of samples to use",
        default=None,
    )
    parser.add_argument(
        "--keep",
        metavar="FILE",
        help="Keep samples with IDs listed in FILE",
        dest="smpfile",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--nem",
        metavar="N",
        type=int,
        dest="nem",
        default=20,
        help="Number of EM runs",
    )
    parser.add_argument(
        "-n",
        "--ncpu",
        metavar="N",
        help="number of CPUs",
        default=os.cpu_count(),
        type=int,
        dest="ncpu",
    )
    if ray_available:
        parser.add_argument(
            "--ray",
            default=False,
            action="store_true",
            help="Use ray (instead of multiprocessing)",
        )
        parser.add_argument(
            "--ray-ip",
            dest="ray_ip",
            metavar="IP:PORT",
            help="IP address of Ray head node",
            default=None,
        )
        parser.add_argument(
            "--redis-password",
            dest="redis_password",
            metavar="PASSWORD",
            help="redis password to connect ot ray cluster",
            default=None,
        )
    else:
        parser.add_argument("--ray", default=False, help=argparse.SUPPRESS)
    return parser


def main():
    parser = get_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    logfile = args.prefix + ".fphtrain.log"
    model_file = args.prefix + ".fphmodel"
    if args.ray:
        fph = fphray.fastphase
        if args.ray_ip:
            try:
                assert args.redis_password
            except AssertionError:
                print(
                    "Password is required to connect to ray cluster."
                    "Use --redis-password option."
                )
                raise
                ray.init(address=args.ray_ip, redis_password=args.redis_password)
            try:
                assert ray.is_initialized() is True
            except AssertionError:
                print("Failed to connect to Ray cluster")
                raise
            ray_resources = ray.available_resources()
            args.ncpu = int(ray_resources["CPU"])
    else:
        fph = fastphase.fastphase
    # get data in
    if args.samples:
        smp = args.samples.split(",")
    else:
        smp = None
    if args.smpfile:
        if not smp:
            smp = open(args.smpfile).read().split()
        else:
            smp += open(args.smpfile).read().split()
    if smp:
        smp = list(set(smp))
    if args.region:
        regions = [args.region]
    else:
        regions = vcf.vcf_chunk_regions(args.vcf, chunksize=args.chunklen)

    print("Will work on ", len(regions), "regions:", regions[:10])
    # fit fphmodels
    fphmodels = {}
    for r in regions:
        print("Training model on region", r)
        fphdat = vcf.vcf2fph(args.vcf, mode=args.vmode, reg=r, samples=smp)
        if len(fphdat["regions"]) == 0:
            print("No SNP in region", r, ". Skipping it.")
            continue
        fphmodels[r] = {}
        fphmodels[r]["variants"] = fphdat["variants"][r]
        fphmodels[r]["parameters"] = []
        nloc = len(fphdat["variants"][r])
        with fph(nloc, nproc=args.ncpu, prfx=logfile) as myfph:
            if args.vmode == "inbred" or args.vmode == "phased":
                for ID, hap in fphdat["data"][r].items():
                    myfph.addHaplotype(ID, hap)
                if args.ray:
                    pars = myfph.optimfit(args.K, verbose=True)
                else:
                    pars = myfph.optimfit(args.K, verbose=True)
                fphmodels[r]["parameters"].append(pars.__dict__)
            elif args.vmode == "genotype":
                for ID, gen in fphdat["data"][r].items():
                    myfph.addGenotype(ID, gen)
                for iem in range(args.nem):
                    print("Fitting EM", iem)
                    pars = myfph.fit(args.K)
                    fphmodels[r]["parameters"].append(pars.__dict__)
            elif args.vmode == "likelihood":
                for ID, genlik in fphdat["data"][r].items():
                    myfph.addGenotypeLikelihood(ID, genlik)
                for iem in range(args.nem):
                    print("Fitting EM", iem)
                    pars = myfph.fit(args.K)
                    fphmodels[r]["parameters"].append(pars.__dict__)
            else:
                raise ValueError("Unknown mode" + args.vmode)
    with open(model_file, "wb") as fout:
        pickle.dump(fphmodels, fout)


if __name__ == "__main__":
    main()
