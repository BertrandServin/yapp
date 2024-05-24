#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import logging
from multiprocessing import cpu_count
from yapp import mendel, family_phaser, recombination, origins, sperm
from yapp import __version__ as version

log_config = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def yapp_parser():
    parser = argparse.ArgumentParser(prog="yapp")
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s version {version}"
    )
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning"], default="info"
    )
    parser.add_argument(
        "-c", type=int, help="Number of CPU to use", default=cpu_count()
    )
    parser.add_argument("--quiet", type=bool, help="Be quiet", default=False)

    subparsers = parser.add_subparsers(
        dest="command", title="Analyses", description="Available commands"
    )
    # Mendel
    parser_mendel = subparsers.add_parser(
        "mendel", help="Correct genealogies from Mendelian errors"
    )
    parser_mendel.add_argument("prfx", type=str, help="prefix for input / output files")
    parser_mendel.set_defaults(func=mendel.main)

    # Phase
    parser_phase = subparsers.add_parser(
        "phase", help="Phase genotypes based on segregation in a pedigree"
    )
    parser_phase.add_argument("prfx", type=str, help="prefix for input / output files")
    parser_phase.add_argument(
        "--reg",
        type=str,
        help="Region to work on in BED format ( chr[:start-end] )",
        default=None,
    )
    parser_phase.add_argument(
        "--focalID",
        type=str,
        help="Run on subpedigree containing individual TOTO",
        default=None,
        metavar="TOTO",
    )
    parser_phase.add_argument(
        "--err",
        type=float,
        help="Assume a genotyping error rate of E",
        metavar="E",
        default=1e-2,
    )
    parser_phase.add_argument(
        "--rho",
        type=float,
        help="Set average recombination rate to R cM/Mb",
        metavar="R",
        default=1,
    )
    parser_phase.set_defaults(func=family_phaser.main)

    # Recomb
    parser_recomb = subparsers.add_parser(
        "recomb", help="Infer recombinations from phased data"
    )
    parser_recomb.add_argument("prfx", type=str, help="prefix for input / output files")
    parser_recomb.add_argument(
        "--rho",
        type=float,
        help="Use average recombination rate of R cM/Mb",
        metavar="R",
        default=1,
    )
    parser_recomb.add_argument(
        "--minsp",
        type=float,
        help="Set minimum segregation probability to call crossover boundary",
        default=0.99,
    )
    parser_recomb.set_defaults(func=recombination.main)

    # Origins
    parser_origins = subparsers.add_parser(
        "origins", help="Trace down ancestral origins from phased data in pedigree"
    )
    parser_origins.add_argument(
        "prfx", type=str, help="prefix for input / output files"
    )
    parser_origins.add_argument(
        "-L", type=int, help="consider IBD if IBS matches >= m markers",
        metavar="m",
        default = 10
    )
    parser_origins.set_defaults(func=origins.main)

    # Sperm
    parser_sperm = subparsers.add_parser(
        "sperm", help='Infer parental chromosomes from "sperm typing" data'
    )
    parser_sperm.add_argument("prfx", type=str, help="prefix for input / output files")
    parser_sperm.add_argument(
        "--pgeno",
        type=float,
        help="Call parent genotype if P(G)>P",
        metavar="P",
        default=0.95,
    )
    parser_sperm.add_argument(
        "--err",
        type=float,
        help="Assume a genotyping error rate of E",
        metavar="E",
        default=1e-2,
    )
    parser_sperm.add_argument(
        "--rho",
        type=float,
        help="Set recombination rate to R cM/Mb",
        metavar="R",
        default=1,
    )
    parser_sperm.add_argument(
        "--minsp",
        type=float,
        help="Set minimum segregation probability to call genotypes",
        default=0.99,
    )
    parser_sperm.set_defaults(func=sperm.main)

    return parser


def main():
    args = sys.argv
    parser = yapp_parser()
    if len(args) < 2:
        parser.print_help()
        sys.exit(1)
    myopts = parser.parse_args(args[1:])
    # Set up logging
    logger = logging.getLogger("yapp")
    logger.setLevel(log_config[myopts.log_level])
    # Console logs
    # Log to file
    fh = logging.FileHandler(myopts.prfx + "_yapp.log")
    fh.setLevel(log_config[myopts.log_level])
    # Formatter
    formatter = logging.Formatter(
        f"%(asctime)s [{myopts.command}] %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if not myopts.quiet:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info(f"Starting YAPP {myopts.command} analysis")
    myopts.func(myopts)
    logger.info(f"Finished YAPP {myopts.command} analysis")
