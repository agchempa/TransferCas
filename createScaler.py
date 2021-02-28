import argparse
import sys
import re
import pickle
import argparse
import ujson as json
import time
import random
random.seed(1)

import pandas as pd
from pprint import pprint
from subprocess import run
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from features import features

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC

GUIDELENGTH = 23
GCMIN = 0.1
GCMAX = 0.9
HOMOPOLYMERLENGTH_T = 4
HOMOPOLYMERLENGTH_NONT = 5
DIRECTREPEAT = "aacccctaccaactggtcggggtttgaaac"

nucleotides = set("actg")
dinucleotides = set([f"{nuc1}{nuc2}" for nuc1 in nucleotides for nuc2 in nucleotides])


def to_json(out_fname, data):
    with open(out_fname, "w") as wopen:
        json.dump(data, wopen, indent=4)

def get_mult_frequency(sequence, patterns):
    numerator = 0
    denominator = 0
    for idx in range(len(sequence) - len(patterns[0]) + 1):
        denominator += 1
        if sequence[idx: idx + len(patterns[0])] in patterns:
            numerator += 1
    return numerator / denominator


def get_frequency(sequence, pattern):
    # print(sequence, pattern)
    numerator = 0
    denominator = 0
    for idx in range(len(sequence) - len(pattern) + 1):
        denominator += 1
        if sequence[idx: idx + len(pattern)] == pattern:
            numerator += 1
    return numerator / denominator

def rc(guide):
    return str(Seq(guide).reverse_complement())

def include_guide(guide):
    if "TTTT" in guide or "AAAA" in guide: return False
    return True

def single_hybridization_energy(job):
    guide = job["guide"]
    pos = job["pos"]
    width = job["width"]

    query = guide[pos - 1: pos - 1 + width]
    target = rc(query)
    result = run(f"RNAhybrid -s 3utr_human {query} {target}", capture_output=True, shell=True)

    m = re.search(r"mfe:(.*)kcal/mol", result.stdout.decode("utf-8"))
    mfe = m.group(1).strip()
    mfe = float(mfe)

    return (guide, mfe)

def update_hybridization_energies(guides):
    pool = Pool()
    results_3_12 = pool.map(single_hybridization_energy, [{"guide": guide, "pos": 3, "width": 12} for guide in guides])
    for guide, mfe in results_3_12:
        guides[guide]["hybMFE_3.12"] = mfe

    results_15_9 = pool.map(single_hybridization_energy, [{"guide": guide, "pos": 15, "width": 9} for guide in guides])
    for guide, mfe in results_15_9:
        guides[guide]["hybMFE_15.9"] = mfe

    return guides

def update_nucleotide_features(guides):
    for guide, features in guides.items():
        # print(rc(guide).upper())
        guide_dinucs = [guide[idx:idx+2] for idx in range(len(guide) - 1)]

        features["GC"] = GC(guide)
        for nuc in nucleotides:
            frequency = len([char for char in guide if char == nuc]) / len(guide)
            features[f"p{nuc.upper()}"] = frequency
        for dinuc in dinucleotides:
            frequency = len([char for char in guide_dinucs if char == dinuc]) / len(guide_dinucs)
            features[f"p{dinuc.upper()}"] = frequency

        features["NTdens_max_A"] = get_frequency(features["flank"][-14:-7 + 1], "a")
        features["NTdens_max_C"] = get_frequency(features["flank"][-18:-15 + 1], "c")
        features["NTdens_max_G"] = get_frequency(features["flank"][-20 - 1:-18], "g")
        features["NTdens_max_T"] = get_frequency(features["flank"][-12:], "t")

        features["NTdens_max_AT"] = get_mult_frequency(features["flank"][-12 - 1:-2], ["a", "t"])
        # features["NTdens_max_GC"] = get_mult_frequency(features["flank"][-22 - 1:-14], ["g", "c"])

        features["NTdens_min_A"] = get_frequency(features["flank"][-20 - 2:-14 - 1], "a") # verified by hand
        features["NTdens_min_G"] = get_frequency(features["flank"][-13 - 1:-5], "g")
        # features["NTdens_min_T"] = get_frequency(features["flank"][-22 - 1:-13], "t") # verified by hand

        features["NTdens_min_AT"] = get_mult_frequency(features["flank"][-21 - 1:-13], ["a", "t"])

        guides[guide] = features

    return guides

def update_MFE(guides):
    in_fname = "temp_rnafold.fa"
    out_fname = "temp_rnafold.out"
    with open(in_fname, 'w+') as wopen:
        for idx, guide in enumerate(guides):
            # wopen.write(f">seq{idx}\n")
            wopen.write(f"{DIRECTREPEAT}{guide}\n")
    run(f"RNAfold --gquad --noPS --unordered -j < {in_fname} > {out_fname}", shell=True)

    with open(out_fname, 'r') as ropen:
        lines = ropen.readlines()

    for idx in range(len(lines) - 1):
        if idx % 2 != 0: continue

        # RNAfold replaces t with u
        # Strip the DR from the left end to obtain the original guide
        guide = lines[idx].strip().replace("u", "t").replace("U", "T")[len(DIRECTREPEAT):]
        data = lines[idx + 1].strip().split()
        if len(data) == 0: continue

        ss = data[0]
        dr = int(ss.startswith("((((((.(((....))).))))))")) # What is this feature?
        mfe = data[1].strip("(").strip(")")
        mfe = float(mfe)
        gquad = int("+" in ss) # What is this feature?

        if guide not in guides:
            print(f"ERROR: guide {guide} in {out_fname} is not present in the input")
            continue

        features = guides[guide]
        features["MFE"] = mfe
        features["DR"] = dr
        features["Gquad"] = gquad
        guides[guide] = features

    return guides


def main():
    parser = argparse.ArgumentParser(description='Score some guides.')
    parser.add_argument('-i', '--input_fname', default="sample.fa")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    input_fname = args.input_fname
    verbose = args.verbose

    if verbose:
        print("* Fetching guides")

    start = time.time()
    guides = {}
    for record in SeqIO.parse(input_fname, "fasta"):
        seqname = record.id
        seq = str(record.seq)
        for idx in range(3, len(seq) - GUIDELENGTH + 1):
            guide = rc(seq[idx: idx + GUIDELENGTH].lower())
            # if not include_guide(guide): continue
            guides[guide] = {"guide": guide, "pos": idx + 1, "flank": rc(guide)}
    end = time.time()

    if verbose:
        print(f"Time: {end - start} seconds")
        print("* Extracting features")
        print("** Extracting MFE")

    start = time.time()
    guides = update_MFE(guides)
    end = time.time()

    if verbose:
        print(f"Time: {end - start} seconds")
        print("** Extracting hybridization energies")

    start = time.time()
    guides = update_hybridization_energies(guides)
    end = time.time()

    if verbose:
        print(f"Time: {end - start} seconds")
        print("** Extracting nucleotide string features")

    start = time.time()
    guides = update_nucleotide_features(guides)
    end = time.time()

    if verbose:
        print(f"Time: {end - start} seconds")

    dat = pd.DataFrame.from_dict([dat for guide, dat in guides.items()])
    scaler = StandardScaler()
    scaler.fit_transform(dat[features])
    pickle.dump(scaler, open("scaler.pkl","wb"))


if __name__ == "__main__":
    main()

