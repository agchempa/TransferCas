import glob
import os
from subprocess import run
from Bio import SeqIO

for fname in glob.glob("samples/*.fa"):
    for record in SeqIO.parse(fname, "fasta"): seqname = record.id

    if os.path.exists(f"data/raw/train_{seqname}.tsv"): continue

    run(f"python createData.py -i {fname} -v", shell=True)
