import pandas as pd
from glob import glob
from features import features

for fname in glob("data/*.tsv"):
    df = pd.read_csv(fname, sep="\t", names=["guide"] + features)
    df = df.drop(columns=['DR'])


    df.to_csv(fname, sep="\t", index=False, header=False)
