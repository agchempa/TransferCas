import ujson as json
import numpy as np

def encode(guide):
    findstr = "actgu"
    encoding = np.zeros((1, len(guide), len(findstr) + 1))
    
    for idx, nt in enumerate(guide):
        jdx = findstr.find(nt.lower())
        if jdx == -1:
            encoding[0, idx, -1] = 1
        else:
            encoding[0, idx, jdx] = 1
    return encoding


def from_json(fname):
    with open(fname, "r") as wopen:
        return json.load(wopen)

def to_json(out_fname, data):
    with open(out_fname, "w") as wopen:
        json.dump(data, wopen, indent=4)
