
from os import setgroups


src_filename = "weights.txt"
sink_filename = "../cnn_accelerator/include/weights.h"

with open(src_filename, "r") as src, open(sink_filename, "w") as sink:
    state = None
    for line in src:
        if "CONV start" in line:
            state = "CONV"
            cols = line.split(" ")
            nFilter = int(cols[2])
            nZ = int(cols[3])
            nExtendFilter = int(cols[4])
            # TODO setup
        elif "FC start" in line:
            state = "FC"
            cols = line.split(" ")
            nHeight = int(cols[2])
            nWidth = int(cols[3])
            # TODO setup
        elif "end" in line:
            state = None
        elif "=" in line:
            continue
        else: