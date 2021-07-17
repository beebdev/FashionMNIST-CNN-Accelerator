
from os import setgroups


src_filename = "weights.txt"
sink_filename = "../cnn_accelerator/include/weights.h"

with open(src_filename, "r") as src, open(sink_filename, "w") as sink:
    state = None
    for line in src:
        if "CONV start" in line:
            state = "CONV"
            cols = line.split(" ")
            conv = []
            conv_filter = []
        elif "FC start" in line:
            state = "FC"
            cols = line.split(" ")
            fc = []
        elif "end" in line:
            state = None
        elif "=" in line:
            if state == "CONV":
                conv.append(conv_filter)
        else:
            cols = line.split(" ")
            if state == "CONV":
                conv_filter.append(cols[:-1])
            elif state == "FC":
                fc.append(cols[:-1])
    src.close()
    sink.close()
    print(conv)
    print(fc)
