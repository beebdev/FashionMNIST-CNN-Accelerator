
src_filename = "weights.txt"
# sink_filename = "../cnn_classification/include/weights.h"
sink_filename = "weights.h"

# Values
filtersize = 0
extfilters = 0

# Functions

# Write #include to weights.h


def write_include(sink, library):
    str = "#include <" + library + ">\n"
    sink.write(str)

# Write constant definition to weights.h


def write_define(sink, label, value):
    str = "#define " + label + " " + value + "\n"
    sink.write(str)

# Write newline to weights.h


def write_newline(sink, nlines):
    for i in range(0, nlines):
        sink.write("\n")


def write_comment(sink, msg):
    sink.write("// " + msg + "\n")


def generate_conv(sink):
    write_comment(sink, "CONV constants")
    write_define(sink, "CONV_STRIDE", "1")
    write_define(sink, "CONV_FILTERSIZE", filtersize)
    write_define(sink, "CONV_EXTFILTSIZE", extfilters)
    write_define(sink, "CONV_IN_DIM", "28")
    write_define(sink, "CONV_OUT_XY",
                 "(CONV_IN_DIM - CONV_EXTFILTSIZE) / CONV_STRIDE + 1")
    write_define(sink, "CONV_OUT_Z", "(CONV_FILTERSIZE)")
    write_newline(sink, 2)

    write_comment(sink, "CONV weights")
    sink.write(
        "float conv_filter[CONV_FILTERSIZE][CONV_EXTFILTSIZE][CONV_EXTFILTSIZE] = {\n")
    for f in conv:
        # Each filter
        sink.write("\t{\n")
        for i in f:
            sink.write("\t\t{")
            for j in i:
                sink.write(j+", ")
            sink.write("},\n")
        sink.write("\t},\n")
    sink.write("};\n")


def generate_fc(sink):
    write_comment(sink, "FC constants")
    write_define(sink, "FC_IN_DIM", "1152")
    write_define(sink, "FC_OUT", "10")
    write_define(sink, "FC_IN_X", "12")
    write_define(sink, "FC_IN_Y", "12")
    write_define(sink, "FC_IN_Z", "8")
    write_comment(sink, "FC weights")
    sink.write(
        "float fc_inputs[FC_OUT];\n")
    sink.write(
        "float fc_weights[FC_IN_DIM][FC_OUT] = {\n")
    for f in fc:
        # Each filter
        sink.write("\t{")
        for j in f:
            sink.write(j+", ")
        sink.write("},\n")
    sink.write("};")

def generate_pool(sink):
    write_comment(sink,"POOL constants")
    write_define(sink, "POOL_STRIDE", "2")
    write_define(sink, "POOL_EXTFILTSIZE", "2")
    write_define(sink, "POOL_IN_DIM", "24")
    write_define(sink, "POOL_IN_CHANNEL", "(CONV_OUT_Z)")
    write_define(sink, "POOL_OUT_XY", "(POOL_IN_DIM - POOL_EXTFILTSIZE) / POOL_STRIDE + 1")
    write_define(sink,"POOL_OUT_Z","(POOL_IN_CHANNEL)")
    
    # Extract weights
with open(src_filename, "r") as src:
    state = None
    for line in src:
        if "CONV start" in line:
            state = "CONV"
            cols = line.split(" ")
            filtersize = cols[2]
            extfilters = cols[3]
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
                conv_filter = []
        else:
            cols = line.split(" ")
            if state == "CONV":
                conv_filter.append(cols[:-1])
            elif state == "FC":
                fc.append(cols[:-1])
    src.close()
    # print(conv)
    # print(fc)

# Generate weights.h
with open(sink_filename, "w") as sink:
    generate_conv(sink)
    generate_pool(sink)
    generate_fc(sink)
    sink.close()
