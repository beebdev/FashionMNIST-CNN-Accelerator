############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_unroll "kernel_conv/kernel_conv_label0"
set_directive_pipeline "conv_layer/conv_layer_label2"
set_directive_array_partition -type cyclic -factor 5 -dim 2 "cnn" img
set_directive_unroll "kernel_conv/kernel_conv_label1"
set_directive_pipeline "kernel_conv/kernel_conv_label1"
set_directive_array_partition -type complete -dim 1 "kernel_conv" buffer
set_directive_unroll "kernel_conv/kernel_conv_label0"
set_directive_unroll "kernel_conv/kernel_conv_label3"
set_directive_unroll "kernel_conv/kernel_conv_label5"
