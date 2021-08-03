############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_array_partition -type complete -dim 1 "fc_layer" dout
set_directive_pipeline "fc_layer/fc_loop_1"
set_directive_unroll -factor 5 "fc_layer/fc_activation_loop"
set_directive_array_partition -type complete -dim 1 "fc_layer" inputv

