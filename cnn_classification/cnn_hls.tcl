open_project cnn_hls_proj
set_top cnn
add_files src/cnn.cpp
add_files include/cnn.h
add_files include/weights.h
add_files -tb src/cnn_top.cpp
open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10
exit