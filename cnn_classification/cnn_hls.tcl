open_project fft_stages.proj -reset
add_files fft_stages.cpp
add_files fft_stages.h
set_top fft_streaming
open_solution "solution1" -reset
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10
csynth_design
exit
