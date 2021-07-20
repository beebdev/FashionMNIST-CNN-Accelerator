#!/bin/bash

make clean
make getParameters
./getParameters
python3 src/getParameters.py
# rm weights.txt