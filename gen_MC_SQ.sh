#!/bin/bash

declare -a arr=("5" "15" "45" "135")

for i in "${arr[@]}"
do
    echo "$i"
    python gen_MC_SQ_kappa.py -i $i 0 5000 64 1 S_q_list_param
done