#!/bin/bash

DATACFG=config

regions=(
    italy
)

date=$(date '+%Y%m%d_%H%M%S')

for i in ${regions[@]}; do
    config="--region_config $DATACFG/$i.yaml --output_folder $date"
    for j in `seq 0 1 0`; do
    cat <<EOF
python compartmental_comparison_experiments.py $config --num $j
EOF
    done
done | parallel --will-cite --line-buffer
