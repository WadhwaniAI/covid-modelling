#!/bin/bash

CFG=config

regions=(
    new_york_city
)

date=$(date '+%Y%m%d_%H%M%S')

for i in ${regions[@]}; do
    config="--region_config $CFG/$i.yaml --output_folder $date"
    for j in `seq 0 1 120`; do
    cat <<EOF
python experiments.py $config --num $j
EOF
    done
done | parallel --will-cite --line-buffer
