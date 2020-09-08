#!/bin/bash

CFG=../seir/config

cities=(
    new_york_city
    italy
    delhi
)

date=$(date '+%Y%m%d_%H%M%S')

for i in ${cities[@]}; do
    config="--region_config $CFG/$i.yaml --output_folder $date"
    for j in `seq 0 1 150`; do
    cat <<EOF
python experiments_synth.py $config --num $j
EOF
    done
done | parallel --will-cite --line-buffer
