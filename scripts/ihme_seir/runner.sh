#!/bin/bash

DATACFG=config
IHMECFG=../ihme/config

cities=(
    delhi
    bengaluru_urban
)

date=$(date '+%Y%m%d_%H%M%S')

for i in ${cities[@]}; do
    config="--region_config $DATACFG/$i.yaml --ihme_config $IHMECFG/$i.yaml --output_folder $date"
    for j in `seq 0 1 15`; do
    cat <<EOF
python synthetic_data_experiments.py $config --num $j
EOF
    done
done | parallel --will-cite --line-buffer
