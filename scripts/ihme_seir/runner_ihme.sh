#!/bin/bash

DATACFG=config
IHMECFG=../ihme/config

regions=(
    pune
)

date=$(date '+%Y%m%d_%H%M%S')

for i in ${regions[@]}; do
    config="--region_config $DATACFG/$i.yaml --ihme_config $IHMECFG/$i.yaml --output_folder $date"
    for j in `seq 0 1 15`; do
    cat <<EOF
python ihme_comparisons.py $config --num $j
EOF
    done
done | parallel --will-cite --line-buffer
