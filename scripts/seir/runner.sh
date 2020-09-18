#!/bin/bash

CFG=config

#regions=(
#    new_york_city
#    italy
#    delhi
#)
#
#date=$(date '+%Y%m%d_%H%M%S')

#for i in ${regions[@]}; do
#    config="--region_config $CFG/$i.yaml --output_folder $date"
#    for j in `seq 0 1 150`; do
#    cat <<EOF
#python experiments.py $config --num $j
#EOF
#    done
#done | parallel --will-cite --line-buffer

regions=(
  delhi
  italy
  new_york_city
  los_angeles
  pune
)

starts=(
  91
  138
  121
  134
  73
)

ends=(
  142
  181
  170
  175
  114
)

date=$(date '+%Y%m%d_%H%M%S')
count=0
for i in ${regions[@]}; do
    config="--region_config $CFG/$i.yaml --output_folder $date"
    for j in `seq ${starts[count]} 1 ${ends[count]}`; do
    cat <<EOF
python experiments.py $config --num $j
EOF
    done
    count=$((count+1))
done | parallel --will-cite --line-buffer