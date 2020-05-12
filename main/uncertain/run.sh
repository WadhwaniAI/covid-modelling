#!/bin/bash

export PYTHONPATH=`git rev-parse --show-toplevel`:$PYTHONPATH
export PYTHONLOGLEVEL=debug

OUTLOOK=5
FITDAYS="--fit-days 7"
STEPS=50000
RUNS=600
BURN=2500
URL=https://api.covid19india.org/csv/latest/raw_data.csv

_data=`mktemp`
wget --output-document=- $URL | \
    python get-c19i.py --state Maharashtra --district pune > $_data

estimates=`mktemp --directory`
for i in `seq $RUNS`; do
    out=`mktemp --tmpdir=$estimates`
    cat <<EOF
python estimation.py $FITDAYS \
       --outlook $OUTLOOK \
       --config config.ini \
       --data $_data \
       --steps $STEPS > \
       $out
EOF
done | parallel --will-cite --line-buffer

_parameters=`mktemp`
python collate.py --burn-in $BURN --estimates $estimates > $_parameters
if [ `stat --format="%s" $_parameters` -gt 0 ]; then
    python process.py \
	   --data $_data \
	   --parameters $_parameters | \
	python viz-seaborn.py --output estimate.png
fi
