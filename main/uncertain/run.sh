#!/bin/bash

export PYTHONPATH=`git rev-parse --show-toplevel`:$PYTHONPATH
export PYTHONLOGLEVEL=debug

OUTLOOK=5
URL=https://api.covid19india.org/csv/latest/raw_data.csv

# tmp=`mktemp`
# echo $tmp
tmp=a.csv
wget --output-document=- $URL | \
    python get-c19i.py --state Maharashtra --district pune > $tmp

python estimation.py \
       --outlook $OUTLOOK \
       --config config.ini \
       --data $tmp \
       --steps 10000 \
       --starts `nproc` > \
       b.csv

python process.py \
       --data a.csv \
       --parameters b.csv \
       --burn-in 1000 \
       --outlook $OUTLOOK > \
       c.csv
python viz-seaborn.py --output c.png < c.csv
