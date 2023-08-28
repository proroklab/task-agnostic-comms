#!/bin/bash

for i in {0..4}
do
    python policy.py --no_comms --scenario $1 --pisa_dim $2 --seed $i
done