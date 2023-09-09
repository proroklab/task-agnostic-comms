#!/bin/bash

for i in {0..4}
do
    python policy.py --task_agnostic --scenario $1 --pisa_dim $2 --pisa_path $3 --seed $i --training_iterations 100 --excalibur
done
