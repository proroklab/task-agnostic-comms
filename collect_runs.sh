#!/bin/bash

# Collect runs with 3 different seeds!
for i in {1..3}
do
  python policy.py --wandb_name flocking_proj_sae --seed $i -c flocking_proj --model joippo --use_proj --training_iterations 300 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --encoder sae --encoding_dim 72 --encoder_file weights/sae_flocking_proj.pt
  python policy.py --wandb_name flocking_proj_noencoder --seed $i -c flocking_proj --model joippo --use_proj --training_iterations 300 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4
  python policy.py --wandb_name flocking_proj_sae_rand --seed $i -c flocking_proj --model joippo --use_proj --training_iterations 300 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --encoder sae --encoding_dim 72
  python policy.py --wandb_name flocking_proj_sae_policy --seed $i -c flocking_proj --model joippo --use_proj --training_iterations 300 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --encoder sae --encoding_dim 72 --encoder_loss policy
done