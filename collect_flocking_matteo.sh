#!/bin/bash

# Collect runs with 3 different seeds!
# Mandatory runs
for i in {1..3}
do
  python policy.py --wandb_name flocking_matteo_nocomms --seed $i -c flocking_matteo --model ippo --no_stand --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
  python policy.py --wandb_name flocking_matteo_sae_frozen --seed $i -c flocking_matteo --model joippo --encoder sae --encoding_dim 32 --encoder_file weights/sae_flocking.pt --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
#  python policy.py --wandb_name flocking_noenc --seed $i -c flocking --model joippo --no_stand --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
#  python policy.py --wandb_name flocking_sae_policy --seed $i -c flocking --model joippo --encoder sae --encoding_dim 32 --encoder_loss policy --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
done

## Less important runs
#for i in {1..3}
#do
#  python policy.py --wandb_name flocking_sae_tune_recon --seed $i -c flocking --model joippo --encoder sae --encoding_dim 32 --encoder_file weights/sae_flocking.pt --encoder_loss recon --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
#  python policy.py --wandb_name flocking_sae_tune_policy --seed $i -c flocking --model joippo --encoder sae --encoding_dim 32 --encoder_file weights/sae_flocking.pt --encoder_loss policy --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
#done