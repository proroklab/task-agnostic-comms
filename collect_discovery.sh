#!/bin/bash

# Collect runs with 3 different seeds!
# Mandatory runs
for i in {1..3}
do
  python policy.py --wandb_name discovery_nocomms --seed $i -c discovery --model ippo --no_stand --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
  python policy.py --wandb_name discovery_sae_frozen --seed $i -c discovery --model joippo --encoder sae --encoding_dim 32 --encoder_file weights/sae_discovery.pt --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
  python policy.py --wandb_name discovery_noenc --seed $i -c discovery --model joippo --no_stand --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
  python policy.py --wandb_name discovery_sae_policy --seed $i -c discovery --model joippo --encoder sae --encoding_dim 32 --encoder_loss policy --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
done

# Less important runs
for i in {1..3}
do
  python policy.py --wandb_name discovery_sae_tune_recon --seed $i -c discovery --model joippo --encoder sae --encoding_dim 32 --encoder_file weights/sae_discovery.pt --encoder_loss recon --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
  python policy.py --wandb_name discovery_sae_tune_policy --seed $i -c discovery --model joippo --encoder sae --encoding_dim 32 --encoder_file weights/sae_discovery.pt --encoder_loss policy --training_iterations 100 --train_batch_size 2500 --sgd_minibatch_size 1024 --num_envs 4 --no_render_eval_callbacks
done