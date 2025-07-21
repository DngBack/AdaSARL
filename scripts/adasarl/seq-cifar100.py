#!/usr/bin/env python3

import os

best_params = {
    200: {
        'idt': 'v1',
        'alpha': 0.5,
        'beta': 1,
        'op_weight': 0.5,
        'sm_weight': 0.01,
        'sim_lr': 0.001,
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45',
        'warmup_epochs': 3,
        'apply_gelu': '1 1 1 1',
        'num_feats': 512,
        'use_balanced_sampling': 1,
        'balance_weight': 1.0
    },
    500: {
        'idt': 'v1',
        'alpha': 0.2,
        'beta': 1,
        'op_weight': 0.5,
        'sm_weight': 0.01,
        'sim_lr': 0.001,
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45',
        'warmup_epochs': 3,
        'apply_gelu': '1 1 1 1',
        'num_feats': 512,
        'use_balanced_sampling': 1,
        'balance_weight': 1.0
    },
}

lst_seed = [1, 3, 5]
lst_buffer_size = [200, 500]
count = 0
output_dir = "experiments/adasarl"
save_model = 0  # set to 1 to save the final model
save_interim = 0  # set to 1 to save intermediate model state and running params
device = 'cuda'

for seed in lst_seed:
    for buffer_size in lst_buffer_size:
        params = best_params[buffer_size]
        exp_id = f"adasarl-cifar100-{buffer_size}-param-{params['idt']}-s-{seed}"
        job_args = f"python main.py  \
            --experiment_id {exp_id} \
            --model adasarl \
            --dataset seq-cifar100 \
            --alpha {params['alpha']} \
            --beta {params['beta']} \
            --op_weight {params['op_weight']} \
            --sm_weight {params['sm_weight']} \
            --sim_lr {params['sim_lr']} \
            --buffer_size {buffer_size} \
            --batch_size {params['batch_size']} \
            --minibatch_size {params['minibatch_size']} \
            --lr {params['lr']} \
            --lr_steps {params['lr_steps']} \
            --n_epochs {params['n_epochs']} \
            --warmup_epochs {params['warmup_epochs']} \
            --apply_gelu {params['apply_gelu']} \
            --num_feats {params['num_feats']} \
            --use_balanced_sampling {params['use_balanced_sampling']} \
            --balance_weight {params['balance_weight']} \
            --output_dir {output_dir} \
            --csv_log \
            --seed {seed} \
            --device {device} \
            --save_model {save_model} \
            --save_interim {save_interim} \
            "
        count += 1
        os.system(job_args)

print('%s jobs counted' % count) 