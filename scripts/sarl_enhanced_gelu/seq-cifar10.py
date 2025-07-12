#!/usr/bin/env python3
"""
Enhanced SARL with GELU activation for Sequential CIFAR-10
This script runs experiments with the sarl_enhanced_gelu model
"""

import os

best_params = {
    200: {
        'idt': 'gelu_v1_best',
        'alpha': 0.5,
        'beta': 1.0,
        'op_weight': 0.5,
        'sm_weight': 0.01,
        'sim_lr': 0.001,
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45',
        'warmup_epochs': 3,
        'apply_gelu': '1 1 1 1',  # Apply GELU to all layers
        'num_feats': 512,
        'use_lr_scheduler': 1
    },
    500: {
        'idt': 'gelu_v1_best',
        'alpha': 0.2,
        'beta': 1.0,
        'op_weight': 0.5,
        'sm_weight': 0.01,
        'sim_lr': 0.001,
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45',
        'warmup_epochs': 3,
        'apply_gelu': '1 1 1 1',  # Apply GELU to all layers
        'num_feats': 512,
        'use_lr_scheduler': 1
    },
}

# Grid search configurations
grid_search_params = {
    # Configuration 1: Higher semantic weight
    200: [
        {
            'idt': 'gelu_v2_high_sm',
            'alpha': 0.5,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.05,  # Increased semantic weight
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 2: Lower alpha, higher semantic weight
        {
            'idt': 'gelu_v3_low_alpha',
            'alpha': 0.3,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.02,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 3: Higher prototype weight
        {
            'idt': 'gelu_v4_high_op',
            'alpha': 0.5,
            'beta': 1.0,
            'op_weight': 1.0,  # Increased prototype weight
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
            'use_lr_scheduler': 1
        },
        # Configuration 4: Higher learning rate for similarity
        {
            'idt': 'gelu_v5_high_sim_lr',
            'alpha': 0.5,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.005,  # Increased similarity learning rate
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 5: Balanced weights
        {
            'idt': 'gelu_v6_balanced',
            'alpha': 0.4,
            'beta': 0.8,
            'op_weight': 0.3,
            'sm_weight': 0.03,
            'sim_lr': 0.002,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 6: No GELU (baseline comparison)
        {
            'idt': 'gelu_v7_no_gelu',
            'alpha': 0.5,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '0 0 0 0',  # No GELU
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 7: GELU on first layer only
        {
            'idt': 'gelu_v8_first_layer',
            'alpha': 0.5,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 0 0 0',  # GELU on first layer only
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 8: Warmup epochs = 5
        {
            'idt': 'gelu_v11_warmup_5',
            'alpha': 0.5,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 5,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 9: Higher semantic weight with warmup 5
        {
            'idt': 'gelu_v12_high_sm_warmup_5',
            'alpha': 0.5,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.05,  # Increased semantic weight
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 5,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        }
    ],
    500: [
        {
            'idt': 'gelu_v2_high_sm',
            'alpha': 0.2,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.05,  # Increased semantic weight
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 2: Very low alpha
        {
            'idt': 'gelu_v3_vlow_alpha',
            'alpha': 0.1,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.02,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 3: Higher prototype weight
        {
            'idt': 'gelu_v4_high_op',
            'alpha': 0.2,
            'beta': 1.0,
            'op_weight': 1.0,  # Increased prototype weight
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
            'use_lr_scheduler': 1
        },
        # Configuration 4: Lower distillation weight
        {
            'idt': 'gelu_v7_low_beta',
            'alpha': 0.2,
            'beta': 0.5,  # Decreased distillation weight
            'op_weight': 0.5,
            'sm_weight': 0.02,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 5: Conservative approach
        {
            'idt': 'gelu_v8_conservative',
            'alpha': 0.3,
            'beta': 1.2,
            'op_weight': 0.2,
            'sm_weight': 0.005,
            'sim_lr': 0.0005,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 6: No GELU (baseline comparison)
        {
            'idt': 'gelu_v9_no_gelu',
            'alpha': 0.2,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '0 0 0 0',  # No GELU
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 7: GELU on first layer only
        {
            'idt': 'gelu_v10_first_layer',
            'alpha': 0.2,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3,
            'apply_gelu': '1 0 0 0',  # GELU on first layer only
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 8: Warmup epochs = 5
        {
            'idt': 'gelu_v13_warmup_5',
            'alpha': 0.2,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 5,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        },
        # Configuration 9: Higher semantic weight with warmup 5
        {
            'idt': 'gelu_v14_high_sm_warmup_5',
            'alpha': 0.2,
            'beta': 1.0,
            'op_weight': 0.5,
            'sm_weight': 0.05,  # Increased semantic weight
            'sim_lr': 0.001,
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 5,
            'apply_gelu': '1 1 1 1',
            'num_feats': 512,
            'use_lr_scheduler': 1
        }
    ]
}

lst_seed = [1, 3, 5]
lst_buffer_size = [200, 500]
count = 0
output_dir = "experiments/sarl_enhanced_gelu"
save_model = 0  # set to 1 to save the final model
save_interim = 0  # set to 1 to save intermediate model state and running params
device = 'cuda'

# Run original best parameters
for seed in lst_seed:
    for buffer_size in lst_buffer_size:
        params = best_params[buffer_size]
        exp_id = f"sarl-enhanced-gelu-cifar10-{buffer_size}-param-{params['idt']}-s-{seed}"
        job_args = f"python main.py  \
            --experiment_id {exp_id} \
            --model sarl_enhanced_gelu \
            --dataset seq-cifar10 \
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
            --use_lr_scheduler {params['use_lr_scheduler']} \
            --output_dir {output_dir} \
            --csv_log \
            --seed {seed} \
            --device {device} \
            --save_model {save_model} \
            --save_interim {save_interim} \
            "
        count += 1
        os.system(job_args)

# Run grid search parameters
for seed in lst_seed:
    for buffer_size in lst_buffer_size:
        for params in grid_search_params[buffer_size]:
            exp_id = f"sarl-enhanced-gelu-cifar10-{buffer_size}-param-{params['idt']}-s-{seed}"
            job_args = f"python main.py  \
                --experiment_id {exp_id} \
                --model sarl_enhanced_gelu \
                --dataset seq-cifar10 \
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
                --use_lr_scheduler {params['use_lr_scheduler']} \
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