import os

best_params = {
    200: {
        'idt': 'enhanced_v1',
        'alpha': 0.5,
        'beta': 1,
        'op_weight': 0.5,
        'sm_weight': 0.01,
        'sim_lr': 0.001,
        'kw': '0.9 0.9 0.9 0.9',
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45',
        'warmup_epochs': 3
    },
    500: {
        'idt': 'enhanced_v1',
        'alpha': 0.2,
        'beta': 1,
        'op_weight': 0.5,
        'sm_weight': 0.01,
        'sim_lr': 0.001,
        'kw': '0.9 0.9 0.9 0.9',
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45',
        'warmup_epochs': 3
    },
}

# Grid search configurations
grid_search_params = {
    # Configuration 1: Higher semantic weight
    200: [
        {
            'idt': 'enhanced_v2_high_sm',
            'alpha': 0.5,
            'beta': 1,
            'op_weight': 0.5,
            'sm_weight': 0.05,  # Increased semantic weight
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 2: Lower alpha, higher semantic weight
        {
            'idt': 'enhanced_v3_low_alpha',
            'alpha': 0.3,
            'beta': 1,
            'op_weight': 0.5,
            'sm_weight': 0.02,
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 3: Higher prototype weight
        {
            'idt': 'enhanced_v4_high_op',
            'alpha': 0.5,
            'beta': 1,
            'op_weight': 1.0,  # Increased prototype weight
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 4: Higher learning rate for similarity
        {
            'idt': 'enhanced_v5_high_sim_lr',
            'alpha': 0.5,
            'beta': 1,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.005,  # Increased similarity learning rate
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 5: Balanced weights
        {
            'idt': 'enhanced_v6_balanced',
            'alpha': 0.4,
            'beta': 0.8,
            'op_weight': 0.3,
            'sm_weight': 0.03,
            'sim_lr': 0.002,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        }
    ],
    500: [
        {
            'idt': 'enhanced_v2_high_sm',
            'alpha': 0.2,
            'beta': 1,
            'op_weight': 0.5,
            'sm_weight': 0.05,  # Increased semantic weight
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 2: Very low alpha
        {
            'idt': 'enhanced_v3_vlow_alpha',
            'alpha': 0.1,
            'beta': 1,
            'op_weight': 0.5,
            'sm_weight': 0.02,
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 3: Higher prototype weight
        {
            'idt': 'enhanced_v4_high_op',
            'alpha': 0.2,
            'beta': 1,
            'op_weight': 1.0,  # Increased prototype weight
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 4: Lower distillation weight
        {
            'idt': 'enhanced_v7_low_beta',
            'alpha': 0.2,
            'beta': 0.5,  # Decreased distillation weight
            'op_weight': 0.5,
            'sm_weight': 0.02,
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        },
        # Configuration 5: Conservative approach
        {
            'idt': 'enhanced_v8_conservative',
            'alpha': 0.3,
            'beta': 1.2,
            'op_weight': 0.2,
            'sm_weight': 0.005,
            'sim_lr': 0.0005,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        }
    ]
}

lst_seed = [1, 3, 5]
lst_buffer_size = [200, 500]
count = 0
output_dir = "experiments/sarl_enhanced"
save_model = 0  # set to 1 to save the final model
save_interim = 0  # set to 1 to save intermediate model state and running params
device = 'cuda'

# Run original best parameters
for seed in lst_seed:
    for buffer_size in lst_buffer_size:
        params = best_params[buffer_size]
        exp_id = f"sarl-enhanced-cifar10-{buffer_size}-param-{params['idt']}-s-{seed}"
        job_args = f"python main.py  \
            --experiment_id {exp_id} \
            --model sarl_enhanced \
            --dataset seq-cifar10 \
            --kw {params['kw']} \
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
            exp_id = f"sarl-enhanced-cifar10-{buffer_size}-param-{params['idt']}-s-{seed}"
            job_args = f"python main.py  \
                --experiment_id {exp_id} \
                --model sarl_enhanced \
                --dataset seq-cifar10 \
                --kw {params['kw']} \
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