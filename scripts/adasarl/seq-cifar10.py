#!/usr/bin/env python3
"""
Enhanced SARL with GELU activation and Balanced Instance Sampling for Sequential CIFAR-10
This script defines a grid search for hyperparameters in the sarl_enhanced_gelu_balanced model
"""
import os

best_params = {
    200: [
        {
            'idt': f'gelu_balanced_v1_buf200_alpha{a}_beta{b}_op{op}_sm{sm}_lr{lr}_simlr{slr}_bs{bs}_mb{mb}_ep{ep}_warm{w}_gelu{g.replace(" ", "-")}_feat{f}_bw{bw}',
            'alpha': a,
            'beta': b,
            'op_weight': op,
            'sm_weight': sm,
            'sim_lr': slr,
            'lr': lr,
            'minibatch_size': mb,
            'batch_size': bs,
            'n_epochs': ep,
            'lr_steps': '15 15',  # Changed to comma-separated string
            'warmup_epochs': w,
            'apply_gelu': g,
            'num_feats': f,
            'use_lr_scheduler': 1,
            'use_balanced_sampling': 1,
            'balance_weight': bw
        }
        for a in [0.2]
        for b in [1.0]
        for op in [0.5]
        for sm in [0.01]
        for slr in [0.001]
        for lr in [0.03]
        for mb in [32]
        for bs in [32]
        for ep in [20]
        for w in [3]
        for g in ['1 1 1 1']
        for f in [512]
        for bw in [1.0]
    ]
    # 500: [
    #     {
    #         'idt': f'gelu_balanced_v1_buf500_alpha{a}_beta{b}_op{op}_sm{sm}_lr{lr}_simlr{slr}_bs{bs}_mb{mb}_ep{ep}_warm{w}_gelu{g.replace(" ", "-")}_feat{f}_bw{bw}',
    #         'alpha': a,
    #         'beta': b,
    #         'op_weight': op,
    #         'sm_weight': sm,
    #         'sim_lr': slr,
    #         'lr': lr,
    #         'minibatch_size': mb,
    #         'batch_size': bs,
    #         'n_epochs': ep,
    #         'lr_steps': '35 45',  # Changed to comma-separated string
    #         'warmup_epochs': w,
    #         'apply_gelu': g,
    #         'num_feats': f,
    #         'use_lr_scheduler': 1,
    #         'use_balanced_sampling': 1,
    #         'balance_weight': bw
    #     }
    #     for a in [0.5]
    #     for b in [1.0]
    #     for op in [0.5]
    #     for sm in [0.01]
    #     for slr in [0.001]
    #     for lr in [0.03]
    #     for mb in [32]
    #     for bs in [32]
    #     for ep in [20]
    #     for w in [3]
    #     for g in ['1 1 1 1']
    #     for f in [512]
    #     for bw in [1.0]
    # ],
}

lst_seed = [1, 3, 5]
lst_buffer_size = [200]
count = 0
output_dir = "experiments/adasarl"
save_model = 1  # Set to 1 to save the final model
save_interim = 0  # Set to 1 to save intermediate model state and running params
device = 'cuda:2'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run balanced sampling experiments
for seed in lst_seed:
    for buffer_size in lst_buffer_size:
        for params in best_params[buffer_size]:
            exp_id = f"adasarl-cifar10-{buffer_size}-param-{params['idt']}-s-{seed}"
            job_args = f"python main.py  \
                --experiment_id \"{exp_id}\" \
                --model adasarl \
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
                --use_balanced_sampling {params['use_balanced_sampling']} \
                --balance_weight {params['balance_weight']} \
                --output_dir \"{output_dir}\" \
                --csv_log \
                --seed {seed} \
                --device {device} \
                --save_model {save_model} \
                --save_interim {save_interim} \
                "
            count += 1
            print(f"Running experiment {count}: {exp_id}")
            result = os.system(job_args)
            if result != 0:
                print(f"Warning: Experiment {exp_id} failed with return code {result}")
            else:
                print(f"Completed experiment {count}: {exp_id}")

print(f'{count} jobs counted')