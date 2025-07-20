# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    
    # Try to find the class name by removing underscores from model name
    try:
        class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    except KeyError:
        # If that fails, try to find a class that contains the model name
        possible_classes = [x for x in mod.__dir__() if not x.startswith('_') and x[0].isupper()]
        class_name = None
        
        # Look for exact match first
        for cls in possible_classes:
            if cls.lower() == model.replace('_', ''):
                class_name = cls
                break
        
        # If no exact match, look for partial match
        if class_name is None:
            for cls in possible_classes:
                if model.replace('_', '').lower() in cls.lower():
                    class_name = cls
                    break
        
        # If still no match, use the first available class
        if class_name is None and possible_classes:
            class_name = possible_classes[0]
        elif class_name is None:
            raise KeyError(f"No suitable class found for model '{model}' in module '{mod.__name__}'")
    
    names[model] = getattr(mod, class_name)


def get_model(args, backbone, loss, transform):
    return names[args.model](backbone, loss, args, transform)
