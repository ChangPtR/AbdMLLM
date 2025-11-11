import torch
import logging
import os

def handle_trainable_modules(model, trainable_modules=None):
    if not trainable_modules:
        return
    if trainable_modules == 'all':
        model.requires_grad_(True)
        print(f"All layers unfrozen.")
        return

    model.requires_grad_(False)
    trainable_layer_list = []
    trainable_layers = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        for tm in trainable_modules:
            condition = 'tm in name' if not isinstance(tm, list) else " and ".join([f"'{t}' in name" for t in tm])
            if eval(condition) and name not in trainable_layer_list:
                param.requires_grad_(True)
                try:
                    trainable_params += torch.prod(torch.tensor(param.size()))
                except:
                    assert tm == 'augment_coefficient'
                trainable_layer_list.append(name)
                trainable_layers += 1
    print("####################################################")
    print(f"{trainable_layers} layers unfrozen, with {human_readable(trainable_params)} trainable parameters.")
    print("####################################################")

def parse_trainable_modules(modules_repr):
    if not isinstance(modules_repr, str):
        return modules_repr
    if modules_repr == 'all':
        return 'all'
    module_list = []
    modules = modules_repr.split('+')
    for m in modules:
        if m == 'tempatt':
            module_list.append('temp_attentions')
        elif m == 'spaatt':
            module_list.append('.attentions')
        elif m == 'scatt':
            module_list.append(['.attentions', 'attn1'])  # should meet both conditions
        elif m == 'spacross':
            module_list.append(['.attentions', 'attn2'])
        elif m == 'allatt':
            module_list.extend(['attn1', 'attn2'])  # should meet either condition
        elif m == 'tempconv':
            module_list.append('temp_conv')
        elif m == 'adpt':
            module_list.extend(['adapter', 'image_cond_layer'])
        else:
            module_list.append(m)

    print('trainable parameters:', module_list)
    return module_list

def human_readable(tensor):
    num = tensor.item()
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])

def init_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)    
    fh = logging.FileHandler(f'{log_dir}/losses.log', mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger