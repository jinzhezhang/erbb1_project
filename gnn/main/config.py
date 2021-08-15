import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')

task = "regression"  # target is a real value (e.g., energy eV).
dataset = "erbb1_clean_log_ic50"

radius = 1
dim = 50
layer_hidden = 6
layer_output = 6

batch_train = 64
batch_test = 1
lr = 1e-4
lr_decay = 0.99
decay_interval = 10
iteration = 200

lr, lr_decay = map(float, [lr, lr_decay])
