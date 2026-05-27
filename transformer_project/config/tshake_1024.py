"""
Transformer on TinyShakespeare, context length 1024, seed 42.
Proposal v4.0 §6 row 2.
"""
out_dir = "out/tshake-1024-s42-d05"
run_name = "tshake-1024-s42-d05"
eval_interval = 250
log_interval = 100
eval_iters = 200
dataset = "tinyshakespeare"
batch_size = 64
block_size = 1024
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.05
bias = False
learning_rate = 3e-4
min_lr = 3e-5
max_iters = 5000
warmup_iters = 200
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
seed = 42
task = "tinyshakespeare"
always_save_checkpoint = True
