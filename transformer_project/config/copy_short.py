# Long-range copy, short (within gMLP window), seed 42
out_dir = "out/lrcopy-short-s42-d05"
run_name = "lrcopy-short-s42-d05"
task_type = "longrange_copy"
length = "short"
block_size = 144
vocab_size = 55
batch_size = 64
n_layer = 4; n_head = 4; n_embd = 128; dropout = 0.05; bias = False
learning_rate = 3e-4; min_lr = 3e-5; max_iters = 5000; lr_decay_iters = 5000
warmup_iters = 200; weight_decay = 0.1; beta1 = 0.9; beta2 = 0.95
grad_clip = 1.0; decay_lr = True; seed = 42; eval_interval = 250; eval_iters = 50
