# Induction, medium (M=200, ~415 chars), seed 42
out_dir = "out/induction-medium-s42-d05"
run_name = "induction-medium-s42-d05"
task_type = "induction"
length = "medium"
block_size = 512
vocab_size = 27
batch_size = 64
n_layer = 4; n_head = 4; n_embd = 128; dropout = 0.05; bias = False
learning_rate = 3e-4; min_lr = 3e-5; max_iters = 20000; lr_decay_iters = 50000
warmup_iters = 200; weight_decay = 0.1; beta1 = 0.9; beta2 = 0.95
grad_clip = 1.0; decay_lr = True; seed = 42; eval_interval = 1000; eval_iters = 50
