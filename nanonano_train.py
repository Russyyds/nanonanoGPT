import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import json

from nanonano_model import GPTConfig, GPT

# load all hyper param
args = {}
with open("hyper_param.json", 'r') as f:
    args = json.load(f)
print(args)

gradient_accumulation_steps = args['gradient_accumulation_steps'] # used to simulate larger batch sizes
batch_size = args['batch_size'] # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = args['block_size']

# adamw optimizer
learning_rate = args['learning_rate'] # max learning rate
max_iters = args['max_iters'] # total number of training iterations
weight_decay = args['weight_decay']
beta1 = args['beta1']
beta2 = args['beta2']
# learning rate decay settings
decay_lr = args['decay_lr'] # whether to decay the learning rate
warmup_iters = args['warmup_iters'] # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = args['min_lr'] # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(args["out_dir"], exist_ok=True)
torch.manual_seed(1234)

device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', args["dataset"])
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

best_val_loss = 1e9

# attempt to derive vocab_size from the args["dataset"]
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=args['n_layer'], n_head=args['n_head'], n_embd=args['n_embd'], block_size=block_size,
                  bias=args['bias'], vocab_size=None, dropout=args['dropout'], attn_type=args['attn_type']) # start with model_args from command line

# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

checkpoint = None # free up memory

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()

local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
iter_num = 0
# compile the model
if args['compile']:
    print("start compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    t_compile = time.time() - t0
    print(f"Compile model time:{t_compile*1000}")
t0 = time.time()
all_iter_dt = torch.zeros([max_iters])
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = (t1 - t0) * 1000
    t0 = t1
    all_iter_dt[iter_num] = dt
    # get loss as float. note: this is a CPU-GPU sync point
    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
    lossf = loss.item() * gradient_accumulation_steps
    if local_iter_num >= 5: # let the training loop settle a bit
        mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num >= max_iters:
        break

print("first_iter_time:", all_iter_dt[0])
print(f"Average iter time:{all_iter_dt.mean()}ms, average iter time except 1st iter:{all_iter_dt[1:].mean()}")