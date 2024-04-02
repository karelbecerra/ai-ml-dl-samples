import numpy as np

def lr_warmup_cosine_decay(epoch, parameters):
  epochs     = parameters.epochs          # total number of steps (depends on dataset total size, batch and epochs)
  warmup    = parameters.warmup.warmup    # epochs to warmup (lr goes up from 0)
  hold      = parameters.warmup.hold      # hold lr on top
  start_lr  = parameters.warmup.start_lr  # start learning rate = lr * alpha
  target_lr = parameters.warmup.target_lr # final target lr (can be cero but not recommended because it will not learn)
  alpha     = parameters.warmup.alpha     # lowest learning rate = lr * alpha

  if epoch < warmup:
    decayed = start_lr + 0.5 * (1 - np.cos(np.pi * epoch / float(warmup)))  
  elif epoch < warmup + hold:
    decayed = 1# + start_lr
  else: # Cosine decay
    cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup - hold) / float(epochs - warmup - hold)))
    decayed = (1 - alpha) * cosine_decay + alpha

  learning_rate = target_lr * decayed

  return learning_rate, decayed
