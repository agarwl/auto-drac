import glob
import os
import tensorflow.compat.v1 as tf
gfile = tf.io.gfile

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def cleanup_log_dir(log_dir):
    try:
        gfile.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
