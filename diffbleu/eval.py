import os
import argparse
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu

from tensor2tensor.layers.modalities import SymbolModality
from tensor2tensor import problems
from tensor2tensor import models
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.data_generators.translate_ende import EOS

PAD = np.int64(0)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--training', type=int, default=0)
parser.add_argument('--decode', action="store_true")
args = parser.parse_args()

data_dir = os.path.expanduser(args.data)
tf.gfile.MakeDirs(data_dir)

if '.ckpt' in args.checkpoint:
    ckpt_path = args.checkpoint
else:
    checkpoint_dir = os.path.expanduser(args.checkpoint)
    tf.gfile.MakeDirs(checkpoint_dir)
    ckpt_name = "transformer_ende_test"
    full_checkpoint_dir = os.path.join(checkpoint_dir, ckpt_name)
    if not os.path.isdir(full_checkpoint_dir):
        full_checkpoint_dir = checkpoint_dir
    ckpt_path = tf.train.latest_checkpoint(full_checkpoint_dir)

Modes = tf.estimator.ModeKeys

# Fetch the problem
ende_problem = problems.problem("translate_ende_wmt32k")

# Copy the vocab file locally so we can encode inputs and decode model outputs
# All vocabs are stored on GCS
vocab_file = os.path.join(data_dir, "vocab.ende.32768")

# Get the encoders from the problem
encoders = ende_problem.feature_encoders(data_dir)
vocab_size = encoders['inputs'].vocab_size


def _crop(s, eos):
    first_zero = next((k for k, c in enumerate(s) if c in [eos]), -1)
    return s[:first_zero]


# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}


def decode(integers, eos_id=1):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if eos_id in integers:
        integers = integers[:integers.index(eos_id)]
    return encoders["inputs"].decode(np.squeeze(integers))


def model_hparams():
    from tensor2tensor.models.transformer import transformer_base
    hparams = transformer_base()
    return hparams


class ProblemHParams(object):
    def __init__(self,
                 source_space_id,
                 source_vocab_size,
                 target_space_id,
                 target_vocab_size):
        input_mod = (registry.Modalities.SYMBOL, source_vocab_size)
        self.input_modality = {"inputs": input_mod}
        self.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        self.loss_multiplier = 1.0
        self.input_space_id = source_space_id
        self.target_space_id = target_space_id

def add_problem_hparams(hparams, problems):
    """Add problem hparams for the problems."""
    # Taken from tpu_trainer_lib.py in tensor2tensor
    problem = registry.problem(problems)
    p_hparams = problem.get_hparams(hparams)
    hparams.problems = [problem]


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None):
    hparams = registry.hparams(hparams_set)()
    if hparams_overrides_str:
        hparams = hparams.parse(hparams_overrides_str)
    if data_dir:
        hparams.add_hparam("data_dir", data_dir)
    if problem_name:
        add_problem_hparams(hparams, problem_name)
    return hparams


def optimistic_restore(session, save_file, variables=None):
    if variables is None:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # taken from https://github.com/tensorflow/tensorflow/issues/312#issuecomment-287455836
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()

    var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

# Create hparams and the model
model_name = "transformer"

p_name = "translate_ende_wmt32k"
hparams = model_hparams()
hparams.data_dir = data_dir
problem = registry.problem(p_name)
p_hparams = problem.get_hparams(hparams)
max_length = hparams.max_length
problem_hparams = ProblemHParams(ende_problem.input_space_id,
                                 encoders['inputs'].vocab_size,
                                 ende_problem.target_space_id,
                                 encoders['targets'].vocab_size)
hparams.problems = [problem_hparams]

# Generate and view the data. WMT data generation can take hours
batch_size = 100
mode = Modes.TRAIN if args.training else Modes.PREDICT
test_dataset = ende_problem.dataset(mode, data_dir)
dataset = test_dataset
dataset = dataset.filter(lambda d: tf.logical_and(tf.shape(d['inputs'])[0] <= max_length,
                                                  tf.shape(d['targets'])[0] <= max_length))
dataset = dataset.padded_batch(batch_size,
                               padded_shapes={'inputs': [max_length],
                                              'targets': [max_length]},
                               padding_values={'inputs': PAD,
                                               'targets': PAD})
iterator = dataset.make_one_shot_iterator()

x_y = iterator.get_next()
x = tf.cast(x_y['inputs'], tf.int32)
y = tf.cast(x_y['targets'], tf.int32)

inputs = {"inputs": tf.expand_dims(tf.expand_dims(x, 2), 3),
          "target_space_id": p_hparams.target_space_id,
          "targets": tf.expand_dims(tf.expand_dims(y, 2), 3)}

# version 1.3.0 introduced changes in the API
from pkg_resources import get_distribution as get_version
t2t_version = int(''.join(get_version('tensor2tensor').version.split('.')[:2]))
is_t2t130_or_greater = t2t_version >= 13
if is_t2t130_or_greater:
    translate_model = registry.model(model_name)(hparams, mode)
    inputs['target_space_id'] = tf.convert_to_tensor(inputs['target_space_id'])
    logits, losses = translate_model(inputs)
    logits = tf.squeeze(logits, [2, 3])
else:
    from tensor2tensor.models.transformer import Transformer
    translate_model = Transformer(hparams, mode, p_hparams)
    translate_model._hparams.problems[0].target_modality = SymbolModality(hparams, encoders['targets'].vocab_size)
    sharded_logits, losses = translate_model.model_fn(features=inputs)
    logits = tf.concat(sharded_logits, 0)
    logits = tf.squeeze(logits, [2, 3])

with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    if args.training:
        preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
    else:
        preds = translate_model.infer(inputs, beam_size=4, alpha=.6)

if isinstance(preds, dict):
    preds = preds['outputs']

bleu_references = []
bleu_preds = []

with tf.Session() as sess:
    print("Restoring model from {}...".format(ckpt_path))
    optimistic_restore(sess, ckpt_path)
    print("done.")
    try:
        def cond(idx):
            return idx < args.training if args.training else True

        k = 0
        i = 0
        while cond(k):
            pred_values, y_values = sess.run([preds, y])
            bleu_preds.extend([_crop(p, EOS) for p in pred_values.tolist()])
            bleu_references.extend([[_crop(r, EOS)] for r in y_values.tolist()])
            if args.decode:
                for t, p in zip(y_values.tolist(), pred_values.tolist()):
                    i += 1
                    print("T[{}] : {}".format(i, decode(t)))
                    print("P[{}] : {}".format(i, decode(p)))
            else:
                print("|", end='', flush=True)
            k += 1
    except tf.errors.OutOfRangeError:
        pass

bleu_score = corpus_bleu(bleu_references, bleu_preds)
gleu_score = corpus_gleu(bleu_references, bleu_preds)
print("BLEU: {}".format(bleu_score))
print("GLEU: {}".format(gleu_score))
