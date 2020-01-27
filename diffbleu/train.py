import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from tensor2tensor.data_generators.translate_ende import EOS

from diffbleu.gleu import GleuScorer, ONEHOT_SOFT, ONEHOT_HARD
from diffbleu.gumbelsoftmax import gumbel_softmax
from diffbleu.utils import custom_corpus_gleu

PAD = np.int64(0)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', required=True)
parser.add_argument('--tmp', required=True)
parser.add_argument('--checkpoints', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--summary', required=True)
parser.add_argument('--noise', type=float, default=0.2)
parser.add_argument('--gs_temp', type=float, default=0.2)
parser.add_argument('--gs_type', default="hard")
parser.add_argument('--iterations', type=int, default=361)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--from_scratch', action="store_true")
args = parser.parse_args()

batch_size=args.batch_size
gs_type=args.gs_type
temperature = args.gs_temp
iterations = args.iterations
data_dir = os.path.expanduser(args.data)
tmp_dir = os.path.expanduser(args.tmp)
checkpoint_dir = os.path.expanduser(args.checkpoints)
output_dir = os.path.expanduser(args.output)
summary_dir = os.path.expanduser(args.summary)

tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(checkpoint_dir)
tf.gfile.MakeDirs(output_dir)

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


def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))


def add_problem_hparams(hparams, problems):
    """Add problem hparams for the problems."""
    # Taken from tpu_trainer_lib.py in tensor2tensor
    hparams.problems = []
    hparams.problem_instances = []
    for problem_name in problems.split("-"):
        problem = registry.problem(problem_name)
        p_hparams = problem.get_hparams(hparams)

        hparams.problem_instances.append(problem)
        hparams.problems.append(p_hparams)


def token_noise(tokens, vocab_size, p=0.1):
    batch_size = tf.shape(tokens)[0]
    seq_length = tf.shape(tokens)[1]
    random_tokens = tf.random_uniform(shape=tf.shape(tokens), minval=0, maxval=vocab_size - 1, dtype=tf.int32)
    bernoulli = tf.distributions.Bernoulli(probs=p, dtype=tf.bool)
    use_real_or_random = bernoulli.sample(sample_shape=tf.stack([batch_size, seq_length]))
    return tf.where(use_real_or_random, random_tokens, tokens)


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None):
    # Taken from tpu_trainer_lib.py in tensor2tensor
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

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


# Create hparams and the model
model_name = "transformer"
hparams_set = "transformer_base"
p_name = "translate_ende_wmt32k"
hparams = create_hparams(hparams_set, data_dir=data_dir, problem_name=p_name)
problem = registry.problem(p_name)
p_hparams = problem.get_hparams(hparams)
max_length = hparams.max_length

# Generate and view the data. WMT data generation can take hours
# ende_problem.generate_data(data_dir, tmp_dir)  # comment after downloaded
training_dataset = ende_problem.dataset(Modes.TRAIN, data_dir)
dataset = training_dataset
dataset = dataset.filter(lambda d: tf.logical_and(tf.shape(d['inputs'])[0] <= max_length,
                                                  tf.shape(d['targets'])[0] <= max_length))
dataset = dataset.shuffle(100 * batch_size)
dataset = dataset.repeat()
dataset = dataset.padded_batch(batch_size,
                               padded_shapes={'inputs': [max_length],
                                              'targets': [max_length]},
                               padding_values={'inputs': PAD,
                                               'targets': PAD})
iterator = dataset.make_one_shot_iterator()

ckpt_name = "transformer_ende_test"
ckpt_path = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, ckpt_name))

x_y = iterator.get_next()
x = tf.cast(x_y['inputs'], tf.int32)
y = tf.cast(x_y['targets'], tf.int32)
assert args.noise >= 0. and args.noise < 1.
noised_y = token_noise(y, vocab_size, p=args.noise) if args.noise > 0. else y

inputs = {"inputs": tf.expand_dims(tf.expand_dims(x, 2), 3),
          "targets": tf.expand_dims(tf.expand_dims(noised_y, 2), 3)}
translate_model = registry.model(model_name)(hparams, Modes.TRAIN, p_hparams)

# version 1.3.0 introduced changes in the API
is_t2t130_or_greater = callable(translate_model)
if is_t2t130_or_greater:
    logits, losses = translate_model(inputs)
else:
    from tensor2tensor.data_generators.problem import SpaceID
    inputs['target_space_id'] = p_hparams.target_space_id
    sharded_logits, losses = translate_model.model_fn(features=inputs)
    logits = tf.concat(sharded_logits, 0)

logits = tf.squeeze(logits, [2, 3])
preds = tf.to_int32(tf.arg_max(logits, dimension=-1))

mle_loss = losses['training']

is_hard = gs_type == "hard"

sampled_reference = tf.one_hot(y, depth=vocab_size, axis=-1)
if not is_hard:
    sampled_reference = label_smoothing(sampled_reference, epsilon=0.01)

gleu_scorer = GleuScorer(seq_length=max_length,
                         vocab_size=vocab_size,
                         eos_idx=EOS,
                         reference=sampled_reference,
                         hypothesis=gumbel_softmax(logits, temperature, hard=is_hard),
                         ngram_lengths=[1, 2, 3, 4],
                         input_type=ONEHOT_HARD if is_hard else ONEHOT_SOFT)

gleu_score = gleu_scorer.batch_gleu_score
gleu_loss = -tf.log(1e-7 + gleu_score)
#gleu_loss = -sum(tf.reduce_sum(tf.log(1e-7 + ngram)) for ngram in gleu_scorer.individual_ngrams)
length_diff = tf.abs(gleu_scorer.ref_lengths - gleu_scorer.hyp_lengths)
length_penalty_loss = tf.nn.relu(tf.reduce_mean(length_diff) - 3.)
anchor_length = 2
anchor_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sampled_reference[:, :anchor_length, :],
                                                                     logits=logits[:, :anchor_length, :]))
#mle_loss_gradients = tf.gradients(mle_loss, tf.trainable_variables())
#total_loss = mle_loss + gleu_loss + length_penalty_loss
#gleu_gradients = tf.gradients(gleu_score, tf.trainable_variables())
total_loss = gleu_loss + length_penalty_loss + anchor_loss

global_step_var = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.98, epsilon=1e-8)
grads_and_vars = optimizer.compute_gradients(total_loss)
grads, _ = list(zip(*grads_and_vars))
gradient_norm = tf.global_norm(grads)
ref_hyp_length_diff = tf.reduce_mean(length_diff)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    gleu_train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step_var)

#tf.summary.scalar('gleu_loss_gradients_norm', tf.global_norm(gleu_gradients))
tf.summary.scalar('gleu', gleu_scorer.batch_gleu_score)
tf.summary.scalar('n_match', gleu_scorer.batch_n_match)
tf.summary.scalar('n_all', gleu_scorer.batch_n_all)

#tf.summary.scalar('mle_loss', mle_loss)
#tf.summary.scalar('mle_loss_gradients_norm', tf.global_norm(mle_loss_gradients))
tf.summary.scalar('gleu_loss', gleu_loss)
tf.summary.scalar('anchor_loss', anchor_loss)
tf.summary.scalar('ref_hyp_length_diff', ref_hyp_length_diff)
tf.summary.scalar('length_penalty_loss', length_penalty_loss)
tf.summary.scalar('total_loss', total_loss)
tf.summary.scalar('total_gradients_norm', gradient_norm)
graph_sums_op = tf.summary.merge_all()

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

model_writer = tf.train.Saver(tf.trainable_variables(), max_to_keep=200)
sum_writer = tf.summary.FileWriter(summary_dir, max_queue=100, flush_secs=240)

WRITE_MODEL_PEDIOD = 5
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.allow_soft_placement = True
print("Creating session...")
with tf.Session(config=config) as sess:
    sess.run(init_op)
    if not args.from_scratch:
        print("Loading initial model from {}...".format(ckpt_path))
        optimistic_restore(sess, ckpt_path)
    try:
        step = 0
        while True:
            step += 1
            print("training step {}...".format(step))
            training_start_time = time.time()
            ops = [gleu_train_op, global_step_var, graph_sums_op, mle_loss, gleu_score, preds, noised_y]
            _, global_step, graph_sums, loss, gleu, pred_values, y_values = sess.run(ops)
            training_step_time = time.time() - training_start_time

            # Compute batch BLEU and GLEU and save summaries of them
            cropped_y = [[_crop(y_values[k, :], EOS)] for k in range(batch_size)]
            cropped_preds = [_crop(pred_values[k, :], EOS) for k in range(batch_size)]
            nltk_bleu = corpus_bleu(cropped_y, cropped_preds, emulate_multibleu=True)
            nltk_gleu, nltk_n_match, nltk_n_all = custom_corpus_gleu(cropped_y, cropped_preds)

            sums = {
                'nltk.bleu': nltk_bleu,
                'nltk.gleu': nltk_gleu,
                'nltk.n_match': nltk_n_match,
                'nltk.n_all': nltk_n_all,
            }

            sum_writer.add_summary(graph_sums, global_step=global_step)
            for label, measure in sums.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag=label, simple_value=measure)])
                sum_writer.add_summary(summary, global_step=global_step)

            print("step took {:.2f} seconds".format(training_step_time))

            if step % WRITE_MODEL_PEDIOD == 0:
                sum_writer.flush()
                model_file = os.path.join(output_dir, 'bleu_improved_{}.ckpt'.format(step))
                model_writer.save(sess, model_file)

            if step >= iterations:
                print("Reached max iterations {}. Exiting.".format(iterations))
                break

    except tf.errors.OutOfRangeError:
        pass

