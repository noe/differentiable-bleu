import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu

from tensorflow.python.training.summary_io import SummaryWriterCache

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from tensor2tensor.data_generators.translate_ende import EOS

from diffbleu.gleu import GleuScorer, ONEHOT_SOFT, ONEHOT_HARD
from diffbleu.bleu import BleuScorer
from diffbleu.gumbelsoftmax import gumbel_softmax
from diffbleu.utils import custom_corpus_gleu
from diffbleu.seq2seqattn import Seq2SeqAttn
from tqdm import tqdm


PAD = np.int64(0)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', required=True)
parser.add_argument('--tmp', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--summary', required=True)
parser.add_argument('--gs_temp', type=float, default=0.2)
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--gs_type', default="gs_hard")
parser.add_argument('--iterations', type=int, default=361)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--mle', action="store_true")
parser.add_argument('--teacher_forcing', action="store_true")
parser.add_argument('--score', default='gleu')
parser.add_argument('--toy_reverse', action="store_true")
parser.add_argument('--length_penalty', action="store_true")
parser.add_argument('--id', default="")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--combinators', action="store_true")

args = parser.parse_args()

instance_id = args.id or args.summary
batch_size = args.batch_size
gs_type = args.gs_type
temperature = args.gs_temp
iterations = args.iterations
data_dir = os.path.expanduser(args.data)
tmp_dir = os.path.expanduser(args.tmp)
output_dir = os.path.expanduser(args.output)
summary_dir = os.path.expanduser(args.summary)

tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
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
    first_zero = next((k for k, c in enumerate(s) if c in [eos]), None)
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


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


min_length = 3
max_length = 50
num_units = 256
embedding_dim = 256


def check_lengths(d):
    margin = 1
    inputs_less_than_max = tf.shape(d['inputs'])[0] <= (max_length - margin)
    inputs_greater_than_min = tf.shape(d['inputs'])[0] >= min_length
    targets_less_than_max = tf.shape(d['targets'])[0] <= (max_length - margin)
    targets_greater_than_min = tf.shape(d['targets'])[0] >= min_length
    inputs_within_limits = tf.logical_and(inputs_less_than_max, inputs_greater_than_min)
    targets_within_limits = tf.logical_and(targets_less_than_max, targets_greater_than_min)
    return tf.logical_and(inputs_within_limits, targets_within_limits)


def toy_reverse(eos):
    def _reverse(d):
        inputs = d['inputs']
        # taken from https://stackoverflow.com/a/42190780/674487
        #tmp = tf.where(tf.equal(inputs, eos))
        #lengths = tf.segment_min(tmp[:, 1], tmp[:, 0])
        lengths = tf.argmax(tf.cast(tf.equal(inputs, eos), tf.int32), axis=1)
        reversed_inputs = tf.reverse_sequence(inputs, lengths, seq_dim=1)
        return {'inputs': inputs,
                'targets': reversed_inputs}

    return _reverse


def combinator(max_length, n):
    def _combine(inputs):
        proj = tf.layers.Dense(inputs, num_units=max_length - n + 1)
        attention = tf.nn.softmax(proj, dim=2)
        return tf.multiply(inputs, attention)

    return combinator


# Generate and view the data. WMT data generation can take hours
# ende_problem.generate_data(data_dir, tmp_dir)  # comment after downloaded
training_dataset = ende_problem.dataset(Modes.TRAIN, data_dir)
dataset = training_dataset

dataset = dataset.filter(check_lengths)
dataset = dataset.shuffle(100 * batch_size)
dataset = dataset.repeat()
dataset = dataset.padded_batch(batch_size,
                               padded_shapes={'inputs': [max_length],
                                              'targets': [max_length]},
                               padding_values={'inputs': PAD,
                                               'targets': PAD})
if args.toy_reverse:
    dataset = dataset.map(toy_reverse(EOS))
    encoders['targets'] = encoders['inputs']

iterator = dataset.make_one_shot_iterator()

x_y = iterator.get_next()
x = tf.cast(x_y['inputs'], tf.int32)
y = tf.cast(x_y['targets'], tf.int32)

shared_embeddings = args.toy_reverse

with tf.variable_scope("seq2seq"):
    translate_model = Seq2SeqAttn(x,
                                  y,
                                  encoders['inputs'].vocab_size,
                                  max_length,
                                  encoders['targets'].vocab_size,
                                  max_length,
                                  embedding_dim,
                                  num_units,
                                  batch_size,
                                  EOS,
                                  is_train=True,
                                  share_embeddings=shared_embeddings,
                                  teacher_forcing=args.teacher_forcing)

logits = translate_model.logits
mle_loss = translate_model.mle_loss
preds = translate_model.preds if args.mle else tf.to_int32(tf.arg_max(logits, dimension=-1))

assert gs_type in ['gs_hard', 'gs_soft', 'softmax']
is_hard = gs_type == "gs_hard"
hyp = (tf.nn.softmax(logits/args.temp) if gs_type == "softmax"
       else gumbel_softmax(logits, temperature, hard=is_hard))

sampled_reference = tf.one_hot(y, depth=vocab_size, axis=-1)
if not is_hard:
    sampled_reference = label_smoothing(sampled_reference, epsilon=0.01)

ngrams = [1, 2, 3, 4]
combinators = [combinator(max_length, n) for n in ngrams] if args.combinators else None
scorer_class = GleuScorer if args.score == 'gleu' else BleuScorer
scorer = scorer_class(seq_length=max_length,
                      vocab_size=vocab_size,
                      eos_idx=EOS,
                      reference=sampled_reference,
                      hypothesis=hyp,
                      ngram_lengths=ngrams,
                      input_type=ONEHOT_HARD if is_hard else ONEHOT_SOFT,
                      combinators=combinators)

score = scorer.batch_score
score_loss = -tf.log(1e-7 + score)
length_diff = tf.abs(scorer.ref_lengths - scorer.hyp_lengths)
length_penalty_loss = tf.nn.relu(tf.reduce_mean(length_diff) - 3.)
ref_hyp_length_diff = tf.reduce_mean(length_diff)

# mle_loss_gradients = tf.gradients(mle_loss, tf.trainable_variables())
total_loss = mle_loss if args.mle else score_loss
#gleu_gradients = tf.gradients(gleu_score, tf.trainable_variables())
#if args.length_penalty:
#    total_loss = total_loss + length_penalty_loss

global_step_var = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr) #, beta1=0.9, beta2=0.98, epsilon=1e-8)

grads_and_vars = optimizer.compute_gradients(total_loss)
gradients, variables = list(zip(*grads_and_vars))
gradient_norm = tf.global_norm(gradients)
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step_var)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step_var)
    #train_op = optimizer.minimize(total_loss, global_step=global_step_var)

#tf.summary.scalar('gleu_loss_gradients_norm', tf.global_norm(gleu_gradients))
tf.summary.scalar('{}'.format(args.score), scorer.batch_score)
#tf.summary.scalar('n_match', scorer.batch_n_match)
#tf.summary.scalar('n_all', scorer.batch_n_all)

tf.summary.scalar('mle_loss', mle_loss)
#tf.summary.scalar('mle_loss_gradients_norm', tf.global_norm(mle_loss_gradients))
tf.summary.scalar('{}_loss'.format(args.score), score_loss)
tf.summary.scalar('ref_hyp_length_diff', ref_hyp_length_diff)
tf.summary.scalar('length_penalty_loss', length_penalty_loss)
tf.summary.scalar('total_loss', total_loss)
tf.summary.scalar('total_gradients_norm', gradient_norm)
graph_sums_op = tf.summary.merge_all()

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

#model_writer = tf.train.Saver(tf.trainable_variables(), max_to_keep=200)
#sum_writer = tf.summary.FileWriter(summary_dir, max_queue=100, flush_secs=240, graph=tf.get_default_graph())

WRITE_MODEL_PEDIOD = 5
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.allow_soft_placement = True
#config.log_device_placement = True
print("Creating session...")
with tf.train.MonitoredTrainingSession(checkpoint_dir=output_dir, save_summaries_steps=5, save_checkpoint_secs=60) as sess:
    #sess.run(init_op)
    sum_writer = SummaryWriterCache.get(output_dir)
    try:
        for step in tqdm(range(iterations), ncols=70, leave=False, unit='batch'):
            # print("[{}] training step {}...".format(instance_id, step))
            training_start_time = time.time()
            ops = [train_op, global_step_var, graph_sums_op, mle_loss, preds, y]
            _, global_step, graph_sums, loss, pred_values, y_values = sess.run(ops)

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

            for label, measure in sums.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag=label, simple_value=measure)])
                sum_writer.add_summary(summary, global_step=global_step)

    except tf.errors.OutOfRangeError:
        pass

