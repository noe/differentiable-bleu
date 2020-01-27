import os
import argparse
import numpy as np
import tensorflow as tf
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from tensor2tensor import problems
from tensor2tensor import models
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.data_generators.translate_ende import EOS

from diffbleu.gleu import GleuScorer
from diffbleu.bleu import BleuScorer


def custom_corpus_bleu(list_of_references,
                       hypotheses,
                       weights=(0.25, 0.25, 0.25, 0.25)):
    from collections import Counter
    from nltk.translate.bleu_score import modified_precision, closest_ref_length, brevity_penalty, Fraction
    import math

    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), "The number of hypotheses and their reference(s) should be the same"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0, p_n, bp

    s = (w * math.log(p_i + 1e-12) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    s = bp * math.exp(math.fsum(s))
    return s


def custom_sentence_bleu(references, hypothesis):
    return custom_corpus_bleu([references], [hypothesis])


scorer_impl = {'gleu': GleuScorer, 'bleu': BleuScorer}
sentence_score = {'gleu': sentence_gleu, 'bleu': custom_sentence_bleu}
corpus_score = {'gleu': corpus_gleu, 'bleu': custom_corpus_bleu}

PAD = np.int64(0)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', required=True)
parser.add_argument('--checkpoints', required=True)
parser.add_argument('--score', required=True)
args = parser.parse_args()

data_dir = os.path.expanduser(args.data)
checkpoint_dir = os.path.expanduser(args.checkpoints)

tf.gfile.MakeDirs(data_dir)

tf.gfile.MakeDirs(checkpoint_dir)

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


def decode(integers, eos):
    """List of ints to str"""
    integers = list(np.squeeze(integers)) if len(integers) > 0 else []
    if eos in integers:
        integers = integers[:integers.index(eos)]
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

# Create hparams and the model
model_name = "transformer"
hparams_set = "transformer_base"

p_name = "translate_ende_wmt32k"
hparams = create_hparams(hparams_set, data_dir=data_dir, problem_name=p_name)
problem = registry.problem(p_name)
p_hparams = problem.get_hparams(hparams)
max_length = hparams.max_length

# Generate and view the data. WMT data generation can take hours
batch_size = 14
test_dataset = ende_problem.dataset(Modes.EVAL, data_dir)
dataset = test_dataset
dataset = dataset.filter(lambda d: tf.logical_and(tf.shape(d['inputs'])[0] <= max_length,
                                                  tf.shape(d['targets'])[0] <= max_length))
dataset = dataset.padded_batch(batch_size,
                               padded_shapes={'inputs': [max_length],
                                              'targets': [max_length]},
                               padding_values={'inputs': PAD,
                                               'targets': PAD})
iterator = dataset.make_one_shot_iterator()

ckpt_name = "transformer_ende_test"
full_checkpoint_dir = os.path.join(checkpoint_dir, ckpt_name)
if not os.path.isdir(full_checkpoint_dir):
    full_checkpoint_dir = checkpoint_dir

ckpt_path = tf.train.latest_checkpoint(full_checkpoint_dir)

translate_model = registry.model(model_name)(hparams, Modes.EVAL)

x_y = iterator.get_next()
x = tf.cast(x_y['inputs'], tf.int32)
y = tf.cast(x_y['targets'], tf.int32)

inputs = {"inputs": tf.expand_dims(tf.expand_dims(x, 2), 3),
          "target_space_id": p_hparams.target_space_id}

preds = translate_model.infer(features=inputs, decode_length=0)

scorer = scorer_impl[args.score](seq_length=max_length,
                                 vocab_size=vocab_size,
                                 eos_idx=EOS,
                                 reference=y,
                                 hypothesis=preds,
                                 ngram_lengths=[1, 2, 3, 4],
                                 input_type="tokens")


bleu_references = []
bleu_preds = []

with tf.Session() as sess:
    print("Restoring model from {}...".format(ckpt_path))
    optimistic_restore(sess, ckpt_path)
    print("done.")
    try:
        while True:
            print(".", end="", flush=True)
            fetches = [preds, y, scorer.sentence_score, scorer.batch_score]
            pred_values, y_values, our_sentence_score, our_batch_score = sess.run(fetches)
            cropped_y = [_crop(p, EOS) for p in y_values.tolist()]
            decoded_y = [decode(s, EOS) for s in cropped_y]
            cropped_preds = [_crop(r, EOS) for r in pred_values.tolist()]
            decoded_preds = [decode(s, EOS) for s in cropped_preds]

            nltk_batch_score = corpus_score[args.score]([[r] for r in cropped_y], cropped_preds)

            THRESHOLD = 0.00001

            if np.abs(our_batch_score - nltk_batch_score) > THRESHOLD:
                msg = "Batch level difference: (nltk: {:.4f}, ours: {:.4f})"
                print(msg.format(nltk_batch_score, our_batch_score))
                input("")

            for k in range(batch_size):
                nltk_sentence_gleu = sentence_score[args.score]([cropped_y[k]], cropped_preds[k])
                if np.abs(our_sentence_score[k] - nltk_sentence_gleu) > THRESHOLD:
                    msg = "Sentence level difference:  (nltk: {:.4f}, ours: {:.4f})"
                    print(msg.format(nltk_sentence_gleu, our_sentence_score[k]))
                    print("  - ref: {}".format(y_values[k, :]))
                    print("  - hyp: {}".format(pred_values[k, :]))
                    input("")

    except tf.errors.OutOfRangeError:
        pass

