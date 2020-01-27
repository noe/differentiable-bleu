#  This is a toy example that tries to reproduce the experiment
#  from Differentiable lower bound for expected BLEU score
#  (https://arxiv.org/abs/1712.04708).
#
#  A lot of code taken from the article's github repo:
#  https://github.com/deepmipt/expected_bleu/
#
import random
import numpy as np
import tensorflow as tf
import os
from diffbleu.gleu import GleuScorer, ONEHOT_SOFT
from diffbleu.bleu import BleuScorer
import argparse
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from tensorflow.contrib.seq2seq import sequence_loss
from diffbleu.gumbelsoftmax import gumbel_softmax
from tensorflow.python.training.summary_io import SummaryWriterCache
from tqdm import tqdm


def generate_reference(max_len, lengths, vocab_size, eos_id, pad_id):
    def _gen_padded_ref(l):
        ref = np.random.choice(vocab_size, size=l - 1, replace=True)
        padding = np.array([eos_id] + [pad_id] * (max_len - l), dtype=int)
        return np.concatenate((ref, padding)).tolist()
    res = [_gen_padded_ref(l) for l in lengths]
    return np.array(res)


def _all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def onehot(a, vocab_size):
    ncols = vocab_size
    out = np.zeros(a.shape + (ncols,), dtype=np.float32)
    out[_all_idx(a, axis=2)] = np.float32(1.)
    return out


def tf_label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1. - epsilon) * inputs) + (epsilon / K)


def _crop(s, eos):
    first_zero = next((k for k, c in enumerate(s) if c in [eos]), None)
    return s[:first_zero]


def toy(batch_size, max_len, vocab_size, seed, score_type, gs_type, output_dir, iterations, use_reg):
    assert gs_type in ['softmax', 'gs_hard']

    min_len = max(5, max_len // 4)
    eos_id = 2
    pad_id = 0

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    reference_lengths = np.random.randint(low=min_len,
                                          high=max_len - 1,
                                          size=batch_size)

    reference_tokens = generate_reference(max_len,
                                          reference_lengths,
                                          vocab_size,
                                          eos_id,
                                          pad_id)

    ref_tokens_var = tf.Variable(reference_tokens, trainable=False)
    ref_onehot = onehot(reference_tokens, vocab_size)
    ref = tf.Variable(ref_onehot, dtype=tf.float32, trainable=False)
    hyp_shape = (batch_size, max_len, vocab_size)
    hyp_logits = tf.Variable(np.random.rand(*hyp_shape), dtype=tf.float32)
    preds = tf.to_int32(tf.arg_max(hyp_logits, dimension=-1))
    #preds = p(preds, 'preds')
    global_step_var = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)

    weights = tf.sequence_mask(reference_lengths, maxlen=max_len, dtype=tf.float32)
    # w_noise = tf.distributions.Bernoulli(probs=0.01, dtype=tf.float32).sample(tf.shape(weights))
    # weights = tf.multiply(weights, w_noise)
    # mle_loss = sequence_loss(targets=ref_tokens_var,
    #                          logits=hyp_logits,
    #                          weights=weights,
    #                          average_across_batch=True)

    if gs_type == 'softmax':
        scorer_input = tf.nn.softmax(hyp_logits)
    else:
        scorer_input = gumbel_softmax(hyp_logits, 0.5, hard=True)

    scorer_class = BleuScorer if score_type == 'bleu' else GleuScorer

    scorer = scorer_class(seq_length=max_len,
                          vocab_size=vocab_size,
                          eos_idx=eos_id,
                          reference=ref,
                          hypothesis=scorer_input,
                          ngram_lengths=[1, 2, 3, 4],
                          input_type=ONEHOT_SOFT)

    score = scorer.batch_score
    length_diff = tf.abs(scorer.ref_lengths - scorer.hyp_lengths)
    ref_hyp_length_diff = tf.reduce_mean(length_diff)
    target_prob = .95
    mean_max_prob = tf.reduce_mean(tf.reduce_max(tf.clip_by_value(scorer_input, -.1, target_prob), axis=-1))
    #mean_max_prob =  p(mean_max_prob, 'maxmean')
    #reg_on_softmax=  -mean_max_prob
    reg_on_softmax= tf.reduce_mean(-tf.square(scorer_input) + scorer_input)
    #score_loss = -tf.log(1e-7 + score) + length_penalty_loss + mle_loss
    #score_loss = -tf.log(1e-7 + score) + length_penalty_loss
    #score_loss = -tf.log(1e-7 + score) + mle_loss
    scale = tf.clip_by_value(tf.to_float(global_step_var)/1., 0., 50000.)
    #score_loss = -tf.log(1e-7 + score) + scale * reg_on_softmax + length_penalty_loss
    score_loss = -tf.log(1e-7 + score)
    if use_reg:
        score_loss = score_loss + 10000. * reg_on_softmax
    #score_loss = -score
    #score_loss = mle_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=.01, beta1=0.9, beta2=0.98, epsilon=1e-8)

    grads_and_vars = optimizer.compute_gradients(score_loss)
    gradients, variables = list(zip(*grads_and_vars))
    gradient_norm = tf.global_norm(gradients)
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step_var)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    tf.summary.scalar('score', scorer.batch_score)
    tf.summary.scalar('score_loss', score_loss)
    sums_op = tf.summary.merge_all()

    with tf.train.MonitoredTrainingSession(checkpoint_dir=output_dir,
                                           save_summaries_steps=5,
                                           save_checkpoint_secs=1200) as sess:

        sum_writer = SummaryWriterCache.get(output_dir)

        sess.run(init_op)
        targets = [train_op,
                   global_step_var,
                   sums_op,
                   score_loss,
                   score,
                   gradient_norm,
                   preds,
                   ref_hyp_length_diff,
                   mean_max_prob]

        best_score = -np.infty
        nltk_score = 0

        for step in tqdm(range(iterations), ncols=70, leave=False, unit='batch'):
            _, global_step, graph_sums, loss_value, score_value, norm, pred_values, diff, mmp = sess.run(targets)

            # Compute batch BLEU and GLEU and save summaries of them
            cropped_y = [[_crop(reference_tokens[k, :], eos_id)] for k in range(batch_size)]
            cropped_preds = [_crop(pred_values[k, :], eos_id) for k in range(batch_size)]
            nltk_bleu = corpus_bleu(cropped_y, cropped_preds)
            nltk_gleu = corpus_gleu(cropped_y, cropped_preds)
            nltk_score = nltk_bleu if score_type == 'bleu' else nltk_gleu

            if nltk_score > best_score:
                best_score = nltk_score

            if step % 10 == 0:
                msg = "Loss: {:.5e}, score: {:.5e}, nltk.score: {:.5e}, norm: {:.5e}, diff: {:01.2f}, maxprob: {:.2f}"
                #print(msg.format(loss_value, score_value, nltk_score, norm, diff, mmp))

            sums = {
                'nltk.bleu': nltk_bleu,
                'nltk.gleu': nltk_gleu,
            }

            for label, measure in sums.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag=label, simple_value=measure)])
                sum_writer.add_summary(summary, global_step=global_step)

        best_score_file = os.path.join(output_dir, 'best_score.txt')
        with open(best_score_file, 'w') as f:
            print("best score: {}".format(best_score), file=f)
            print("last score: {}".format(nltk_score), file=f)


def main():
    seed = 312

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gs_type', default="gs_hard")
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--score', default='gleu')
    parser.add_argument('--use_reg', action='store_true')

    args = parser.parse_args()
    toy(args.batch_size,
        args.max_len,
        args.vocab_size,
        seed,
        args.score,
        args.gs_type,
        args.output_dir,
        args.iterations,
        args.use_reg)


if __name__ == '__main__':
    main()
