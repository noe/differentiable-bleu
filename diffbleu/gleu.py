from __future__ import division
import tensorflow as tf
import numpy as np


ONEHOT_HARD = "onehot_hard"
ONEHOT_SOFT = "onehot_soft"
TOKENS = "tokens"


class GleuScorer(object):

    def __init__(self,
                 vocab_size,
                 seq_length,
                 eos_idx,
                 reference=None,
                 hypothesis=None,
                 input_type=ONEHOT_HARD,
                 ngram_lengths=None,
                 parallel_iterations=1,
                 combinators=None):

        # either all inputs are given value or none of them is
        inputs = [reference, hypothesis]
        assert None not in inputs or all(i is None for i in inputs)
        assert input_type in [ONEHOT_HARD, ONEHOT_SOFT, TOKENS]
        self.input_type = input_type
        self.parallel_iterations = parallel_iterations
        self.combinators = combinators

        if reference is None:  # hypothesis is also None (asserted above)
            if input_type in [ONEHOT_SOFT, ONEHOT_HARD]:
                onehot_shape = (None, seq_length, vocab_size)
                hypothesis = tf.placeholder(tf.float32, shape=onehot_shape, name='hypothesis')
                reference = tf.placeholder(tf.float32, shape=onehot_shape, name='reference')
            elif input_type == TOKENS:
                hypothesis = tf.placeholder(tf.int32, shape=(None, seq_length), name='hypothesis')
                reference = tf.placeholder(tf.int32, shape=(None, seq_length), name='reference')

        if input_type in [ONEHOT_SOFT, ONEHOT_HARD]:
            reference_onehot = reference
            hypothesis_onehot = hypothesis
        elif input_type == TOKENS:
            reference_onehot = tf.one_hot(reference, depth=vocab_size, axis=-1, dtype=tf.float32)
            hypothesis_onehot = tf.one_hot(hypothesis, depth=vocab_size, axis=-1, dtype=tf.float32)

        if ngram_lengths is None:
            ngram_lengths = [1, 2, 3, 4]

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.hypothesis = hypothesis
        self.reference = reference
        self.hypothesis_onehot = hypothesis_onehot
        self.reference_onehot = reference_onehot

        # count the total ngram count in the reference, based on their length
        ref_length_mask = self.compute_length_mask('ref_mask', reference_onehot, self.seq_length, eos_idx)
        ref_lengths = tf.reduce_sum(ref_length_mask, axis=1)
        tpfn = sum([tf.maximum(ref_lengths - float(n - 1), 0.) for n in ngram_lengths])
        self.ref_lengths = ref_lengths

        # count the total ngram count in the reference, based on their length
        hyp_length_mask = self.compute_length_mask('hyp_mask', hypothesis_onehot, self.seq_length, eos_idx)
        hyp_lengths = tf.reduce_sum(hyp_length_mask, axis=1)

        tpfp = sum([tf.maximum(hyp_lengths - float(n - 1), 0.) for n in ngram_lengths])
        self.tpfn = tpfn
        self.tpfp = tpfp
        self.hyp_lengths = hyp_lengths

        # count the ngram matches between hypothesis and reference
        self.ngrams = self.build_ngrams(ngram_lengths,
                                        reference_onehot,
                                        ref_length_mask,
                                        hypothesis_onehot,
                                        hyp_length_mask)

        n_match = tf.reduce_sum(self.ngrams, axis=1)
        self.sentence_n_match = n_match

        dividend = tf.maximum(tpfp, tpfn)
        self.sentence_n_all = dividend

        # move zeros from dividend to n_match
        ones = tf.ones_like(n_match)
        zeros = tf.ones_like(n_match)
        fixed_nmatch = tf.where(dividend > 0., n_match, zeros)
        fixed_dividend = tf.where(dividend > 0., dividend, ones)
        self.sentence_gleu_score = tf.div(fixed_nmatch, fixed_dividend)

        self.batch_n_match = tf.reduce_sum(n_match)
        self.batch_n_all = tf.reduce_sum(dividend)
        self.batch_gleu_score = self.batch_n_match / (1e-7 + self.batch_n_all)
        self.batch_score = self.batch_gleu_score
        self.sentence_score = self.sentence_gleu_score

        # store result and byproduct tensors for easier debugging
        self.results = {'gleu': self.sentence_gleu_score,
                        'tpfp': tpfp,
                        'tpfn': tpfn,
                        'nmatch': n_match}

    def num_iters(self, n, length=None):
        length = self.seq_length if length is None else length
        return max(length - n + 1, 0)

    def compute_length_mask(self, name, sequences, seq_length, eos_idx):
        """"""
        with tf.variable_scope(name):
            is_not_eos = 1. - sequences[:, :, eos_idx]

            token_masks = [is_not_eos[:, 0]]
            for token_idx in range(1, seq_length):
                is_beyond_eos = tf.reduce_prod(is_not_eos[:, 0:token_idx + 1], axis=1)
                token_masks.append(is_beyond_eos)
            result = tf.stack(token_masks, axis=1)
            return result

    def compute_validity_mask(self, n, self_onegram_matches, length_mask):
        ngram_length_mask = length_mask[:, n - 1:]
        self_ngram_matches = self.build_ngram(n, self_onegram_matches)
        dim = self.num_iters(n)
        upper_triangular_ones = tf.constant(np.triu(np.ones((dim, dim), dtype=np.float32), k=1))
        batch_size = tf.shape(self_ngram_matches)[0]
        upper_triangular_ones = tf.tile(tf.expand_dims(upper_triangular_ones, 0), [batch_size, 1, 1])
        masked_matches = tf.multiply(upper_triangular_ones, self_ngram_matches)
        mask = tf.reduce_prod(1. - masked_matches, axis=1)
        validity_mask = tf.multiply(mask, ngram_length_mask)
        counts = tf.reduce_sum(self_ngram_matches, axis=2)
        return validity_mask, counts

    def compute_num_ngrams(self, n, num_onegrams):
        num_ngrams = num_onegrams - float(n - 1)
        return tf.maximum(num_ngrams, 0.)

    def build_ngrams(self, ngram_lengths, reference, reference_mask, hypothesis, hypothesis_mask):

        with tf.variable_scope('onegrams'):
            #  build 1-grams, with shape [n, seq_length, seq_length]
            onegram_matches = self.build_onegram_matches(reference,
                                                         reference_mask,
                                                         hypothesis,
                                                         hypothesis_mask)

        hyp_onegrams = self.build_onegram_matches(hypothesis,
                                                  hypothesis_mask,
                                                  hypothesis,
                                                  hypothesis_mask)

        hyp_validities, hyp_counts = zip(*[self.compute_validity_mask(n, hyp_onegrams, hypothesis_mask)
                                           for n in ngram_lengths])

        ref_onegrams = self.build_onegram_matches(reference,
                                                  reference_mask,
                                                  reference,
                                                  reference_mask)

        ref_validities, ref_counts = zip(*[self.compute_validity_mask(n, ref_onegrams, reference_mask)
                                           for n in ngram_lengths])

        individual_ngrams = [self.masked_ngram(n,
                                               onegram_matches,
                                               ref_validities[k],
                                               ref_counts[k],
                                               hyp_validities[k],
                                               hyp_counts[k])
                             for k, n in enumerate(ngram_lengths)]

        # matrix with all ngram matches concatenated
        self.individual_ngrams = individual_ngrams
        return tf.concat(individual_ngrams, axis=1, name='ngrams')

    def masked_ngram(self, n, onegrams, ref_validity, ref_counts, hyp_validity, hyp_counts):
        ngram = self.build_ngram(n, onegrams) if n != 1 else onegrams
        ngram = self.combinators[n](ngram) if self.combinators else ngram
        ref_counts = tf.multiply(ref_counts, ref_validity)
        hyp_counts = tf.multiply(hyp_counts, hyp_validity)
        ref_matrix = self.masked_ngram_ref(n, ngram, ref_counts, hyp_validity)
        hyp_matrix = self.masked_ngram_hyp(n, ngram, ref_validity, hyp_counts)
        return tf.minimum(ref_matrix, hyp_matrix)

    def masked_ngram_ref(self, n, ngram, ref_counts, hyp_validity):
        dim = self.num_iters(n)
        ref_counts = tf.tile(tf.expand_dims(ref_counts, 1), [1, dim, 1])
        hyp_validity = tf.tile(tf.expand_dims(hyp_validity, 2), [1, 1, dim])
        mask = tf.multiply(hyp_validity, ngram)
        masked = tf.multiply(mask, ref_counts)
        return tf.reshape(masked, [-1, dim * dim])

    def masked_ngram_hyp(self, n, ngram, ref_validity, hyp_counts):
        dim = self.num_iters(n)
        ref_validity = tf.tile(tf.expand_dims(ref_validity, 1), [1, dim, 1])
        hyp_counts = tf.tile(tf.expand_dims(hyp_counts, 2), [1, 1, dim])
        mask = tf.multiply(ref_validity, ngram)
        masked = tf.multiply(mask, hyp_counts)
        return tf.reshape(masked, [-1, dim * dim])

    def build_ngram(self, n, onegrams):
        # onegrams: (batch, seq_length, seq_length)
        log_onegram = tf.expand_dims(tf.log(1e-17 + onegrams), 3)
        kernel = tf.expand_dims(tf.expand_dims(tf.eye(n), 2), 2)
        log_result = tf.nn.conv2d(log_onegram, kernel, strides=[1, 1, 1, 1], padding="VALID")
        result = tf.exp(tf.squeeze(log_result, 3))
        return result

    def build_onegram_matches(self, s1, s1_mask, s2, s2_mask):
        """
        :param s1:   (batch_size, seq_length, vocab_size)
        :param s1_mask: (batch_size, seq_length)
        :param s2:  (batch_size, seq_length, vocab_size)
        :param s2_mask: (batch_size, seq_length)
        :return:
        """
        s2_cond = tf.tile(tf.expand_dims(s2_mask, 2), [1, 1, self.vocab_size])
        masked_s2 = tf.multiply(s2_cond, s2)
        s1_cond = tf.tile(tf.expand_dims(s1_mask, 2), [1, 1, self.vocab_size])
        masked_s1 = tf.multiply(s1_cond, s1)

        #if self.input_type == ONEHOT_SOFT:  # unit-normalize ref and hyp
        #    masked_hypothesis = tf.nn.l2_normalize(masked_hypothesis, dim=1)
        #    masked_reference = tf.nn.l2_normalize(masked_reference, dim=2)

        def body(seq_index, lengths):
            masked_inputs = tf.expand_dims(masked_s2[seq_index, :, :], 0)  # ---- (1, seq_length, vocab_size)
            masked_kernel = masked_s1[seq_index, :, :]  # - (seq_length, vocab_size)
            # filter shape has to be [filter_width, in_channels, out_channels]:
            masked_kernel = tf.expand_dims(tf.transpose(masked_kernel), 0)  # -- (1, vocab_size, seq_length)
            conv = tf.nn.conv1d(masked_inputs,
                                masked_kernel,
                                stride=1,
                                padding='VALID')  # ---------------- (seq_length, seq_length)

            return seq_index + 1, lengths.write(seq_index, conv)

        batch_size = tf.shape(s1)[0]
        with tf.variable_scope("onegrams"):
            convs_ta = tf.TensorArray(tf.float32, batch_size)

            _, loop_result = tf.while_loop(lambda index, *_: index < batch_size,
                                           body,
                                           (tf.constant(0), convs_ta),
                                           parallel_iterations=self.parallel_iterations)
            result = tf.squeeze(loop_result.stack(), axis=1)  # -------- (batch_size, seq_length, seq_length)

            return result


def build_graph(scorer):
    labels = list(scorer.results.keys())
    placeholders = {label: tf.placeholder(tf.float32, [None], '{}_value'.format(label)) for label in labels}
    placeholders['reference'] = scorer.reference
    placeholders['reference_mask'] = scorer.reference_mask
    placeholders['hypothesis'] = scorer.hypothesis

    partial_losses = {label: tf.reduce_mean(tf.square(scorer.results[label] - placeholders[label]),
                                            name='{}_loss'.format(label))
                      for label in labels}
    gleu_loss = partial_losses.pop('gleu')
    loss = sum(partial_losses.values())
    return placeholders, loss, gleu_loss


def create_mask(s, eos):
    first_eos_index = next((k for k, c in enumerate(s) if c == eos), -1)
    return [1.] * first_eos_index + [0.] * (len(s) - first_eos_index)


def _crop(s, eos):
    first_zero = next((k for k, c in enumerate(s) if c in [eos]), -1)
    return s[:first_zero]


def custom_sentence_gleu(references, hypothesis, min_len=1, max_len=4):
    from collections import Counter
    from nltk.util import everygrams

    assert len(references) == 1

    hyp_ngrams = Counter(everygrams(hypothesis, min_len, max_len))
    tpfp = sum(hyp_ngrams.values())  # True positives + False positives.

    reference = references[0]

    ref_ngrams = Counter(everygrams(reference, min_len, max_len))
    tpfn = sum(ref_ngrams.values())  # True positives + False negatives.
    overlap_ngrams = ref_ngrams & hyp_ngrams
    tp = sum(overlap_ngrams.values())  # True positives.
    n_all = max(tpfp, tpfn)

    n_match = tp if n_all > 0 else 0

    # corner case: empty corpus or empty references---don't divide by zero!
    if n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = n_match / n_all

    return gleu_score, n_match, tpfp, tpfn


def np_label_smoothing(inputs, epsilon=0.1):
    K = inputs.shape[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def np_onehot(seq_list):
    onehot = []
    for tokens in seq_list:
        b = np.zeros((tokens.size, tokens.max() + 1))
        b[np.arange(tokens.size), tokens] = 1
        onehot.append(b)
    return np.array(onehot)


def np_softmax(x, temp=0.01):
    x = x / temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def main():
    from nltk.translate.gleu_score import corpus_gleu, sentence_gleu

    eos = 6
    reference_batch = [[1, 1, 2, 1, eos]]#, [5, 1, eos, 0, 0], [2, 5, 3, eos, 1]]
    candidate_batch = [[1, 3, 1, eos, 0]]#, [5, 2, eos, 0, 0], [2, 2, 3, eos, 0]]
    row = 0

    seq_length = len(candidate_batch[row])

    true_batch_gleu = corpus_gleu([[_crop(r, eos)] for r in reference_batch], [_crop(c, eos) for c in candidate_batch])

    gleu_score, n_match, tpfp, tpfn = custom_sentence_gleu([_crop(reference_batch[row], eos)],
                                                            _crop(candidate_batch[row], eos))

    true_gleu_scores = [sentence_gleu([_crop(reference_batch[k], eos)],
                                       _crop(candidate_batch[k], eos)) for k in range(len(candidate_batch))]
    print("true gleu: {}, n_match: {}, tpfp: {}, tpfn: {}".format(gleu_score, n_match, tpfp, tpfn))

    gleu_scorer = GleuScorer(seq_length=seq_length, vocab_size=eos + 1, eos_idx=eos, input_type=ONEHOT_SOFT)

    #feed_hyp = np_label_smoothing(np_onehot(np.array(candidate_batch)), epsilon=1e-5)
    #feed_refs = np_label_smoothing(np_onehot(np.array(reference_batch)), epsilon=1e-5)

    feed_hyp = np_onehot(np.array(candidate_batch))
    feed_refs = np_onehot(np.array(reference_batch))

    #print("---> {}".format(feed_refs))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {gleu_scorer.hypothesis: feed_hyp,
                     gleu_scorer.reference: feed_refs}

        targets = [gleu_scorer.batch_gleu_score,
                   gleu_scorer.sentence_n_match,
                   gleu_scorer.tpfn,
                   gleu_scorer.tpfp,
                   gleu_scorer.sentence_gleu_score,
                   gleu_scorer.individual_ngrams[0]]
        (batch_gleu,
         n_match,
         tpfn,
         tpfp,
         gleu,
         ngram) = sess.run(targets, feed_dict=feed_dict)

    print("our gleu: {}, n_match: {}, tpfp: {}, tpfn: {}".format(gleu[row], n_match[row], tpfp[row], tpfn[row]))

    print("\n\nBatch gleu's. official: {}. ours: {}".format(true_batch_gleu, batch_gleu))

    print("\n\nall gleus....")
    print("true ones: {}".format(true_gleu_scores))
    print("ours: {}".format(gleu))

    print("ngram: {}".format(ngram))


if __name__ == '__main__':
    main()
