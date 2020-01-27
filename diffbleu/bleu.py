from __future__ import division
import tensorflow as tf
import numpy as np


ONEHOT_HARD = "onehot_hard"
ONEHOT_SOFT = "onehot_soft"
TOKENS = "tokens"


def p(x, label):
    return tf.Print(x, [x], message=label, summarize=3123112)

EPS = 1e-12

class BleuScorer(object):

    def __init__(self,
                 vocab_size,
                 seq_length,
                 eos_idx,
                 reference=None,
                 hypothesis=None,
                 input_type=ONEHOT_HARD,
                 ngram_lengths=None,
                 parallel_iterations=1):

        # either all inputs are given value or none of them is
        inputs = [reference, hypothesis]
        assert None not in inputs or all(i is None for i in inputs)
        assert input_type in [ONEHOT_HARD, ONEHOT_SOFT, TOKENS]
        self.input_type = input_type
        self.parallel_iterations = parallel_iterations

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
        self.ref_lengths = ref_lengths

        # count the total ngram count in the reference, based on their length
        hyp_length_mask = self.compute_length_mask('hyp_mask', hypothesis_onehot, self.seq_length, eos_idx)
        hyp_lengths = tf.reduce_sum(hyp_length_mask, axis=1)

        self.hyp_lengths = hyp_lengths

        # count the ngram matches between hypothesis and reference
        self.clipped_matches, self.unclipped_matches = self.build_ngrams(ngram_lengths,
                                                                         reference_onehot,
                                                                         ref_length_mask,
                                                                         hypothesis_onehot,
                                                                         hyp_length_mask)

        sentence_clipped = [tf.reduce_sum(m, axis=1) for m in self.clipped_matches]
        sentence_unclipped = [tf.reduce_sum(m, axis=1) for m in self.unclipped_matches]

        batch_clipped = [tf.reduce_sum(m) for m in self.clipped_matches]
        batch_unclipped = [tf.reduce_sum(m) for m in self.unclipped_matches]

        sentence_p_n = [tf.div(p_clipped, p_unclipped + EPS)
                        for p_clipped, p_unclipped in zip(sentence_clipped, sentence_unclipped)]

        sentence_bp = tf.exp(1. - tf.div(ref_lengths, hyp_lengths + EPS))
        sentence_bp = tf.where(hyp_lengths > 0., sentence_bp, tf.zeros_like(hyp_lengths))
        sentence_bp = tf.where(hyp_lengths > ref_lengths, tf.ones_like(ref_lengths), sentence_bp)

        batch_p_n = [tf.div(p_clipped, p_unclipped + EPS)
                     for p_clipped, p_unclipped in zip(batch_clipped, batch_unclipped)]


        batch_hyp_length = tf.reduce_sum(hyp_lengths)
        batch_ref_length = tf.reduce_sum(ref_lengths)
        batch_bp = tf.exp(1. - (batch_ref_length/(batch_hyp_length + EPS)))
        batch_bp = tf.where(batch_hyp_length > 1. -  EPS, batch_bp, tf.constant(0.))
        batch_bp = tf.where(batch_hyp_length > batch_ref_length, tf.constant(1.), batch_bp)

        weights = [1./len(ngram_lengths) for _ in ngram_lengths]
        sentence_bleu = tf.multiply(sentence_bp, tf.exp(sum([weights[k] * tf.log(sentence_p_n[k] + EPS)
                                                             for k in range(len(ngram_lengths))])))

        corpus_bleu = batch_bp * tf.exp(sum([weights[k] * tf.log(batch_p_n[k] + EPS)
                                             for k in range(len(ngram_lengths))]))
        self.batch_p_n = batch_p_n
        self.batch_bp = batch_bp
        self.batch_bleu_score = corpus_bleu
        self.sentence_p_n = sentence_p_n
        self.sentence_bp = sentence_bp
        self.sentence_bleu_score = sentence_bleu
        self.batch_score = self.batch_bleu_score
        self.sentence_score = self.sentence_bleu_score

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

        clipped_matches, unclipped_matches = zip(*individual_ngrams)
        clipped_matches = list(clipped_matches)
        unclipped_matches = list(unclipped_matches)
        return clipped_matches, unclipped_matches

    def masked_ngram(self, n, onegrams, ref_validity, ref_counts, hyp_validity, hyp_counts):
        ngram = self.build_ngram(n, onegrams) if n != 1 else onegrams
        ref_counts = tf.multiply(ref_counts, ref_validity)
        hyp_counts = tf.multiply(hyp_counts, hyp_validity)
        ref_matrix = self.masked_ngram_ref(n, ngram, ref_counts, hyp_validity)
        hyp_matrix = self.masked_ngram_hyp(n, ngram, ref_validity, hyp_counts)
        clipped = tf.minimum(ref_matrix, hyp_matrix)
        unclipped = hyp_counts
        return clipped, unclipped

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
        log_onegram = tf.expand_dims(tf.log(EPS + onegrams), 3)
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
    bleu_loss = partial_losses.pop('bleu')
    loss = sum(partial_losses.values())
    return placeholders, loss, bleu_loss


def create_mask(s, eos):
    first_eos_index = next((k for k, c in enumerate(s) if c == eos), -1)
    return [1.] * first_eos_index + [0.] * (len(s) - first_eos_index)


def _crop(s, eos):
    first_zero = next((k for k, c in enumerate(s) if c in [eos]), -1)
    return s[:first_zero]


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
    return s, p_n, bp


def main():
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

    eos = 3
    #reference_batch = [[1, 1, 3, 1, eos], [5, 1, eos, 0, 0], [2, 5, 3, eos, 1]]
    #candidate_batch = [[1, 2, 1, eos, 0], [5, 2, eos, 0, 0], [2, 2, 3, eos, 0]]
    candidate_batch = [[56, 11600,1731,233,12301,14,737,12,3102,120,5664,14507,5517,110,120,28983,31171,3] + [1]*72 + [29]*166]
    reference_batch = [[56,11600,5599,52,233,1623,195,12,10882,138,1042,11944,508,2,32,9337,2133,54,23149,74,31171,3,1] + [0]*233]
    row = 0

    seq_length = len(candidate_batch[row])
    reference_length = len(reference_batch[row])
    assert seq_length == reference_length, "hyp.length: {}, ref.length: {}".format(seq_length, reference_length)

    (true_batch_bleu,
     true_batch_p_n,
     true_batch_bp) = custom_corpus_bleu([[_crop(r, eos)] for r in reference_batch],
                                         [_crop(c, eos) for c in candidate_batch])

    true_bleu_scores = [sentence_bleu([_crop(reference_batch[k], eos)],
                                      _crop(candidate_batch[k], eos)) for k in range(len(candidate_batch))]

    bleu_scorer = BleuScorer(seq_length=seq_length,
                             vocab_size=max(np.max(candidate_batch), np.max(reference_batch)) + 1,
                             eos_idx=eos,
                             input_type="tokens")

    #feed_hyp = np_label_smoothing(np_onehot(np.array(candidate_batch)), epsilon=1e-5)
    #feed_refs = np_label_smoothing(np_onehot(np.array(reference_batch)), epsilon=1e-5)

    feed_hyp = np_onehot(np.array(candidate_batch))
    feed_refs = np_onehot(np.array(reference_batch))

    #print("---> {}".format(feed_refs))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {bleu_scorer.hypothesis: candidate_batch,
                     bleu_scorer.reference: reference_batch}

        targets = [bleu_scorer.batch_bleu_score,
                   bleu_scorer.batch_p_n,
                   bleu_scorer.batch_bp,
                   bleu_scorer.sentence_bleu_score,
                   bleu_scorer.sentence_p_n,
                   bleu_scorer.sentence_bp]
        batch_bleu, batch_p_n, batch_bp, sent_bleu, sent_p_n, sent_bp = sess.run(targets, feed_dict=feed_dict)

    print("\n\n")
    print("TRUE: bleu: {}, bp: {}, p_n: {}".format(true_batch_bleu, true_batch_bp, true_batch_p_n))
    print("OURS: bleu: {}, bp: {}, p_n: {}".format(batch_bleu, batch_bp, batch_p_n))

    for k in range(len(sent_bp)):
        print("sent #{}".format(k))
        s, p_n, bp = custom_corpus_bleu([[_crop(reference_batch[k], eos)]],
                                         [_crop(candidate_batch[k], eos)])
        print("TRUE: bleu: {}, bp: {}, p_n: {}".format(s, bp, p_n))
        spn = [sent_p_n[i][k] for i in range(4)]
        print("OURS: bleu: {}, bp: {}, p_n: {}".format(sent_bleu[k], sent_bp[k], spn))


if __name__ == '__main__':
    main()
