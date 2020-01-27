from __future__ import division
from collections import Counter

from nltk.util import ngrams, everygrams


def custom_corpus_gleu(list_of_references, hypotheses, min_len=1, max_len=4):
    """
    Copy of the GLEU implementation in NLTK that also returns n_match and n_all
    :param list_of_references:
    :param hypotheses:
    :param min_len:
    :param max_len:
    :return:
    """
    # sanity check
    assert len(list_of_references) == len \
        (hypotheses), "The number of hypotheses and their reference(s) should be the same"

    # sum matches and max-token-lengths over all sentences
    corpus_n_match = 0
    corpus_n_all = 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_ngrams = Counter(everygrams(hypothesis, min_len, max_len))
        tpfp = sum(hyp_ngrams.values())  # True positives + False positives.

        hyp_counts = []
        for reference in references:
            ref_ngrams = Counter(everygrams(reference, min_len, max_len))
            tpfn = sum(ref_ngrams.values())  # True positives + False negatives.

            overlap_ngrams = ref_ngrams & hyp_ngrams
            tp = sum(overlap_ngrams.values())  # True positives.

            # While GLEU is defined as the minimum of precision and
            # recall, we can reduce the number of division operations by one by
            # instead finding the maximum of the denominators for the precision
            # and recall formulae, since the numerators are the same:
            #     precision = tp / tpfp
            #     recall = tp / tpfn
            #     gleu_score = min(precision, recall) == tp / max(tpfp, tpfn)
            n_all = max(tpfp, tpfn)

            if n_all > 0:
                hyp_counts.append((tp, n_all))

        # use the reference yielding the highest score
        if hyp_counts:
            n_match, n_all = max(hyp_counts, key=lambda hc: hc[0 ] /hc[1])
            corpus_n_match += n_match
            corpus_n_all += n_all

    # corner case: empty corpus or empty references---don't divide by zero!
    if corpus_n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = corpus_n_match / corpus_n_all

    return gleu_score, corpus_n_match, corpus_n_all
