import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas
import numpy as np
plt.style.use('seaborn')

NLTK_BLEU_RUN = "run_.-tag-nltk.bleu.csv"
NLTK_GLEU_RUN = "run_.-tag-nltk.gleu.csv"
SCORE_RUN = {'gleu': NLTK_GLEU_RUN, 'bleu': NLTK_BLEU_RUN}
OUR_BLEU_RUN = "run_.-tag-bleu.csv"
OUR_GLEU_RUN = "run_.-tag-gleu.csv"

NLTK_GLEU_UNIT = "nltk GLEU of the training minibatch"
NLTK_BLEU_UNIT = "nltk BLEU of the training minibatch"
SCORE_UNIT = {'gleu': NLTK_GLEU_UNIT, 'bleu': NLTK_BLEU_UNIT}


STEPS_UNIT = "steps"
BPTT_LABEL = 'Backprop Through Time'
TF_LABEL = "Teacher Forcing"


def load_data(f):
    data = pandas.read_csv(f)
    del data['Wall time']
    return data


def tensorboard_window(smoothing, y):
    # same as tensorboard:
    # https://github.com/tensorflow/tensorboard/blob/f801ebf1f9fbfe2baee1ddd65714d0bccc640fb1/tensorboard/plugins/scalar/vz_line_chart/vz-line-chart.ts#L55
    ratio = (1000.**smoothing - 1) / 999.
    return int(.5 * len(y) * ratio)


def smooth(y, smoothing=None):
    if smoothing is None:
        return y

    window_len = tensorboard_window(smoothing, y)

    # taken from https://stackoverflow.com/a/41420229/674487

    if (window_len // 2) * 2 == window_len:  # Allow even value for window_len
        window_len = window_len - 1

    front = np.zeros(window_len // 2)
    back = np.zeros(window_len // 2)

    for i in range(1, (window_len // 2) * 2, 2):
        front[i // 2] = np.convolve(y[:i], np.ones((i,)) / i, mode='valid')

    for i in range(1, (window_len // 2) * 2, 2):
        back[i // 2] = np.convolve(y[-i:], np.ones((i,)) / i, mode='valid')

    central = np.convolve(y, np.ones((window_len,)) / window_len, mode='valid')

    return np.concatenate([front, central, back[::-1]])


def figure(data_dict, x_axis=(0, 10000), y_axis=(0., 1.), y_label="", legend_loc=2, smoothing=.7):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(x_axis[0], x_axis[1])
    ax.set_ylim(y_axis[0], y_axis[1])
    ax.set_xlabel(STEPS_UNIT)
    ax.set_ylabel(y_label)
    handles = []
    for label, data in data_dict.items():
        x = data['Step'].tolist()
        y = data['Value'].tolist()
        smooth_y = smooth(y, smoothing=smoothing)
        smooth_line, = ax.plot(x, y, alpha=.3, label=label)
        line, = ax.plot(x, smooth_y, label=label, color=smooth_line._color)
        handles.append(line)
    legend = ax.legend(handles=handles, loc=legend_loc, frameon=True)
    fig.tight_layout()
    return fig


def plot_reverse(data_dir, interactive):
    bptt_dir = data_dir / "reverse_bptt"
    teacherf_dir = data_dir / "reverse_teacherforcing"
    gleu_softmax_dir = data_dir / "reverse_gleu_softmax"
    gleu_gs_dir = data_dir / "reverse_gleu_gs_0.5"
    bptt_nltk_bleu = load_data(bptt_dir / NLTK_BLEU_RUN)
    bptt_nltk_gleu = load_data(bptt_dir / NLTK_GLEU_RUN)
    teacherf_nltk_bleu = load_data(teacherf_dir / NLTK_BLEU_RUN)
    teacherf_nltk_gleu = load_data(teacherf_dir / NLTK_GLEU_RUN)
    gleu_softmax_nltk_gleu = load_data(gleu_softmax_dir / NLTK_GLEU_RUN)
    gleu_gs_nltk_gleu = load_data(gleu_gs_dir / NLTK_GLEU_RUN)
    bleu_softmax_nltk_bleu = load_data(gleu_softmax_dir / NLTK_BLEU_RUN)
    bleu_gs_nltk_bleu = load_data(gleu_gs_dir / NLTK_BLEU_RUN)

    bptt_vs_tf_data = {BPTT_LABEL: bptt_nltk_bleu, TF_LABEL: teacherf_nltk_bleu}
    f1 = figure(bptt_vs_tf_data, y_label=NLTK_BLEU_UNIT)
    f1_filename = data_dir / 'reverse_bptt_vs_tf.pdf'

    tf_gleu_softmax_gs_data = {BPTT_LABEL: bptt_nltk_gleu,
                               'GLEU loss + softmax': gleu_softmax_nltk_gleu,
                               'GLEU loss + Hard GS': gleu_gs_nltk_gleu}
    f2 = figure(tf_gleu_softmax_gs_data, y_label=NLTK_GLEU_UNIT, y_axis=[0., 0.2])
    f2_filename = data_dir / "reverse_gleu.pdf"

    if interactive:
        plt.show()
    else:
        f1.savefig(str(f1_filename.resolve()), format='pdf')
        f2.savefig(str(f2_filename.resolve()), format='pdf')

    # These values are always zero, so it's not worth it to plot
    # tf_bleu_softmax_gs_data = {BPTT_LABEL: bptt_nltk_bleu,
    #                            'BLEU loss + softmax': bleu_softmax_nltk_bleu,
    #                            'BLEU loss + Hard GS': bleu_gs_nltk_bleu}
    # figure(tf_bleu_softmax_gs_data, y_label="nltk BLEU", y_axis=[0., 0.01])


def plot_toy(data_dir, interactive):
    for score, loc in [('gleu', 4), ('bleu', 1)]:
        score_data = {}
        for vocab_size in [10000, 20000]:
            for seq_length in [10, 50]:
                dir = data_dir / "toy_{}_vocab{}_len{}".format(score, vocab_size, seq_length)
                data_file = dir / SCORE_RUN[score]
                label = "{} length: {}, vocab: {}".format(score.upper(), seq_length, vocab_size)
                score_data[label] = load_data(data_file)
        fig = figure(score_data,
                     y_label="nltk {}".format(score.upper()),
                     legend_loc=loc,
                     y_axis=[0., 1.01],
                     smoothing=None)
        if interactive:
            plt.show()
        else:
            fig_file = data_dir / "toy_{}.pdf".format(score)
            fig.savefig(str(fig_file.resolve()), format='pdf')


def main():
    default_dir = Path(__file__).parents[4] / "data"
    data_dir = default_dir if len(sys.argv) < 2 else Path(sys.argv[1])
    print("Data directory: {}".format(data_dir))
    interactive = False
    plot_reverse(data_dir, interactive)
    plot_toy(data_dir, interactive)


if __name__ == '__main__':
    main()
