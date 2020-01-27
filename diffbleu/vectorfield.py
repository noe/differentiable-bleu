import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product
plt.style.use('classic')


def logit(x):
    eps = 1e-17
    if x == 0.:
        return logit(eps)
    if x == 1.:
        return logit(1. - eps)
    return np.log(x) - np.log(1 - x)


def reinforce(ref, hyp_probs, rewards, hyp_i=None, ref_j=None):
    extra = [] if hyp_i is None or ref_j is None else [hyp_i, ref_j]
    extra_l = '' if hyp_i is None or ref_j is None else 'hyp_i, ref_j,'
    probs = hyp_probs
    #probs = tf.Print(probs, extra + [ref, hyp_probs, probs, rewards], message=extra_l + 'ref, hyp_probs, probs, rew',  summarize=312341)
    tmp = tf.log(tf.clip_by_value(probs, 1e-20, 1.0))
    return -tf.reduce_sum(tmp) * rewards


class Model(object):
    def __init__(self, ref_len):
        idx = [0, 1]
        self.ref = tf.placeholder(name="ref",
                                  dtype=tf.float32,
                                  shape=(ref_len,))

        self.hyp_logits = tf.placeholder(name="hyp_logits",
                                         shape=(len(idx),),
                                         dtype=tf.float32)

        xe = tf.nn.sigmoid_cross_entropy_with_logits
        self.losses = [[xe(logits=self.hyp_logits[i], labels=self.ref[j])
                        for i in idx] for j in range(ref_len)]
        flattened_losses = [l for ll in self.losses for l in ll]
        self.total_loss = sum(flattened_losses) / len(flattened_losses)

        bernoulli = tf.distributions.Bernoulli(logits=self.hyp_logits, dtype=tf.float32)
        num_samples = 20
        samples = bernoulli.sample(sample_shape=(num_samples,))
        #samples = tf.Print(samples, [tf.nn.sigmoid(self.hyp_logits), samples], message="prob, samples", summarize=32132)
        comp_f = lambda a, b: tf.to_float(tf.equal(a, b))
        reward_f = lambda r, h: .25 * sum([comp_f(r[i], h[j]) for i, j in product([0, 1], repeat=2)])
        sample_rewards = [reward_f(self.ref, samples[i, :]) for i in range(num_samples)]
        #sample_rewards = tf.Print(sample_rewards, [sample_rewards], message="rewards", summarize=32421)
        self.reward = tf.stop_gradient(tf.reduce_mean(sample_rewards))

        individual_reward = [[tf.stop_gradient(tf.reduce_mean([comp_f(samples[k, i], self.ref[j])
                                                               for k in range(num_samples)]))
                              for i in idx] for j in range(ref_len)]
        #individual_reward = tf.Print(individual_reward, [tf.sigmoid(self.hyp_logits), individual_reward], summarize=234124, message="hyp, rew")
        s_logits = tf.sigmoid(self.hyp_logits)
        self.reinforce_losses = [[reinforce(self.ref, s_logits, individual_reward[j][i], s_logits[i], self.ref[j])
                                 for i in idx] for j in range(ref_len)]
        self.reinforce_loss = reinforce(self.ref, s_logits, self.reward)

    def plot_quiver(self, ref_vals, fig, use_reinforce=False):
        losses = self.reinforce_losses if use_reinforce else self.losses
        total_loss = self.reinforce_loss if use_reinforce else self.total_loss

        ax10 = plt.subplot2grid((2, 4), (1, 0), fig=fig)
        self._plot_quiver(ref_vals, losses[1][0], ax10)
        ax10.set_xlabel("$\sigma(hyp_0)$")
        ax10.set_ylabel("$\sigma(hyp_1)$")
        ax10.set_title("$-\\nabla_{hyp} J(\sigma(hyp_0), ref_1)$")

        ax00 = plt.subplot2grid((2, 4), (0, 0), sharex=ax10, fig=fig)
        plt.setp(ax00.get_xticklabels(), visible=False)
        self._plot_quiver(ref_vals, losses[0][0], ax00)
        ax00.set_ylabel("$\sigma(hyp_1)$")
        ax00.set_title("$-\\nabla_{hyp} J(\sigma(hyp_0), ref_0)$")

        ax11 = plt.subplot2grid((2, 4), (1, 1), sharey=ax10, fig=fig)
        plt.setp(ax11.get_yticklabels(), visible=False)
        self._plot_quiver(ref_vals, losses[1][1], ax11)
        ax11.set_xlabel("$\sigma(hyp_1)$")
        ax11.set_title("$-\\nabla_{hyp} J(\sigma(hyp_1), ref_1)$")

        ax01 = plt.subplot2grid((2, 4), (0, 1), sharex=ax11, sharey=ax00, fig=fig)
        plt.setp(ax01.get_xticklabels(), visible=False)
        plt.setp(ax01.get_yticklabels(), visible=False)
        self._plot_quiver(ref_vals, losses[0][1], ax01)
        ax01.set_title("$-\\nabla_{hyp} J(\sigma(hyp_1), ref_0)$")

        ax = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2, fig=fig)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

        #self._plot_quiver(ref_vals, total_loss, ax)
        #ax.set_xlabel("$\sigma(hyp_0)$")
        #ax.set_ylabel("$\sigma(hyp_1)$")
        #ax.set_title("$-\\nabla_{hyp} J(\sigma(hyp), ref)$")

        fig.tight_layout()#rect=(0.05, 0.09, 0.97, 0.95))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        return fig

    def _plot_quiver(self, ref_vals, loss, axs):
        grads = - tf.gradients(loss, self.hyp_logits)[0]
        xx = []
        yy = []
        uu = []
        vv = []
        with tf.Session() as sess:
            vals = np.arange(0., 1., 0.05).tolist()
            for x, y in product(vals, repeat=2):
                hyp_vals = np.array([logit(x), logit(y)])
                fd = {self.hyp_logits: hyp_vals, self.ref: ref_vals}
                grad_vals = sess.run(grads, feed_dict=fd)
                xx.append(x)
                yy.append(y)
                uu.append(grad_vals[0])
                vv.append(grad_vals[1])

        Q = axs.quiver(xx, yy, uu, vv, pivot='mid')
        axs.set_xlim([0., 1.])
        axs.set_ylim([0., 1.])


def main():
    seed = 12
    interactive = True
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    model = Model(ref_len=2)
    fig = plt.figure(figsize=(10, 5))

    val_list = list(product(range(2), repeat=2))
    val_list = [[0, 1]]
    for i, j in val_list:
        ref_vals = [i, j]
        model.plot_quiver(ref_vals=np.array(ref_vals).astype(np.float32),
                          fig=fig,
                          use_reinforce=True)

        if not interactive:
            file_name = './vectorfield_ref{}{}.pdf'.format(*ref_vals)
            fig.savefig(file_name, format='pdf')
        else:
            while True:  # workaround to TkAgg bug in MacOSX
                try:
                    plt.show()
                except UnicodeDecodeError:
                    continue
                break


if __name__ == '__main__':
    main()
