import tensorflow as tf
from tensorflow.contrib.seq2seq import (TrainingHelper,
                                        GreedyEmbeddingHelper,
                                        BasicDecoder,
                                        BahdanauAttention,
                                        AttentionWrapper,
                                        dynamic_decode,
                                        sequence_loss,
                                        BeamSearchDecoder)


class Seq2SeqAttn(object):
    def __init__(self,
                 inputs,
                 targets,
                 src_vocab_size,
                 src_max_length,
                 tgt_vocab_size,
                 tgt_max_length,
                 emb_dim,
                 num_units,
                 batch_size,
                 eos_token,
                 is_train,
                 share_embeddings=False,
                 teacher_forcing=False):

        xavier = tf.contrib.layers.xavier_initializer
        start_tokens = tf.zeros([batch_size], dtype=tf.int32)
        input_lengths = tf.argmin(tf.abs(inputs - eos_token), axis=-1, output_type=tf.int32)

        target_lengths = tf.argmin(tf.abs(targets - eos_token), axis=-1, output_type=tf.int32)

        input_embedding_table = tf.get_variable("encoder_embedding", [src_vocab_size, emb_dim], initializer=xavier(), dtype=tf.float32)
        input_embedding = tf.nn.embedding_lookup(input_embedding_table, inputs)
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=False)
        encoder_cell = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_cell,
                                                     input_keep_prob=0.8,
                                                     output_keep_prob=1.0)

        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        (encoder_output,
         encoder_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                          cell_bw=encoder_cell,
                                                          inputs=input_embedding,
                                                          sequence_length=input_lengths,
                                                          dtype=tf.float32,
                                                          time_major=False)

        encoder_output = tf.concat(encoder_output, axis=2)
        encoder_state = tf.concat([encoder_state[0], encoder_state[1]], axis=1)

        if share_embeddings:
            assert src_vocab_size == tgt_vocab_size
            target_embedding_table = input_embedding_table
        else:
            target_embedding_table = tf.get_variable("decoder_embedding", [src_vocab_size, emb_dim], initializer=xavier(), dtype=tf.float32)

        prefixed_targets = tf.concat([tf.expand_dims(start_tokens, 1), targets], axis=1)
        target_embedding = tf.nn.embedding_lookup(target_embedding_table, prefixed_targets)

        if teacher_forcing:
            helper = TrainingHelper(target_embedding,
                                    target_lengths + 1,
                                    time_major=False)
        else:
            helper = GreedyEmbeddingHelper(target_embedding_table, start_tokens, eos_token)

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units * 2, state_is_tuple=False)
        projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)

        attention_mechanism = BahdanauAttention(num_units,
                                                encoder_output,
                                                memory_sequence_length=input_lengths)

        decoder_cell = AttentionWrapper(decoder_cell,
                                        attention_mechanism,
                                        attention_layer_size=num_units)
        #decoder_cell = tf.nn.rnn_cell.DropoutWrapper(cell=decoder_cell,
        #                                             input_keep_prob=0.8,
        #                                             output_keep_prob=1.0)

        encoder_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        decoder = BasicDecoder(cell=decoder_cell,
                               helper=helper,
                               initial_state=encoder_state,
                               output_layer=projection_layer)

        decoder_outputs, states, lengths = dynamic_decode(decoder,
                                                          output_time_major=False,
                                                          impute_finished=True,
                                                          maximum_iterations=tgt_max_length)
        unpadded_logits = decoder_outputs.rnn_output
        missing_elems = tgt_max_length - tf.shape(unpadded_logits)[1]
        padding = [[0, 0], [0, missing_elems], [0, 0]]
        logits = tf.pad(unpadded_logits, padding, 'CONSTANT', constant_values=0.)

        weights = tf.sequence_mask(target_lengths + 1, # the "+1" is to include EOS
                                   maxlen=tgt_max_length,
                                   dtype=tf.float32)
        #self.mle_loss = sequence_loss(targets=targets,
        #                              logits=logits,
        #                              weights=weights,
        #                              average_across_batch=True)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        mle_loss = (tf.reduce_sum(crossent * weights) / batch_size)
        preds = decoder_outputs.sample_id

        self.preds = preds
        self.logits = logits
        self.mle_loss = mle_loss





