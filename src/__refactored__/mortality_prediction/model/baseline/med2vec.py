import tensorflow as tf
import json
import numpy as np

from src.__refactored__.nn_utils.general import mask_for_high_rank
from src.__refactored__.nn_utils.nn import bn_dense_layer
from src.__refactored__.mortality_prediction.model._dynamic_rnn_ import dynamic_rnn
from src.__refactored__.utils.configs import cfg
from src.__refactored__.mortality_prediction.model.__template__ import ModelTemplate
from src.__refactored__.mortality_prediction.data.datafile_util import fullpath, tesa_dict


class Med2VecModel(ModelTemplate):

    def __init__(self,scope, dataset):

        super(Med2VecModel, self).__init__(scope, dataset)
        # ------ start ------
        self.max_visits = dataset.max_visits
        self.max_len_visit = dataset.max_len_visit
        self.vocabulary_size = len(dataset.dictionary)
        # ---- place holder -----
        self.inputs = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_visits, self.max_len_visit],
                                     name='train_inputs')  # batch_size,max_visits, max_visit_len
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, 1],
                                     name='train_labels')  # batch_size, binary classification
        # ---- masks for padding -----
        self.inputs_mask = tf.cast(self.inputs, tf.bool)

        # ------------ other ---------
        self.batch_size = tf.shape(self.inputs)[0]

        # building model and other parts
        self.outputs, self.final_state, self.tensor_len = self.build_network()
        self.output = self.last_relevant()
        self.loss, self.optimizer, self.accuracy, self.props, self.yhat = self.build_loss_optimizer()

    def build_loss_optimizer(self):
        with tf.name_scope('loss_optimization'):
            logits = bn_dense_layer(self.output, 1, True, 0.,
                                    'bn_dense_map', 'sigmoid',
                                    False, wd=0., keep_prob=1.,
                                    is_train=True)

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(self.labels, tf.float32))
            tf.compat.v1.add_to_collection('losses', tf.reduce_mean(losses, name='loss_mean'))
            loss = tf.add_n(tf.compat.v1.get_collection('losses', self.scope), name='loss')
            tf.compat.v1.summary.scalar(loss.op.name, loss)
            tf.compat.v1.add_to_collection('ema/scalar', loss)

            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.round(logits), tf.cast(self.labels, tf.float32))
            with tf.name_scope('accuracy'):
                # Mean accuracy over all labels:
                # http://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return loss, optimizer, accuracy, logits, tf.round(logits)

    def build_accuracy(self):
        pass

    def last_relevant(self):
        batch_size = tf.shape(self.outputs)[0]
        max_length = tf.shape(self.outputs)[1]
        out_size = int(self.outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (self.tensor_len - 1)
        flat = tf.reshape(self.outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        print('last output shape: ', relevant.get_shape())
        return relevant

    def build_network(self):

        with tf.name_scope('code_embeddings'):
            ##############################################################################
            # med2vec - Baseline Method
            ##############################################################################
            med2vec_vectors_file = fullpath('dataset/baselines/med2vec/mimic3/outputs_med2vec.9.npz')
            med2vec_origin_dict_file = fullpath('dataset/baselines/med2vec/mimic3/outputs.types')

            origin_weights = np.load(med2vec_vectors_file)
            origin_weights = origin_weights['W_emb']
            embedding_size = origin_weights.shape[1]
            padding_array = np.zeros(embedding_size)
            with open(med2vec_origin_dict_file) as read_file:
                origin_dict = json.load(read_file)
            new_dict = {}
            for k in origin_dict.keys():
                key = k.replace('.', '')
                new_dict[key] = origin_dict[k]
            weights = []
            padding_count = 0
            for k, v in tesa_dict().items():
                if v not in new_dict.keys():
                    weights.append(padding_array)
                    padding_count += 1
                else:
                    weights.append(origin_weights[new_dict[v]])
            weights = np.array(weights, dtype=float)
            print(weights.shape, padding_count)

            code_embeddings = tf.Variable(weights, dtype=tf.float32)
            inputs_embed = tf.nn.embedding_lookup(code_embeddings, self.inputs)

        with tf.name_scope('visit_embedding'):
            # bs, max_visits, max_len_visit, embed_size
            inputs_masked = mask_for_high_rank(inputs_embed, self.inputs_mask)
            inputs_reduced = tf.reduce_mean(inputs_masked, 2)  # batch_size, max_visits, embed_size

        with tf.name_scope('visit_masking'):
            visit_mask = tf.reduce_sum(tf.cast(self.inputs_mask, tf.int32), -1)  # [bs,max_visits]
            visit_mask = tf.cast(visit_mask, tf.bool)
            tensor_len = tf.reduce_sum(tf.cast(visit_mask, tf.int32), -1)  # [bs]

        with tf.name_scope('RNN_computaion'):
            reuse = None if not tf.compat.v1.get_variable_scope().reuse else True
            # ############################################################
            # pg 505: All models were trained with 50,00 steps;
            # the batch size is 128 and the RNN cell type is GRU.
            # ############################################################
            cell = tf.contrib.rnn.GRUCell(cfg.hn, reuse=reuse)
            # if cfg.cell_type == 'gru':
            #     cell = tf.contrib.rnn.GRUCell(cfg.hn, reuse=reuse)
            # elif cfg.cell_type == 'lstm':
            #     cell = tf.contrib.rnn.LSTMCell(cfg.hn, reuse=reuse)
            # elif cfg.cell_type == 'basic_lstm':
            #     cell = tf.contrib.rnn.BasicLSTMCell(cfg.hn, reuse=reuse)
            # elif cfg.cell_type == 'basic_rnn':
            #     cell = tf.contrib.rnn.BasicRNNCell(cfg.hn, reuse=reuse)

            outputs, final_state = dynamic_rnn(cell, inputs_reduced, tensor_len, dtype=tf.float32)
        return outputs, final_state, tensor_len

