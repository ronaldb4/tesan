import tensorflow as tf
import math

# import attention mechanisms
from src.nn_utils.nn import linear, dropout, bn_dense_layer, scaled_tanh
from src.nn_utils.general import exp_mask_for_high_rank, mask_for_high_rank
from src.concept_embedding.model.__template__ import ModelTemplate
from src.concept_embedding.model._context_fusion_ import multi_dimensional_attention


##############################################################################
# RandomInterval - variant for ablation study
##############################################################################
class RandomIntervalModel(ModelTemplate):
    def __init__(self,scope,dataset):
        super(RandomIntervalModel, self).__init__(scope,dataset)

        # ------ start ------
        self.context_fusion = None

        self.code_embeddings = None
        self.final_embeddings = None

        self.nce_weights = None
        self.final_weights = None

        self.final_wgt_sim = None
        self.final_emb_sim = None


        # ---- place holder -----
        self.train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[None, None, 2], name='train_inputs')
        self.train_masks = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name='train_masks')

        self.train_labels = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='train_labels')
        self.valid_dataset = tf.constant(self.valid_samples, dtype=tf.int32, name='valid_samples')

        # ------------ other ---------
        self.output_class = 3  # 0 for contradiction, 1 for neural and 2 for entailment
        self.batch_size = tf.shape(self.train_inputs)[0]
        self.code_len = tf.shape(self.train_inputs)[1]

        # context codes
        self.context_codes = self.train_inputs[:, :, 0]

        # mask for padding codes are all 0, actual codes are 1
        self.context_mask = tf.cast(self.context_codes, tf.bool)

        # time interval between context code and label code
        self.context_delta = self.train_inputs[:, :, 1]

        #building model and other parts
        self.context_fusion, self.code_embeddings = self.build_network()
        self.loss, self.optimizer, self.nce_weights = self.build_loss_optimizer()
        self.final_embeddings, self.final_weights = self.build_embedding()
        self.final_emb_sim, self.final_wgt_sim = self.build_similarity()

    def build_loss_optimizer(self):
        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.random.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        losses = tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=self.train_labels,
            inputs=self.context_fusion,
            num_sampled=self.num_negative_examples,
            num_classes=self.vocabulary_size)

        # loss = tf.reduce_mean(losses, name='loss_mean')
        tf.compat.v1.add_to_collection('losses', tf.reduce_mean(losses, name='loss_mean'))
        loss = tf.add_n(tf.compat.v1.get_collection('losses', self.scope), name='loss')
        tf.compat.v1.summary.scalar(loss.op.name, loss)
        tf.compat.v1.add_to_collection('ema/scalar', loss)

        optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)
        return loss, optimizer, nce_weights

    def build_accuracy(self):
        pass

    def build_embedding(self):
        with tf.name_scope('build_embedding'):
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.code_embeddings), 1, keepdims =True))
            final_embeddings = self.code_embeddings / norm

            weights_norm = tf.sqrt(tf.reduce_sum(tf.square(self.nce_weights), 1, keepdims=True))
            final_weights = self.nce_weights / weights_norm
        return final_embeddings, final_weights

    def build_similarity(self):
        with tf.name_scope('build_similarity'):
            valid_embeddings = tf.nn.embedding_lookup(self.final_embeddings, self.valid_dataset)
            final_emb_sim = tf.matmul(valid_embeddings, self.final_embeddings, transpose_b=True)

            valid_embeddings = tf.nn.embedding_lookup(self.final_weights, self.valid_dataset)
            final_wgt_sim = tf.matmul(valid_embeddings, self.final_weights, transpose_b=True)

        return final_emb_sim, final_wgt_sim


    def build_network(self):
        # Look up embeddings for inputs.
        with tf.name_scope('code_embeddings'):
            init_code_embed = tf.random.uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            code_embeddings = tf.Variable(init_code_embed)
            context_embed = tf.nn.embedding_lookup(code_embeddings, self.context_codes)

        ##############################################################################
        # RandomInterval - ???????? variant for ablation study ????????
        ##############################################################################
        with tf.name_scope('random_interval'):
            # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]
            init_date_embed = tf.random.uniform([self.dates_size, self.embedding_size], -1.0, 1.0)
            date_embeddings = tf.Variable(init_date_embed)

            date_embed = tf.nn.embedding_lookup(date_embeddings, self.train_masks)

            # self_attention
            cntxt_embed = temporal_delta_sa_with_dense(rep_tensor=context_embed,
                                                      rep_mask=self.context_mask,
                                                      delta_tensor=date_embed,
                                                      is_train=True,
                                                      activation=self.activation,
                                                      is_scale=self.is_scale)

            # Attention pooling
            context_fusion = multi_dimensional_attention(cntxt_embed, self.context_mask, is_train=True)
        return context_fusion, code_embeddings

def temporal_delta_sa_with_dense(rep_tensor, rep_mask, delta_tensor, keep_prob=1., is_train=None, wd=0., activation='relu',is_scale=True):

    batch_size, code_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.compat.v1.variable_scope('temporal_attention'):
        # mask generation
        attn_mask = tf.cast(tf.linalg.tensor_diag(- tf.ones([code_len], tf.int32)) + 1, tf.bool)  # batch_size, code_len, code_len

        # non-linear for context
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, code_len, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # non-linear for time interval
        time_rep_map = bn_dense_layer(delta_tensor, ivec, True, 0., 'bn_dense_map_time', activation,
                                 False, wd, keep_prob, is_train) # bs,sl,sl,vec
        time_rep_map_dp = dropout(time_rep_map, keep_prob, is_train)

        # attention
        with tf.compat.v1.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.compat.v1.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))

            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec

            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            time_rep_etd = linear(time_rep_map_dp, ivec, False, scope='linear_time') # bs,sl,sl,vec
            # logits = scaled_tanh(dependent_etd + head_etd + time_rep_etd + f_bias, 5.0)  # bs,sl,sl,vec

            attention_fact = dependent_etd + head_etd + time_rep_etd + f_bias
            if is_scale:
                logits = scaled_tanh(attention_fact, 5.0)  # bs,sl,sl,vec
            else:
                fact_bias = tf.compat.v1.get_variable('fact_bias', [ivec], tf.float32, tf.constant_initializer(0.))
                logits = linear(tf.nn.tanh(attention_fact), ivec, False, scope='linear_attn_fact') + fact_bias

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.compat.v1.variable_scope('output'):
            o_bias = tf.compat.v1.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) + o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)# bs,sl,vec

        return output
