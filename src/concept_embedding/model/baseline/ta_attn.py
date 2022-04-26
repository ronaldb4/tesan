import tensorflow as tf
import math

from src.nn_utils.general import mask_for_high_rank
from src.concept_embedding.model.__template__ import ModelTemplate
from src.concept_embedding.model._context_fusion_ import exp_mask_for_high_rank


##############################################################################
# MCE (CBOW with Time Aware Attention)
##############################################################################
class TaAttnModel(ModelTemplate):
    def __init__(self,scope, dataset):
        super(TaAttnModel, self).__init__(scope, dataset)

        # ------ start ------
        self.context_fusion = None

        self.code_embeddings = None
        self.final_embeddings = None

        self.nce_weights = None
        self.final_weights = None

        self.final_wgt_sim = None
        self.final_emb_sim = None

        self.train_masks = None

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
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
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
            init_code_embed = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            code_embeddings = tf.Variable(init_code_embed)
            context_embed = tf.nn.embedding_lookup(code_embeddings, self.context_codes)

        with tf.name_scope('ta_attn'):
            context_fusion = time_aware_attention(self.train_inputs,context_embed,self.context_mask,self.embedding_size,k=100)
        return context_fusion, code_embeddings

def time_aware_attention(train_inputs, embed, mask, embedding_size, k):
    with tf.compat.v1.variable_scope('time_aware_attention'):
        attn_weights = tf.Variable(tf.truncated_normal([embedding_size, k], stddev=1.0 / math.sqrt(k)))
        attn_biases = tf.Variable(tf.zeros([k]))

        # weight add bias
        attn_embed = tf.nn.bias_add(attn_weights, attn_biases)

        # multiplying it with Ei
        attn_scalars = tf.tensordot(embed, attn_embed, axes=[[2], [0]])

        # get abs of distance
        train_delta = tf.abs(train_inputs[:, :, 1])

        # distance function is log(dist+1)
        dist_fun = tf.log(tf.to_float(train_delta) + 1.0)

        # reshape the dist_fun
        dist_fun = tf.reshape(dist_fun, [tf.shape(dist_fun)[0], tf.shape(dist_fun)[1], 1])

        # the attribution logits
        attn_logits = tf.multiply(attn_scalars, dist_fun)

        # the attribution logits sum
        attn_logits_sum = tf.reduce_sum(attn_logits, -1, keepdims=True)
        attn_logits_sum = exp_mask_for_high_rank(attn_logits_sum, mask)

        # get weights via softmax
        attn_softmax = tf.nn.softmax(attn_logits_sum, 1)

        # the weighted sum
        attn_embed_weighted = tf.multiply(attn_softmax, embed)
        attn_embed_weighted = mask_for_high_rank(attn_embed_weighted, mask)

        reduced_embed = tf.reduce_sum(attn_embed_weighted, 1)
        # obtain two scalars
        scalar1 = tf.log(tf.to_float(tf.shape(embed)[1]) + 1.0)
        scalar2 = tf.reduce_sum(tf.pow(attn_softmax, 2), 1)
        # the scalared embed
        reduced_embed = tf.multiply(reduced_embed, scalar1)
        reduced_embed = tf.multiply(reduced_embed, scalar2)

        return reduced_embed, attn_embed_weighted
