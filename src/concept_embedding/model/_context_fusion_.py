import tensorflow as tf
from src.nn_utils.general import exp_mask_for_high_rank
from src.nn_utils.nn import bn_dense_layer


# # ----------------------fundamental-----------------------------
# ##########################################################
# common "context fusion" method
# ##########################################################
def multi_dimensional_attention(rep_tensor, rep_mask, keep_prob=1., is_train=None, wd=0., activation='relu'):
    # bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.compat.v1.variable_scope('multi_dimensional_attention'):
        map1 = bn_dense_layer( rep_tensor, ivec, True, 0., 'bn_dense_map1', activation, False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(       map1, ivec, True, 0., 'bn_dense_map2',   'linear', False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        return attn_output

