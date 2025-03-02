import tensorflow as tf
from src.nn_utils.general import flatten, reconstruct, exp_mask, add_reg_without_bias
from src.nn_utils.general import selu

def scaled_tanh(x, scale=5.):
    return scale * tf.nn.tanh(1. / scale * x)

# ----------------------fundamental-----------------------------
def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        out = tf.nn.softmax(logits,-1)
        return out


def softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


# def softsel_with_dropout(target, logits, mask=None,
#                          keep_prob=1., is_train=None, scope=None):
#     """
#     :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
#     :param logits: [ ..., J], dtype=float
#     :param mask: [ ..., J], dtype=bool
#     :param scope:
#     :return: [..., d], dtype=float
#     """
#     with tf.name_scope(scope or "softsel_with_dropout"):
#         a = softmax(logits, mask=mask)
#         if keep_prob < 1.0:
#             assert is_train is not None
#             a = tf.cond(is_train, lambda: tf.nn.dropout(a, keep_prob), lambda: a)
#         target_rank = len(target.get_shape().as_list())
#         out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
#         return out

# ------------------------------------------------------
# ------------------ special ---------------------------


def _linear(xs, output_size, bias,bias_start=0., scope=None):
    with tf.compat.v1.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs,-1)
        input_size = x.get_shape()[-1]
        W = tf.compat.v1.get_variable('W', shape=[input_size,output_size],dtype=tf.float32,)
        if bias:
            bias = tf.compat.v1.get_variable('bias', shape=[output_size],dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)# for dense layer [(-1, d)]
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope) # dense
    print('dense output shape:',flat_out.get_shape())
    out = reconstruct(flat_out, args[0], 1) # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out


# def linear_3d(tensor, hn, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
#               is_train=None):
#
#     with tf.compat.v1.variable_scope(scope or 'linear_3d'):
#         assert len(tensor.get_shape().as_list()) == 3
#         num_int = tensor.get_shape()[0]
#         vec_int = tensor.get_shape()[-1]
#         weight_3d = tf.compat.v1.get_variable('weight_3d', [num_int, vec_int, hn], tf.float32)
#
#         if input_keep_prob < 1.0:
#             assert is_train is not None
#             tensor = tf.cond(is_train, lambda: tf.nn.dropout(tensor, input_keep_prob), lambda: tensor)
#         if bias:
#             bias_3d = tf.compat.v1.get_variable('bias_3d', [num_int, 1, hn], tf.float32,
#                                       tf.constant_initializer(bias_start))
#             linear_output = tf.matmul(tensor, weight_3d) + bias_3d
#         else:
#             linear_output = tf.matmul(tensor, weight_3d)
#
#         if squeeze:
#             assert hn == 1
#             linear_output = tf.squeeze(linear_output, -1)
#         if wd:
#             add_var_reg(weight_3d)
#         return linear_output


# def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
#     with tf.compat.v1.variable_scope(scope or "conv1d"):
#         num_channels = in_.get_shape()[-1] #dc
#         filter_ = tf.compat.v1.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
#         bias = tf.compat.v1.get_variable("bias", shape=[filter_size], dtype='float')
#         strides = [1, 1, 1, 1]
#         if is_train is not None and keep_prob < 1.0:
#             in_ = dropout(in_, keep_prob, is_train)
#         xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d] # (b,l,wl,d')
#         out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d] # (b,l,d')
#         return out


# def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
#     with tf.compat.v1.variable_scope(scope or "multi_conv1d"):
#         assert len(filter_sizes) == len(heights)
#         outs = []
#         for filter_size, height in zip(filter_sizes, heights):
#             if filter_size == 0:
#                 continue
#             # (b*sn,sl,wl,dc)
#             out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height)) #(b,l,d')
#             outs.append(out)
#         concat_out = tf.concat(outs,2) #(b,l,d)
#         return concat_out


# def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
#     with tf.compat.v1.variable_scope(scope or "highway_layer"):
#         d = arg.get_shape()[-1]  # embedding dim
#         trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob,
#                        is_train=is_train)
#         trans = tf.nn.relu(trans)
#         gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob,
#                       is_train=is_train)
#         gate = tf.nn.sigmoid(gate)
#         out = gate * trans + (1 - gate) * arg
#         return out
#
#         # read


# def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
#     with tf.compat.v1.variable_scope(scope or "highway_network"):
#         prev = arg
#         cur = None
#         for layer_idx in range(num_layers):
#             cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
#                                 input_keep_prob=input_keep_prob, is_train=is_train)
#             prev = cur
#         return cur


# def highway_net(
#         input_tensor, hn, bias, bias_start=0.0, scope=None, activation='relu', enable_bn=False,
#         wd=0., keep_prob=1.0, is_train=None):
#     ivec = input_tensor.get_shape().as_list()[-1]
#     with tf.compat.v1.variable_scope(scope or "highway_layer"):
#         trans = bn_dense_layer(input_tensor, ivec, bias, bias_start, 'map', activation, enable_bn, wd, keep_prob, is_train)
#         gate =  bn_dense_layer(input_tensor, ivec, bias, bias_start, 'gate', 'linear', enable_bn, wd, keep_prob, is_train)
#         gate = tf.nn.sigmoid(gate)
#         out = gate * trans + (1 - gate) * input_tensor
#
#         return out
# ------------------------------------------------------------
# --------------------get logits------------------------------


def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0,
               input_keep_prob=1.0, is_train=None, func=None):
    if func is None:
        func = "linear"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'double':
        return double_linear_logits(args, size, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                                    is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    else:
        raise Exception()


def double_linear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0,
                         is_train=None):
    with tf.compat.v1.variable_scope(scope or "Double_Linear_Logits"):
        first = tf.tanh(linear(args, size, bias, bias_start=bias_start, scope='first',
                               wd=wd, input_keep_prob=input_keep_prob, is_train=is_train))
        second = linear(first, 1, bias, bias_start=bias_start, squeeze=True, scope='second',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            second = exp_mask(second, mask)
        return second


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.compat.v1.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (isinstance(args, (tuple, list)) and not args):
            raise ValueError("`args` must be specified")
        if not isinstance(args, (tuple, list)):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank - 1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits
# ------------------------------------------------------------
# ------------------------------------------------------------

# ### --------------- other key func


# def feature_combination(org_tensor, new_features, wd=0., keep_prob=1., is_train=None, scope=None):
#     """
#     Features Combination 1: ruminating layer implementation
#     z = tanh(Wz0*in + Wz1*x1+..Wzn*xn+b);
#     f = tanh(Wf0*in + Wf1*x1+..Wfn*xn+b)
#     out = fquan\elem∗⁆t in+(1-f)\elem∗z
#     :param org_tensor: rank 3 with shape [bs,sl,vec]
#     :param new_features: list of tensor with rank 2 [bs,vec_x1] or [bs,sl,vec_x2]
#     :param wd:
#     :param keep_prob:
#     :param is_train:
#     :param scope:
#     :return:
#     """
#
#     with tf.compat.v1.variable_scope(scope or 'fea_comb'):
#         bs, sl, vec = tf.shape(org_tensor)[0],tf.shape(org_tensor)[1],tf.shape(org_tensor)[2]
#         vec_int = org_tensor.get_shape()[2]
#         features = [new_fea if len(new_fea.get_shape().as_list())==3 else tf.expand_dims(new_fea, 1)
#                     for new_fea in new_features]
#
#         # --- z ---
#         z_0 = linear([org_tensor], vec_int, True, scope='linear_W_z_0',
#                      wd=wd, input_keep_prob=keep_prob, is_train=is_train)
#         z_other = [linear([fea], vec_int, False, scope='linear_W_z_%d' % (idx_f + 1),
#                           wd=wd, input_keep_prob=keep_prob, is_train=is_train)
#                    for idx_f, fea in enumerate(features)]
#         z = tf.nn.tanh(sum([z_0] + z_other))
#
#         # --- f ---
#         f_0 = linear([org_tensor], vec_int, True, scope='linear_W_f_0',
#                      wd=wd, input_keep_prob=keep_prob, is_train=is_train)
#         f_other = [linear([fea], vec_int, False, scope='linear_W_f_%d' % (idx_f + 1),
#                           wd=wd, input_keep_prob=keep_prob, is_train=is_train)
#                    for idx_f, fea in enumerate(features)]
#         f = tf.nn.sigmoid(sum([f_0] + f_other))
#
#         return f*org_tensor + (1-f)*z


# def pooling_with_mask(rep_tensor, rep_mask, method='max', scope=None):
#     # rep_tensor have one more rank than rep_mask
#     with tf.name_scope(scope or '%s_pooling' % method):
#
#         if method == 'max':
#             rep_tensor_masked = exp_mask_for_high_rank(rep_tensor, rep_mask)
#             output = tf.reduce_max(rep_tensor_masked, -2)
#         elif method == 'mean':
#             rep_tensor_masked = mask_for_high_rank(rep_tensor, rep_mask)  # [...,sl,hn]
#             rep_sum = tf.reduce_sum(rep_tensor_masked, -2)  #[..., hn]
#             denominator = tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1, True)  # [..., 1]
#             denominator = tf.where(tf.equal(denominator, tf.zeros_like(denominator, tf.int32)),
#                                    tf.ones_like(denominator, tf.int32),
#                                    denominator)
#             output = rep_sum / tf.cast(denominator, tf.float32)
#         else:
#             raise AttributeError('No Pooling method name as %s' % method)
#         return output


# def fusion_two_mat(input1, input2, hn=None, scope=None, wd=0., keep_prob=1., is_train=None):
#     ivec1 = input1.get_shape()[-1]
#     ivec2 = input2.get_shape()[-1]
#     if hn is None:
#         hn = ivec1
#     with tf.compat.v1.variable_scope(scope or 'fusion_two_mat'):
#         part1 = linear(input1, hn, False, 0., 'linear_1', False, wd, keep_prob, is_train)
#         part2 = linear(input2, hn, True, 0., 'linear_2', False, wd, keep_prob, is_train)
#         return part1 + part2


# # ----------- with normalization ------------
def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    elif activation == 'sigmoid':
        activation_func = tf.nn.sigmoid
    elif activation == 'tanh':
        activation_func = tf.nn.tanh
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.compat.v1.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map', False, wd, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train,
                updates_collections=None,  decay=0.9,
                scope='bn')

        return activation_func(linear_map)


# def bn_layer(input_tensor, is_train, enable,scope=None):
#     with tf.compat.v1.variable_scope(scope or 'bn_layer'):
#         if enable:
#             return tf.contrib.layers.batch_norm(
#                 input_tensor, center=True, scale=True, is_training=is_train, scope='bn')
#         else:
#             return tf.identity(input_tensor)


# -------------- emb mat--------------
# def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_mat=None,
#                            extra_trainable=False, scope=None):
#     """
#     generate embedding matrix for looking up
#     :param dict_size: indices 0 and 1 corresponding to empty and unknown token
#     :param emb_len:
#     :param init_mat: init mat matching for [dict_size, emb_len]
#     :param extra_mat: extra tensor [extra_dict_size, emb_len]
#     :param extra_trainable:
#     :param scope:
#     :return: if extra_mat is None, return[dict_size+extra_dict_size,emb_len], else [dict_size,emb_len]
#     """
#     with tf.compat.v1.variable_scope(scope or 'gene_emb_mat'):
#         emb_mat_ept_and_unk = tf.constant(value=0, dtype=tf.float32, shape=[2, emb_len])
#         if init_mat is None:
#             emb_mat_other = tf.compat.v1.get_variable('emb_mat',[dict_size - 2, emb_len], tf.float32)
#         else:
#             emb_mat_other = tf.compat.v1.get_variable("emb_mat",[dict_size - 2, emb_len], tf.float32,
#                                             initializer=tf.constant_initializer(init_mat[2:], dtype=tf.float32,
#                                                                                 verify_shape=True))
#         emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)
#
#         if extra_mat is not None:
#             if extra_trainable:
#                 extra_mat_var = tf.compat.v1.get_variable("extra_emb_mat",extra_mat.shape, tf.float32,
#                                                 initializer=tf.constant_initializer(extra_mat,
#                                                                                     dtype=tf.float32,
#                                                                                     verify_shape=True))
#                 return tf.concat([emb_mat, extra_mat_var], 0)
#             else:
#                 extra_mat_con = tf.constant(extra_mat, dtype=tf.float32)
#                 return tf.concat([emb_mat, extra_mat_con], 0)
#         else:
#             return emb_mat


# def token_and_char_emb(if_token_emb=True, context_token=None, tds=None, tel=None,
#                        token_emb_mat=None, glove_emb_mat=None,
#                        if_char_emb=True, context_char=None, cds=None, cel=None,
#                        cos=None, ocd=None, fh=None, use_highway=True,highway_layer_num=None,
#                        wd=0., keep_prob=1., is_train=None):
#     with tf.compat.v1.variable_scope('token_and_char_emb'):
#         if if_token_emb:
#             with tf.compat.v1.variable_scope('token_emb'):
#                 token_emb_mat = generate_embedding_mat(tds, tel, init_mat=token_emb_mat,
#                                                        extra_mat=glove_emb_mat,
#                                                        scope='gene_token_emb_mat')
#
#                 c_token_emb = tf.nn.embedding_lookup(token_emb_mat, context_token)  # bs,sl,tel
#
#         if if_char_emb:
#             with tf.compat.v1.variable_scope('char_emb'):
#                 char_emb_mat = generate_embedding_mat(cds, cel, scope='gene_char_emb_mat')
#                 c_char_lu_emb = tf.nn.embedding_lookup(char_emb_mat, context_char)  # bs,sl,tl,cel
#
#                 assert sum(ocd) == cos and len(ocd) == len(fh)
#
#                 with tf.compat.v1.variable_scope('conv'):
#                     c_char_emb = multi_conv1d(c_char_lu_emb, ocd, fh, "VALID",
#                                               is_train, keep_prob, scope="xx")  # bs,sl,cocn
#         if if_token_emb and if_char_emb:
#             c_emb = tf.concat([c_token_emb, c_char_emb], -1)  # bs,sl,cocn+tel
#         elif if_token_emb:
#             c_emb = c_token_emb
#         elif if_char_emb:
#             c_emb = c_char_emb
#         else:
#             raise AttributeError('No embedding!')
#
#     if use_highway:
#         with tf.compat.v1.variable_scope('highway'):
#             c_emb = highway_network(c_emb, highway_layer_num, True, wd=wd,
#                                     input_keep_prob=keep_prob,is_train=is_train)
#     return c_emb


# def generate_feature_emb_for_c_and_q(feature_dict_size, feature_emb_len,
#                                      feature_name , c_feature, q_feature=None, scope=None):
#     with tf.compat.v1.variable_scope(scope or '%s_feature_emb' % feature_name):
#         emb_mat = generate_embedding_mat(feature_dict_size, feature_emb_len, scope='emb_mat')
#         c_feature_emb = tf.nn.embedding_lookup(emb_mat, c_feature)
#         if q_feature is not None:
#             q_feature_emb = tf.nn.embedding_lookup(emb_mat, q_feature)
#         else:
#             q_feature_emb = None
#         return c_feature_emb, q_feature_emb

# -------------------END---------------------

