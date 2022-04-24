import tensorflow as tf
from src.nn_utils.general import flatten, reconstruct

# ##########################################################
# this is common to all mortality prediction models
# ##########################################################
def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    assert not time_major
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_outputs, final_state = tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                                  initial_state=initial_state, dtype=dtype,
                                                  parallel_iterations=parallel_iterations,
                                                  swap_memory=swap_memory,
                                                  time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state
