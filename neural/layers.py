import tensorflow as tf

# implements a k-competitive filtering of a 1D activation (from KATE paper:
# http://www.cs.rpi.edu/~zaki/PaperDir/SIGKDD17.pdf)
# Author: Yu Chen
def KCompetitiveLayer(x, topk, factor=6.26):
    #print 'run k_comp_tanh'
    dim = int(x.get_shape()[1])
    # batch_size = tf.to_float(tf.shape(x)[0])
    if topk > dim:
        warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
        topk = dim

    P = (x + tf.abs(x)) / 2
    N = (x - tf.abs(x)) / 2

    values, indices = tf.nn.top_k(P, topk / 2) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
    # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
    my_range_repeated = tf.tile(my_range, [1, topk / 2])  # will be [[0, 0], [1, 1]]
    full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    full_indices = tf.reshape(full_indices, [-1, 2])
    P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)


    values2, indices2 = tf.nn.top_k(-N, topk - topk / 2)
    my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
    my_range_repeated = tf.tile(my_range, [1, topk - topk / 2])
    full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
    full_indices2 = tf.reshape(full_indices2, [-1, 2])
    N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0., validate_indices=False)

    P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True) # 6.26
    N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
    P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]), default_value=0., validate_indices=False)
    N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]), default_value=0., validate_indices=False)

    res = P_reset - N_reset

    return res
