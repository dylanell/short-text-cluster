import tensorflow as tf

def validateType(out_type):
    # check if valid output type
    valid_types = ['softmax', 'tanh', 'sigmoid', 'logits', 'raw']
    assert(out_type in valid_types), 'out_type \'%s\' is not valid' % out_type


# two layer ff neural network that takes input from summed word embeddings
class NBOW(object):
    def __init__(self, emb_dims, dims, inputs, targets, null_word,
                 kp=1.0, eta=1e-3, out_type='softmax'):
        # get arguments
        vocab_len, d = emb_dims
        latent_dim, K = dims

        # initializers for any weights and biases in the model
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = 0.01

        # embedding matrix for all words in our vocabulary
        embeddings = tf.get_variable('embedding_weights',
                                     shape =[vocab_len, d],
                                     initializer=w_init)

        """ embedding and reshaping """
        # mark all words that are not null
        used = tf.cast(tf.not_equal(inputs, null_word), tf.float32)

        # gett the lengths of each sentence in the input
        length = tf.cast(tf.reduce_sum(used, 1), tf.int32)

        # count how many samples in this batch
        num_samp = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(used, 1), -1), tf.int32))

        # create a mask to zero out null words
        mask = tf.expand_dims(used, [-1])

        # embed the words and mask so null characters are zero vectors
        inputs_emb = mask * tf.nn.embedding_lookup(embeddings, inputs)

        # inputs are the sum of all word vectors
        #inputs_vec = tf.reduce_sum(inputs_emb, axis=1)

        # inputs are the average word vector (zeros vectors not counted)
        inputs_vec = tf.reduce_sum(inputs_emb, axis=1) / tf.cast(tf.expand_dims(length, -1), tf.float32)

        self._test = inputs_vec

        self._latent = tf.contrib.layers.fully_connected(inputs_vec, latent_dim,
                                                       activation_fn=tf.nn.tanh)

        dropout = tf.nn.dropout(self._latent, keep_prob=kp)

        logits = tf.contrib.layers.fully_connected(dropout, K,
                                                 activation_fn=None)

        if (out_type=='sigmoid'):
            self._prediction = tf.nn.sigmoid(logits)

            stepwise_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        elif (out_type=='tanh'):
            self._prediction = tf.nn.tanh(logits)

            self._error = tf.reduce_mean((self._prediction - targets)**2)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        else:
            self._prediction = tf.argmax(logits, axis=1)

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)

    @property
    def predict(self):
        return self._prediction

    @property
    def encode(self):
        return self._latent

    @property
    def optimize(self):
        return self._optimizer

    @property
    def loss(self):
        return self._error

    @property
    def probe(self):
        return self._test


class LSTM(object):
    def __init__(self, emb_dims, dims, inputs, targets, null_word,
                 kp=1.0, eta=1e-3, out_type='softmax'):
        # get arguments
        vocab_len, d = emb_dims
        latent_dim, K = dims

        # initializers for any weights and biases in the model
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = 0.01

        # embedding matrix for all words in our vocabulary
        embeddings = tf.get_variable('embedding_weights',
                                     shape =[vocab_len, d],
                                     initializer=w_init)

        # mark the used words
        used = tf.cast(tf.not_equal(inputs, null_word), tf.int32)

        # gett eh lengths of each sentence in the input
        length = tf.cast(tf.reduce_sum(used, 1), tf.int32)

        # these ops embed the word vector placeholders using the embedding weights
        inputs_emb = tf.nn.embedding_lookup(embeddings, inputs)

        rnn_inputs = tf.transpose(inputs_emb, [1, 0, 2])
        #cnn_inputs = tf.transpose(inputs_emb, [0, 2, 1])

        cell = tf.contrib.rnn.LSTMCell(latent_dim, state_is_tuple=True)

        outputs, states = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=rnn_inputs,
                                           sequence_length=length,
                                           dtype=tf.float32,
                                           time_major=True)

        # get the hidden state of the network

        # take last hidden state as latent state
        #self._latent = states.h

        # take average of all non-zero hidden states as latent state
        self._latent = tf.reduce_sum(outputs, axis=0) / tf.cast(tf.expand_dims(length, -1), tf.float32)

        dropout = tf.nn.dropout(self._latent, keep_prob=kp)

        logits = tf.contrib.layers.fully_connected(dropout, K,
                                                 activation_fn=None)

        if (out_type=='sigmoid'):
            self._prediction = tf.nn.sigmoid(logits)

            stepwise_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        elif (out_type=='tanh'):
            self._prediction = tf.nn.tanh(logits)

            self._error = tf.reduce_mean((self._prediction - targets)**2)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        else:
            self._prediction = tf.argmax(logits, axis=1)

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)

    @property
    def predict(self):
        return self._prediction

    @property
    def encode(self):
        return self._latent

    @property
    def optimize(self):
        return self._optimizer

    @property
    def loss(self):
        return self._error


class DynamicCNN(object):
    def __init__(self, emb_dims, filt_dims, fc_dims, inputs, targets,
                 null_word, k_top_v=3, kp=1.0, eta=1e-3, out_type='softmax'):

        vocab_len, d = emb_dims
        latent_dim, K = fc_dims

        num_layers = len(filt_dims)

        # initializers for any weights and biases in the model
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = 0.01

        # filter = # [height, width, input_depth, num_filters]
        c1_filt_dim = [filt_dims[0][0], 1, 1, filt_dims[0][1]]
        c1_strides = [1, 1, 1, 1]
        c1_pad = [[0, 0],
                  [filt_dims[0][0] - 1, filt_dims[0][0] - 1],
                  [0, 0],
                  [0, 0]]

        # amount of kmax pooling at top layer
        k_top = tf.constant(k_top_v, dtype=tf.float32)

        # number of convolution layers
        L = tf.constant(num_layers, dtype=tf.float32)

        # filter weights for conv1
        c1_filt = tf.get_variable('conv1_filters',
                                     shape=c1_filt_dim,
                                     initializer=w_init)

        c1_bias = tf.get_variable("conv1_biases",
        initializer=(b_init * tf.ones([d/2, c1_filt_dim[-1]], tf.float32)))

        # embedding matrix for all words in our vocabulary
        embeddings = tf.get_variable('embedding_weights',
                                     shape =[vocab_len, d],
                                     initializer=w_init)

        """ embedding and reshaping """
        # mark all words that are not null
        used = tf.cast(tf.not_equal(inputs, null_word), tf.float32)

        # count how many samples in this batch (batch_size)
        num_samp = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(used, 1), -1), tf.int32))

        # count how many samples in this batch (batch_size)
        s = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(used, 0), -1), tf.int32))

        # create a mask to zero out null words
        mask = tf.expand_dims(used, [-1])

        # embed the words and mask
        inputs_emb = mask * tf.nn.embedding_lookup(embeddings, inputs)

        # reshape inputs to be like a batch of "images" with pixel depth 1,
        # therefore making them 4D tensors
        inputs_resh = tf.expand_dims(inputs_emb, -1)

        """ convolution layer 1 """
        # pad the inputs
        c1_padded = tf.pad(inputs_resh, c1_pad)

        # perform wide convolution 1
        c1_out = tf.nn.conv2d(c1_padded, c1_filt,
                                 strides=c1_strides, padding='VALID')

        # perform folding
        c1_fold = tf.add(c1_out[:, :, ::2, :], c1_out[:, :, 1::2, :])

        # get the dynamic k value
        k1 = tf.cast(tf.maximum(k_top,
                        tf.ceil(((L - 1)/L)*tf.cast(s, tf.float32))), tf.int32)

        # take the k1 max pool of the convolution output
        c1_topk = tf.nn.top_k(tf.transpose(c1_fold, [0, 3, 2, 1]), k1)[0]
        c1_topk = tf.transpose(c1_topk, [0, 3, 2, 1])

        # add bias and non-linear activation
        c1_act = tf.nn.tanh(tf.add(c1_topk, c1_bias))

        """ fully connected layer """
        # weightst for fully connected layer
        flat_size = (d/2) * k_top_v * c1_filt_dim[-1]

        flat = tf.reshape(c1_act, [num_samp, flat_size])

        dropout1 = tf.nn.dropout(flat, keep_prob=kp)

        self._latent = tf.contrib.layers.fully_connected(dropout1, latent_dim,
                                                 activation_fn=tf.nn.tanh)

        dropout2 = tf.nn.dropout(self._latent, keep_prob=kp)

        logits = tf.contrib.layers.fully_connected(dropout2, K,
                                                 activation_fn=None)

        if (out_type=='sigmoid'):
            self._prediction = tf.nn.sigmoid(logits)

            stepwise_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        elif (out_type=='tanh'):
            self._prediction = tf.nn.tanh(logits)

            self._error = tf.reduce_mean((self._prediction - targets)**2)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        else:
            self._prediction = tf.argmax(logits, axis=1)

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)

    @property
    def predict(self):
        return self._prediction

    @property
    def encode(self):
        return self._latent

    @property
    def optimize(self):
        return self._optimizer

    @property
    def loss(self):
        return self._error


class TextCNN(object):
    def __init__(self, emb_dims, filt_dims, fc_dims, inputs,
                 targets, null_word, kp=1.0, eta=1e-3, out_type='softmax'):

        vocab_len, d = emb_dims
        latent_dim, K = fc_dims

        flat_dim = 0
        for f in filt_dims:
            flat_dim += f[1]

        # initializers for any weights and biases in the model
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = 0.01

        # conv filter shapes = [height, width, input_depth, num_filters]
        c1_filt_dim = [filt_dims[0][0], d, 1, filt_dims[0][1]]
        c1_strides = [1, 1, 1, 1]
        c1_pad = [[0, 0],
                  [filt_dims[0][0] - 1, filt_dims[0][0] - 1],
                  [0, 0],
                  [0, 0]]

        c2_filt_dim = [filt_dims[1][0], d, 1, filt_dims[1][1]]
        c2_strides = [1, 1, 1, 1]
        c2_pad = [[0, 0],
                  [filt_dims[1][0] - 1, filt_dims[1][0] - 1],
                  [0, 0],
                  [0, 0]]

        c3_filt_dim = [filt_dims[2][0], d, 1, filt_dims[2][1]]
        c3_strides = [1, 1, 1, 1]
        c3_pad = [[0, 0],
                  [filt_dims[2][0] - 1, filt_dims[2][0] - 1],
                  [0, 0],
                  [0, 0]]

        # filter weights for conv1
        c1_filt = tf.get_variable('conv1_filters',
                                     shape=c1_filt_dim,
                                     initializer=w_init)

        c1_bias = tf.get_variable("conv1_biases",
        initializer=(b_init * tf.ones([1], tf.float32)))

        # filter weights for conv2
        c2_filt = tf.get_variable('conv2_filters',
                                     shape=c2_filt_dim,
                                     initializer=w_init)

        c2_bias = tf.get_variable("conv2_biases",
        initializer=(b_init * tf.ones([1], tf.float32)))

        # filter weights for conv3
        c3_filt = tf.get_variable('conv3_filters',
                                     shape=c3_filt_dim,
                                     initializer=w_init)

        c3_bias = tf.get_variable("conv3_biases",
        initializer=(b_init * tf.ones([1], tf.float32)))

        # embedding matrix for all words in our vocabulary
        embeddings = tf.get_variable('embedding_weights',
                                     shape =[vocab_len, d],
                                     initializer=w_init)

        """ embedding and reshaping """
        # mark all words that are not null
        used = tf.cast(tf.not_equal(inputs, null_word), tf.float32)

        # count how many samples in this batch
        num_samp = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(used, 1), -1), tf.int32))

        # create a mask to zero out null words
        mask = tf.expand_dims(used, [-1])

        # embed the words and mask
        inputs_emb = mask * tf.nn.embedding_lookup(embeddings, inputs)

        # reshape inputs to be like a batch of "images" with pixel depth 1,
        # therefore making them 4D tensors
        inputs_resh = tf.expand_dims(inputs_emb, -1)

        """ convolution layer 1 """
        # pad the inputs
        c1_padded = tf.pad(inputs_resh, c1_pad)

        # perform wide convolution 1
        c1_out = tf.nn.conv2d(c1_padded, c1_filt,
                                 strides=c1_strides, padding='VALID')

        c1_biased = c1_out + c1_bias

        c1_act = tf.nn.relu(c1_biased)

        c1_pool = tf.reduce_max(c1_act, axis=1)

        """ convolution layer 2 """
        # pad the inputs
        c2_padded = tf.pad(inputs_resh, c2_pad)

        # perform wide convolution 1
        c2_out = tf.nn.conv2d(c2_padded, c2_filt,
                                 strides=c2_strides, padding='VALID')

        c2_biased = c2_out + c2_bias

        c2_act = tf.nn.relu(c2_biased)

        c2_pool = tf.reduce_max(c2_act, axis=1)

        """ convolution layer 3 """
        # pad the inputs
        c3_padded = tf.pad(inputs_resh, c3_pad)

        # perform wide convolution 1
        c3_out = tf.nn.conv2d(c3_padded, c3_filt,
                                 strides=c3_strides, padding='VALID')

        c3_biased = c3_out + c3_bias

        c3_act = tf.nn.relu(c3_biased)

        c3_pool = tf.reduce_max(c3_act, axis=1)

        """ fully connected layer """
        concat = tf.concat([c1_pool, c2_pool, c3_pool], axis=2)
        flat = tf.reshape(concat, [num_samp, flat_dim])

        dropout1 = tf.nn.dropout(flat, keep_prob=kp)

        self._latent = tf.contrib.layers.fully_connected(dropout1, latent_dim,
                                                 activation_fn=tf.nn.tanh)

        dropout2 = tf.nn.dropout(self._latent, keep_prob=kp)

        logits = tf.contrib.layers.fully_connected(dropout2, K,
                                                 activation_fn=None)

        if (out_type=='sigmoid'):
            self._prediction = tf.nn.sigmoid(logits)

            stepwise_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        elif (out_type=='tanh'):
            self._prediction = tf.nn.tanh(logits)

            self._error = tf.reduce_mean((self._prediction - targets)**2)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)
        else:
            self._prediction = tf.argmax(logits, axis=1)

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )

            self._error = tf.reduce_mean(stepwise_cross_entropy)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(self._error)

    @property
    def predict(self):
        return self._prediction

    @property
    def encode(self):
        return self._latent

    @property
    def optimize(self):
        return self._optimizer

    @property
    def loss(self):
        return self._error


# vanilla feed-forward neural network with sigmoid activation at all
# non-output layers and softmax activation at the output if out_type=
# classify (default), sigmoiud activation at the output of out_type=
# regression, or raw logits at the output if out_type=logits
class MultiLayerPerceptron(object):
    def __init__(self, dims, inputs,
                 targets=None, scope='MLP', out_type='logits'):
        # tensorflow ops
        self._encode = None
        self._predict = None
        self._loss = None
        #self._optimize = None  # NOTE: not yet implemented internally

        # check if valid output type
        valid_types = ['softmax', 'tanh', 'sigmoid', 'logits', 'raw']
        assert(out_type in valid_types), 'out_type \'%s\' is not valid' % out_type

        # variables
        self.dims = dims
        self.inputs = inputs
        self.targets = targets
        self.out_type = out_type
        self.W = list()
        self.b = list()
        self.num_layers = len(dims)-1
        self.layers = list()

        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = 0.01

        # initialize parameters
        with tf.variable_scope(scope):
            for i in range(self.num_layers):
                w_name = 'w_' + str(i)
                b_name = 'b_' + str(i)
                self.W.append(tf.get_variable(w_name, shape = [self.dims[i], self.dims[i+1]], initializer=w_init))
                self.b.append(tf.get_variable(b_name, initializer=(b_init*tf.ones([self.dims[i+1]], tf.float32))))

        # build the model architecture
        for i in range(self.num_layers):
            if (i == (self.num_layers - 1)):
                if (self.out_type == 'softmax'):
                    # if layers has length, grab input from last layer,
                    # otherwise grab input from inputs
                    if (len(self.layers)):
                        y = tf.nn.softmax(tf.add(tf.matmul(self.layers[i-1], self.W[i]), self.b[i]))
                    else:
                        y = tf.nn.softmax(tf.add(tf.matmul(self.inputs, self.W[i]), self.b[i]))
                    # add this layer to the layer list
                    self.layers.append(y)
                elif (self.out_type == 'tanh'):
                    if(len(self.layers)):
                        y = tf.nn.tanh(tf.add(tf.matmul(self.layers[i-1], self.W[i]), self.b[i]))
                    else:
                        y = tf.nn.tanh(tf.add(tf.matmul(self.inputs, self.W[i]), self.b[i]))
                    self.layers.append(y)
                elif (self.out_type == 'sigmoid'):
                    if(len(self.layers)):
                        y = tf.nn.sigmoid(tf.add(tf.matmul(self.layers[i-1], self.W[i]), self.b[i]))
                    else:
                        y = tf.nn.sigmoid(tf.add(tf.matmul(self.inputs, self.W[i]), self.b[i]))
                    self.layers.append(y)
                elif (self.out_type == 'logits'):
                    if(len(self.layers)):
                        y = tf.add(tf.matmul(self.layers[i-1], self.W[i]), self.b[i])
                    else:
                        y = tf.add(tf.matmul(self.inputs, self.W[i]), self.b[i])
                    self.layers.append(y)
                elif (self.out_type == 'raw'):
                    if(len(self.layers)):
                        y = tf.add(tf.matmul(self.layers[i-1], self.W[i]), self.b[i])
                    else:
                        y = tf.add(tf.matmul(self.inputs, self.W[i]), self.b[i])
                    self.layers.append(y)
            else:
                if(len(self.layers)):
                    y = tf.nn.tanh(tf.add(tf.matmul(self.layers[i-1], self.W[i]), self.b[i]))
                else:
                    y = tf.nn.tanh(tf.add(tf.matmul(self.inputs, self.W[i]), self.b[i]))
                self.layers.append(y)

    def getVars(self):
        var_list = []

        # get all weight variables
        for v in self.W:
            var_list.append(v)

        # get all bias variables
        for v in self.b:
            var_list.append(v)

        return var_list

    @property
    def predict(self):
        if self._predict is None:
            self._predict = self.layers[-1]
        return self._predict

    @property
    def encode(self):
        if self._encode is None:
            if (self.num_layers >= 2):
                self._encode = self.layers[self.num_layers - 2]
            else:
                self._encode = self.layers[-1]
        return self._encode

    @property
    def loss(self):
        if self.targets is None:
            self._loss = None
        else:
            if self._loss is None:
                if (self.out_type == 'logits'):
                    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.targets,
                        logits=self.predict,
                    )
                    self._loss = tf.reduce_mean(stepwise_cross_entropy)
                else:
                    self._loss = tf.reduce_mean((self.predict - self.targets)**2)
        return self._loss
