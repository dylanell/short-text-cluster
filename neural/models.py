import tensorflow as tf

def validateType(out_type):
    # check if valid output type
    valid_types = ['softmax', 'tanh', 'sigmoid', 'logits', 'raw']
    assert(out_type in valid_types), 'out_type \'%s\' is not valid' % out_type

class TextCNN(object):
    def __init__(self, emb_dims, filt_dims, fc_dims, inputs,
                 null_word, targets=None, scope='TCNN', out_type='logits', kp=1.0, eta=1e-3):
        # tensorflow ops
        self._encode = None
        self._predict = None
        self._loss = None

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

        self._test = kp

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

        self._prediction = tf.argmax(logits, axis=1)

        # define the loss
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets,
            logits=logits,
        )
        self._error = tf.reduce_mean(stepwise_cross_entropy)

        # define the optimizer
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

# end MultiLayerPerceptron
