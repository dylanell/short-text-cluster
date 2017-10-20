import tensorflow as tf


# vanilla feed-forward neural network with sigmoid activation at all
# non-output layers and softmax activation at the output if out_type=
# classify (default), sigmoiud activation at the output of out_type=
# regression, or raw logits at the output if out_type=logits
class MultiLayerPerceptron(object):
    def __init__(self, dims, inputs, scope='MLP', targets=None, out_type='classify'):
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
        self.layers = []

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
