import tensorflow as tf
from tensorflow.python.framework import tensor_shape

class DNNLayer(tf.keras.layers.Layer):
    """
    Model: The Multi Layer Perceptron

    Developer: anbo

    Date: 2019-11-29

    Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        - The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

    Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_units[-1])``.
        - For instance, for a 2D input with shape ``(batch_size, input_dim)``,
            the output would have shape ``(batch_size, hidden_units[-1])``.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, apply_final_act=True, seed=1024, **kwargs):
        """
        Args:
            hidden_units: list of positive integer, the layer number and units in each layer.
            activation: Activation function to use.
            l2_reg: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
            dropout_rate: float in [0,1). Fraction of the units to dropout.
            use_bn: bool. Whether use BatchNormalization before activation or not.
            apply_final_act: whether to apply act in final layer
            seed: A Python integer to use as random seed.

        """
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.apply_final_act = apply_final_act
        self.hidden_outputs = []
        super(DNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)

        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=[hidden_units[i], hidden_units[i + 1]],
                                        initializer=tf.keras.initializers.he_normal(seed=self.seed),
                                        regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                        trainable=True)
                        for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=[self.hidden_units[i], ],
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=True)
                     for i in range(len(self.hidden_units))]

        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization(name='bn_layer_{}'.format(i)) for i in range(len(self.hidden_units))]

        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate,
                                                           seed=self.seed + i, name='dropout_layer_{}'.format(i))
                                   for i in range(len(self.hidden_units))]

        self.activation_layers = [tf.keras.layers.Activation(self.activation, name='act_layer_{}'.format(i)) for i in range(len(self.hidden_units))]

        super(DNNLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            if i < len(self.hidden_units)-1 or self.apply_final_act:
                fc = self.activation_layers[i](fc)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc
            self.hidden_outputs.append(fc)

        return deep_input

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)
        elif len(self.hidden_units) > 0:
            return input_shape[:-1].concatenate(self.hidden_units[-1])
        else:
            return input_shape

    def get_config(self, ):
        config = {
            'activation': self.activation,
            'hidden_units': self.hidden_units,
            'l2_reg': self.l2_reg,
            'use_bn': self.use_bn,
            'dropout_rate': self.dropout_rate,
            'apply_final_act': self.apply_final_act,
            'seed': self.seed
        }
        base_config = super(DNNLayer, self).get_config()
        config.update(base_config)
        return config


class MaskBlockLayer(tf.keras.layers.Layer):
  """
    Model: Mask Block Layer

    Paper: MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

    Link: https://arxiv.org/abs/2102.07619

    Author: Zhiqiang Wang, Qingyun She, Junlin Zhang

    Developer: anbo

    Date: 2021-04-12

    inputs: inputs: tuple, (input_a, input_b)
            input_a: (batch, dim1)
            input_b: (batch, dim2)

    return: (batch, dim)
    """
  def __init__(self,
               apply_ln_emb=True,
               hidden_size=32,
               l2_reg=0.001,
               seed=11,
               **kwargs):
    super(MaskBlockLayer, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.l2_reg = l2_reg
    self.seed = seed
    self.apply_ln_emb = apply_ln_emb

  def build(self, input_shape):
    input_size = int(input_shape[1][-1])
    input_size_a = int(input_shape[0][-1])
    from alps_biz.core.layer.sequence import LayerNorm

    self.W_d1 = self.add_weight(
        name="W_d1",
        shape=(input_size, self.hidden_size),
        initializer=tf.keras.initializers.he_normal(seed=self.seed),
        regularizer=tf.keras.regularizers.L1L2(0, self.l2_reg),
        trainable=True)
    self.bias_d1 = self.add_weight(name='bias_d1',
                                   shape=(self.hidden_size,),
                                   initializer=tf.keras.initializers.Zeros(),
                                   regularizer=tf.keras.regularizers.L1L2(
                                       0, self.l2_reg),
                                   trainable=True)
    self.W_d2 = self.add_weight(
        name="W_d2",
        shape=(self.hidden_size, input_size_a),
        initializer=tf.keras.initializers.he_normal(seed=self.seed),
        regularizer=tf.keras.regularizers.L1L2(0, self.l2_reg),
        trainable=True)
    self.bias_d2 = self.add_weight(name='bias_d2',
                                   shape=(input_size_a,),
                                   initializer=tf.keras.initializers.Zeros(),
                                   regularizer=tf.keras.regularizers.L1L2(
                                       0, self.l2_reg),
                                   trainable=True)

    if self.apply_ln_emb:
      self.ln_emb_layer = LayerNorm(name='ln_emb_layer')
    self.ln_hid_layer = LayerNorm(name='ln_hidden_layer')
    self.hidden_weight = self.add_weight(
        name="hidden_weight",
        shape=(input_size_a, input_size_a),
        initializer=tf.keras.initializers.he_normal(seed=self.seed),
        regularizer=tf.keras.regularizers.L1L2(0, self.l2_reg),
        trainable=True)

    super(MaskBlockLayer, self).build(input_shape)

  def apply_instance_guide_mask(self, inputs):
    """
        :param inputs:
        :return:
        """
    output_1 = tf.keras.backend.bias_add(
        tf.keras.backend.dot(inputs, self.W_d1), self.bias_d1)
    output = tf.keras.backend.bias_add(
        tf.keras.backend.dot(output_1, self.W_d2), self.bias_d2)
    tf.logging.info('MaskBlockLayer: output {}'.format(output))
    return output

  def call(self, inputs, **kwargs):
    """
        Args:
            inputs: tuple, (input_a, input_b)
            input_a: (batch, dim1)
            input_b: (batch, dim2)

        returns: (batch, l)
        """
    input_a, input_b = inputs

    V_mask = self.apply_instance_guide_mask(input_b)
    if self.apply_ln_emb:
      input_a = self.ln_emb_layer(input_a)

    if tf.keras.backend.ndim(input_a) == 3 and tf.keras.backend.ndim(
        V_mask) == 2:
      V_mask = tf.expand_dims(V_mask, axis=1)

    V_masked = tf.multiply(V_mask, input_a)
    V_hidden = self.ln_hid_layer(
        tf.keras.backend.dot(V_masked, self.hidden_weight))
    V_output = tf.nn.relu(V_hidden)
    return V_output

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape[0])
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape

  def get_config(self):
    config = {
        'apply_ln_emb': self.apply_ln_emb,
        'hidden_size': self.hidden_size,
        'l2_reg': self.l2_reg,
        'seed': self.seed
    }
    base_config = super(MaskBlockLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaskNetModel(tf.keras.Model):
    """
    MModel: Mask Net model

    Paper: MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

    Link: https://arxiv.org/abs/2102.07619

    Author: Zhiqiang Wang, Qingyun She, Junlin Zhang

    Developer: anbo

    Date: 2021-04-12

    inputs:
        2d tensor (batch_size, dim_1)

    outputs:
        2d tensor (batch_size, out_dim)

    """
    def __init__(self, n_mask_blocks=1, lego='parallel', hidden_size=32, projection_hidden_units=[4,1], act_fn='relu', l2_reg=0.001,
                 dropout_rate=0, use_bn=False, seed=1024, apply_final_act=False, name='MaskNetModel'):
        """
        Args:
            hidden_units: list, unit in each hidden layer
            act_fn: string, activation function
            l2_reg: float, regularization value
            dropout_rate: float, fraction of the units to dropout.
            use_bn: boolean, if True, apply BatchNormalization in each hidden layer
            seed: int, random value for initialization

        """
        super(MaskNetModel, self).__init__(name='MaskNetModel')
        self.n_mask_blocks = n_mask_blocks
        self.lego = lego
        self.mask_block_layer = []

        for i in range(self.n_mask_blocks):
            if self.lego == 'serial' and i>0:
                apply_ln_emb = False
            else:
                apply_ln_emb = True
            self.mask_block_layer.append(MaskBlockLayer(apply_ln_emb=apply_ln_emb, hidden_size=hidden_size, l2_reg=l2_reg, seed=seed, name="{}_mask_block_{}th_layer".format(name, i)))

        self.dnn_layer = DNNLayer(hidden_units=projection_hidden_units, activation=act_fn, l2_reg=l2_reg,
                                  dropout_rate=dropout_rate, use_bn=use_bn, apply_final_act=apply_final_act, seed=seed, name="{}_dnn_layer".format(name))

    def call(self, inputs, extra_input=None, training=None):
        """
        Args:
            inputs: (batch, dim1)

        Returns:
            2d tensor (batch_size, out_dim)

        """
        if self.lego == 'parallel':
            mask_block_output_list = []
            for i in range(self.n_mask_blocks):
                mask_block_output_i = self.mask_block_layer[i]([inputs, inputs])
                mask_block_output_list.append(mask_block_output_i)

            mask_block_output = tf.keras.layers.Concatenate(axis=-1)(mask_block_output_list) if len(mask_block_output_list)>1 else mask_block_output_list[0]

        elif self.lego == 'serial':
            mask_block_output = inputs
            for i in range(self.n_mask_blocks):
                mask_block_output = self.mask_block_layer[i]([mask_block_output, inputs])

        else:
            raise NotImplementedError

        tf.logging.info('MaskNetModel: mask_block_output {}'.format(mask_block_output))

        if extra_input is not None:
            if tf.keras.backend.ndim(extra_input) > 2:
                extra_input = tf.keras.layers.Flatten()(extra_input)
            combined_input = tf.keras.layers.Concatenate(axis=-1)([mask_block_output, extra_input])
        else:
            combined_input = mask_block_output

        tf.logging.info('MaskNetModel: combined_input {}'.format(combined_input))
        output = self.dnn_layer(combined_input, training=training)

        return output