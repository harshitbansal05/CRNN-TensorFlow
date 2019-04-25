import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class CRNN(object):
    def __init__(self, phase, hidden_nums, layers_nums, num_classes):
        super(CRNN, self).__init__()
        if phase == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._hidden_nums = hidden_nums
        self._layers_nums = layers_nums
        self._num_classes = num_classes
        self._is_training = self._init_phase()

    def _init_phase(self):
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
               stride=1, w_init=None, b_init=None,
               use_bias=True, name=None):

        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3
            in_channel = in_shape[channel_axis]

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1]
            else:
                strides = [1, stride, stride, 1]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format='NHWC')
            
            ret = tf.identity(tf.nn.bias_add(conv, b, data_format='NHWC')
                              if use_bias else conv, name=name)

        return ret

    def relu(inputdata, name=None):
        
        return tf.nn.relu(features=inputdata, name=name)

    def maxpooling(inputdata, kernel_size, stride=None,
                   padding='VALID', name=None):
        
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1]
        else:
            kernel = [1, kernel_size, kernel_size, 1]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1]
        else:
            strides = [1, stride, stride, 1]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    def dropout(inputdata, keep_prob, is_training, name, noise_shape=None):
        
        return tf.cond(
            pred=is_training,
            true_fn=lambda: tf.nn.dropout(
                inputdata,
                keep_prob=keep_prob,
                noise_shape=noise_shape,
                ),
            false_fn=lambda: inputdata,
            name=name
        )

    def layerbn(inputdata, is_training, name, momentum=0.999, eps=1e-3):

        return tf.layers.batch_normalization(
            inputs=inputdata, training=is_training, name=name, momentum=momentum, epsilon=eps)

    def squeeze(inputdata, axis=None, name=None):
        
        return tf.squeeze(input=inputdata, axis=axis, name=name)

    def conv_stage(self, inputdata, out_dims, name, is_bn=True, is_pool=True):
        with tf.variable_scope(name_or_scope=name):
            conv = self.conv2d(
                inputdata=inputdata, out_channel=out_dims,
                kernel_size=3, stride=1, use_bias=True, name='conv'
            )
            if is_bn: 
                conv = self.layerbn(
                    inputdata=conv,
                    is_training=self._is_training, 
                    name='bn'
                    )
            conv = self.relu(
                inputdata=conv,
                name='relu'
                )
            if is_pool:
                conv = self.maxpooling(
                    inputdata=conv,
                    kernel_size=2,
                    stride=2, name='max_pool'
                )
        return conv

    def feature_sequence_extraction(self, inputdata, name):
        with tf.variable_scope(name_or_scope=name):
            conv1 = self.conv_stage(
                inputdata=inputdata, out_dims=64, name='conv1'
            )
            conv2 = self.conv_stage(
                inputdata=conv1, out_dims=128, name='conv2'
            )
            conv3 = self.conv2d(
                inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3'
            )
            bn3 = self.layerbn(
                inputdata=conv3, is_training=self._is_training, name='bn3'
            )
            relu3 = self.relu(
                inputdata=bn3, name='relu3'
            )
            conv4 = self.conv2d(
                inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4'
            )
            bn4 = self.layerbn(
                inputdata=conv4, is_training=self._is_training, name='bn4'
            )
            relu4 = self.relu(
                inputdata=bn4, name='relu4')
            max_pool4 = self.maxpooling(
                inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID', name='max_pool4'
            )
            conv5 = self.conv2d(
                inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5'
            )
            bn5 = self.layerbn(
                inputdata=conv5, is_training=self._is_training, name='bn5'
            )
            relu5 = self.relu(
                inputdata=bn5, name='bn5'
            )
            conv6 = self.conv2d(
                inputdata=relu5, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv6'
            )
            bn6 = self.layerbn(
                inputdata=conv6, is_training=self._is_training, name='bn6'
            )
            relu6 = self.relu(
                inputdata=bn6, name='relu6'
            )
            max_pool6 = self.maxpooling(
                inputdata=relu6, kernel_size=[2, 1], stride=[2, 1], name='max_pool6'
            )
            conv7 = self.conv2d(
                inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7'
            )
            bn7 = self.layerbn(
                inputdata=conv7, is_training=self._is_training, name='bn7'
            )
            relu7 = self.relu(
                inputdata=bn7, name='bn7'
            )

        return relu7

    def map_to_sequence(self, inputdata, name):
        with tf.variable_scope(name_or_scope=name):
            shape = inputdata.get_shape().as_list()
            assert shape[1] == 1
            ret = self.squeeze(inputdata=inputdata, axis=1, name='squeeze')

        return ret

    def sequence_label(self, inputdata, name):
        with tf.variable_scope(name_or_scope=name):
            fw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]
            
            bw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, inputdata,
                dtype=tf.float32
            )

            stack_lstm_layer = self.dropout(
                inputdata=stack_lstm_layer,
                keep_prob=0.5,
                is_training=self._is_training,
                name='sequence_drop_out'
            )

            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()

            shape = tf.shape(stack_lstm_layer)
            rnn_reshaped = tf.reshape(stack_lstm_layer, [shape[0] * shape[1], shape[2]])

            w = tf.get_variable(
                name='w',
                shape=[hidden_nums, self._num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                trainable=True
            )

            logits = tf.matmul(rnn_reshaped, w, name='logits')
            logits = tf.reshape(logits, [shape[0], shape[1], self._num_classes], name='logits_reshape')
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')
            rnn_out = tf.transpose(logits, [1, 0, 2], name='transpose_time_major')

        return rnn_out, raw_pred

    def inference(self, inputdata, name, reuse=False):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            inputdata = tf.divide(inputdata, 255.0)   # DOUBT

            cnn_out = self.feature_sequence_extraction(
                inputdata=inputdata, 
                name='feature_extraction_module'
            )

            sequence = self.map_to_sequence(
                inputdata=cnn_out, 
                name='map_to_sequence_module'
            )

            net_out, raw_pred = self.sequence_label(
                inputdata=sequence, 
                name='sequence_rnn_module'
            )

        return net_out

    def compute_loss(self, inputdata, labels, name, reuse):
        inference_ret = self.inference(
            inputdata=inputdata,
            name=name, 
            reuse=reuse
        )

        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=labels,
                inputs=inference_ret,
                sequence_length=CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE)
                ),
            name='ctc_loss',
            )

        return inference_ret, loss
