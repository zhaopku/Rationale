import tensorflow as tf
import collections
import numpy as np

_GatedSkimLSTMStateTuple = collections.namedtuple("GatedSkimLSTMStateTuple", ("c", "h", "s"))

class GatedSkimLSTMStateTuple(_GatedSkimLSTMStateTuple):
	__slots__ = ()

	@property
	def dtype(self):
		(c, h, s) = self
		if c.dtype != h.dtype:
			raise TypeError("Inconsistent internal state: %s vs %s" %
			                (str(c.dtype), str(h.dtype)))
		if c.dtype != s.dtype:
			raise TypeError("Inconsistent internal state: %s vs %s" %
			                (str(c.dtype), str(h.dtype)))

		return c.dtype


class GatedSkimLSTMCell(tf.contrib.rnn.LayerRNNCell):
	def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None, name=None, hard_gate=False,
	           threshold=0.5, initializer=None, next_=False):
		super(GatedSkimLSTMCell, self).__init__(_reuse=reuse, name=name)
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation or tf.tanh
		self.hard_gate = hard_gate
		self.threshold = threshold
		self.initializer = initializer
		self.next_ = next_

	@property
	def state_size(self):
		return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
		        if self._state_is_tuple else 2 * self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def build(self, inputs_shape):
		"""
		build variables for LSTM cell
		:param inputs_shape: [batch_size, input_depth]
		:return:
		"""
		if inputs_shape[1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
			                 % inputs_shape)

		input_depth = inputs_shape[1].value
		h_depth = self._num_units

		self._bias = tf.get_variable(name='bias', shape=[4 * self._num_units],
		                             initializer=tf.zeros_initializer(dtype=self.dtype))
		self._kernel = tf.get_variable(name='kernel', shape=[input_depth+h_depth, 4 * self._num_units], initializer=self.initializer)

		self.built = True

	def call(self, inputs, state):
		"""
		:param inputs: `2-D` tensor with shape `[batch_size, input_size]`.
		:param state: An `LSTMStateTuple` of state tensors, each shaped
	        `[batch_size, self.state_size]`, if `state_is_tuple` has been set to
	        `True`.  Otherwise, a `Tensor` shaped
	        `[batch_size, 2 * self.state_size]`.
		:return:
		"""
		if self._state_is_tuple:
			c, h, read_prob = state
		else:
			c, h, read_prob = tf.split(value=state, num_or_size_splits=2, axis=1)

		gate_inputs = tf.matmul(tf.concat([inputs, h], 1), self._kernel)
		gate_inputs = tf.add(gate_inputs, self._bias)

		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)
		self._forget_bias = tf.cast(self._forget_bias, f.dtype)

		# [batch_size, hidden_size]
		new_c = tf.add(tf.multiply(c, tf.nn.sigmoid(tf.add(f, self._forget_bias))),
		               tf.multiply(tf.nn.sigmoid(i), self._activation(j)))

		new_h = tf.multiply(self._activation(new_c), tf.nn.sigmoid(o))

		read_prob = tf.expand_dims(read_prob, axis=-1)
		read_c = tf.add(tf.multiply(read_prob, new_c), tf.multiply((1 - read_prob), c), name='skim_c')
		read_h = tf.add(tf.multiply(read_prob, new_h), tf.multiply((1 - read_prob), h), name='skim_h')

		if self._state_is_tuple:
			new_state = GatedSkimLSTMStateTuple(read_c, read_h, read_prob)
		else:
			new_state = tf.concat([read_c, read_h, read_prob], 1)

		return read_h, new_state
