import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import numpy as np
import tensorflow_hub as hub
import random
from models.encoder_cell import GatedSkimLSTMCell, GatedSkimLSTMStateTuple

ELMOSIZE = 1024

class ModelGumbel:
	def __init__(self, args, textData, initializer=None, eager=False):
		print('Creating single lstm Model')
		self.args = args
		self.textData = textData

		self.dropOutRate = None
		self.initial_state = None
		self.learning_rate = None
		self.loss = None
		self.optOp = None
		self.labels = None
		self.input = None
		self.target = None
		self.length = None
		self.embedded = None
		self.predictions = None
		self.batch_size = None
		self.corrects = None
		self.initializer = initializer
		self.eager = eager

		if self.args.elmo:
			self.embedding_size = ELMOSIZE
		else:
			self.embedding_size = self.args.embeddingSize

		self.mask = None
		self.v0 = None
		self.v1 = None
		self.v2 = None
		self.v3 = None
		self.v4 = None
		self.v5 = None
		self.v6 = None
		self.v7 = None

		if not eager:
			self.buildNetwork()

	def buildInputs(self):
		with tf.name_scope('placeholders'):
			# [batch_size, max_steps]
			input_shape = [None, self.args.maxSteps]

			if self.args.elmo:
				self.data = tf.placeholder(tf.string, shape=input_shape, name='data')
			else:
				self.data = tf.placeholder(tf.int32, shape=input_shape, name='data')
			# [batch_size]
			self.length = tf.placeholder(tf.int32, shape=[None], name='length')

			# [batch_size]
			self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')

			# scalar
			self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
			self.dropOutRate = tf.placeholder(tf.float32, shape=(), name='dropOut')
			self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

	def buildEmbeddings(self):
		with tf.name_scope('embedding_layer'):
			if not self.args.preEmbedding:
				print('Using randomly initialized embeddings!')
				embeddings = tf.get_variable(
					shape=[self.textData.getVocabularySize(), self.args.embeddingSize],
					initializer=tf.contrib.layers.xavier_initializer(),
					name='embeddings')
				# [batch_size, n_turn, max_steps, embedding_size]
				self.embedded = tf.nn.embedding_lookup(embeddings, self.data)
			elif not self.args.elmo:
				print('Using pretrained glove word embeddings!')
				embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding', dtype=tf.float32)
				# [batch_size, n_turn, max_steps, embedding_size]
				self.embedded = tf.nn.embedding_lookup(embeddings, self.data)
			else:
				# elmo not supported for eager execution

				elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.args.trainElmo)
				# [batch_size*n_turn, max_steps]
				data_elmo = tf.reshape(self.data, shape=[-1, self.args.maxSteps], name='data_elmo')
				# [batch_size*n_turn]
				length_elmo = tf.reshape(self.length, shape=[-1], name='length_elmo')
				# [batch_size*n_turn, elmo_size]
				self.embedded = elmo(
					inputs={
						"tokens": data_elmo,
						"sequence_len": length_elmo
					},
					signature="tokens",
					as_dict=True)['elmo']
				self.embedded = tf.reshape(self.embedded, shape=[self.batch_size, self.args.maxSteps, ELMOSIZE], name='elmo_embedded')
			# [batch_size, n_turn, max_steps, embedding_size]
			self.embedded = tf.nn.dropout(self.embedded, self.dropOutRate, name='embedding_dropout')

	def buildNetwork(self):
		with tf.name_scope('inputs'):
			if not self.eager:
				self.buildInputs()
			self.buildEmbeddings()

		with tf.name_scope('generator'):
			# [batch_size, max_steps]
			mask = self.generator()
			self.mask = mask

		with tf.name_scope('encoder'):
			outputs = self.encoder(mask)

		with tf.variable_scope('output'):
			weights = tf.get_variable(name='weights', shape=[self.args.hiddenSize, self.args.nClasses],
									  initializer=self.initializer)

			biases = tf.get_variable(name='biases', shape=[self.args.nClasses],
			                         initializer=self.initializer)
			# [batchSize, nClasses]
			logits = tf.nn.xw_plus_b(x=outputs, weights=weights, biases=biases)
		with tf.name_scope('predictions'):
			# [batchSize]
			self.predictions = tf.argmax(logits, axis=-1, name='predictions', output_type=tf.int32)
			# single number
			self.corrects = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32), name='corrects')

		with tf.name_scope('loss'):
			# [batch_size]
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='loss')

			# punish selections, [batch_size]
			valid_masks = tf.sequence_mask(self.length, maxlen=self.args.maxSteps, dtype=tf.float32)
			mask = mask * valid_masks
			self.true_mask = mask
			# [batch_size]
			mask_per_sample = tf.reduce_sum(mask, axis=-1) / tf.cast(self.length, tf.float32)
			self.mask_per_sample = mask_per_sample
			# discourages transitions

			# [batch_size, max_steps-1]
			mask_shift_right = tf.slice(mask, begin=[0, 0], size = [-1, self.args.maxSteps-1])
			pad = tf.expand_dims(mask[:, 0], -1)
			mask_shift_right = tf.concat([pad, mask_shift_right], axis=-1, name='mask_shift_right')

			transitions = tf.abs(mask - mask_shift_right)
			transitions = transitions * valid_masks

			transitions_per_sample = tf.reduce_sum(transitions, axis=-1) / tf.cast(self.length, tf.float32)

			additional_loss = self.args.theta * mask_per_sample + self.args.gamma * transitions_per_sample

			self.loss = tf.reduce_mean(loss+additional_loss)

		if self.args.eager:
			return

		with tf.name_scope('backpropagation'):
			trainable_params = tf.trainable_variables()

			m = tf.reduce_mean(self.args.theta * mask_per_sample)
			t = tf.reduce_mean(self.args.gamma * transitions_per_sample)
			gradients_m = tf.gradients(m, trainable_params)
			gradients_t = tf.gradients(t, trainable_params)
			#
			# opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
			#                              epsilon=1e-08)
			# self.optOp = opt.apply_gradients(zip(gradients, trainable_params))

			opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
											   epsilon=1e-08)
			self.optOp = opt.minimize(self.loss)

	def encoder(self, mask):
		"""

		:param mask: [batch_size, mask], the prob of the word being read
		:return:
		"""
		with tf.variable_scope('cell', reuse=False):

			def get_cell(hiddenSize, dropOutRate):
				print('building skim cell!')
				cell = GatedSkimLSTMCell(num_units=hiddenSize, state_is_tuple=True,
				                         initializer=self.initializer, threshold=self.args.threshold)
				cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
				                                     output_keep_prob=dropOutRate)
				return cell

			# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn

			cell = get_cell(self.args.hiddenSize, self.dropOutRate)


		c = tf.zeros([self.batch_size, self.args.hiddenSize], dtype=tf.float32)
		h = tf.zeros([self.batch_size, self.args.hiddenSize], dtype=tf.float32)
		# [batchSize, maxSteps, hiddenSize]
		state = (GatedSkimLSTMStateTuple(c, h, mask[:, 0]))

		outputs = []
		# TODO: remember the length of each sentence
		with tf.variable_scope("loop", reuse=tf.AUTO_REUSE):
			for time_step in range(self.args.maxSteps):
				# [batch_size, hidden_size]
				# [batch_size, 1]
				(cell_output, state) = cell(self.embedded[:, time_step, :], state)
				c, h, read_prob = state
				outputs.append(cell_output)
				if time_step == self.args.maxSteps - 1:
					break
				state = GatedSkimLSTMStateTuple(c, h, mask[:, time_step+1])

		# [maxSteps, batchSize, hiddenSize]
		outputs = tf.stack(outputs)
		# [batchSize, maxSteps, hiddenSize]
		outputs = tf.transpose(outputs, [1, 0, 2], name='outputs')

		# [batchSize, maxSteps]
		last_relevant_mask = tf.one_hot(indices=self.length - 1, depth=self.args.maxSteps, name='last_relevant',
		                                dtype=tf.int32)
		# [batchSize, hiddenSize]
		last_relevant_outputs = tf.boolean_mask(outputs, last_relevant_mask, name='last_relevant_outputs')

		return last_relevant_outputs

	@staticmethod
	def gumbel(logits, temperature):
		"""

		:param logits:
		:param temperature:
		:return:
		"""
		# g = -log(-log(u)), u ~ U(0, 1)
		noise = tf.random.uniform(shape=tf.shape(logits), name='noise')

		noise = tf.add(noise, 1e-9)
		noise = -tf.log(noise)

		noise = tf.add(noise, 1e-9)
		noise = -tf.log(noise)

		logits_new = (logits + noise) / temperature

		probs = tf.nn.softmax(logits=logits_new, axis=-1, name='probs')

		return probs

	def sample(self, probs):
		hard_mask = tf.cast(tf.greater(probs, self.args.threshold), tf.float32, name='hard_mask')

		# x if true
		mask_final = tf.where(condition=self.is_training, x=probs, y=hard_mask)

		return mask_final

	def generator(self):
		with tf.name_scope('cell'):
			def get_cell(hiddenSize, dropOutRate):
				print('building ordinary cell!')
				cell = BasicLSTMCell(num_units=hiddenSize, state_is_tuple=True)
				cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
				                                     output_keep_prob=dropOutRate)
				return cell

			# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn
			multiCell = []
			for i in range(self.args.rnnLayers):
				multiCell.append(get_cell(self.args.hiddenSize, self.dropOutRate))
			multiCell = tf.contrib.rnn.MultiRNNCell(multiCell, state_is_tuple=True)

		with tf.name_scope('get_rnn_outputs'):
			# outputs: [batch_size, max_steps, hidden_size]
			outputs, states = tf.nn.dynamic_rnn(cell=multiCell,
		                                   inputs=self.embedded, sequence_length=self.length,
		                                   dtype=tf.float32)
		with tf.name_scope('hidden'):
			weights = tf.get_variable(name='weights', shape=[self.args.hiddenSize, 2], dtype=tf.float32)
			biases = tf.get_variable(name='biases', shape=[self.args.nClasses], dtype=tf.float32)

			# logits: [batch_size*max_steps, 2]
			logits = tf.nn.xw_plus_b(x=tf.reshape(outputs, [-1, self.args.hiddenSize]),
			                         weights = weights, biases = biases, name='logits')
			# probs: [batch_size*max_steps, 2]
			probs = self.gumbel(logits=logits, temperature=self.args.temperature)

			# probs_selected: [batch_size*max_steps, 1]
			probs_selected = tf.slice(probs, begin=[0, 1], size=[-1, 1], name='probs_selected')
			# probs_selected: [batch_size, max_steps]
			probs_selected = tf.reshape(probs_selected, [self.batch_size, self.args.maxSteps])

			mask = self.sample(probs_selected)

			return mask


	def step_eager(self, data, length, labels, test):
		"""
		not supported for training, currently only debugging
		:param data:
		:param length:
		:param labels:
		:param test:
		:return:
		"""

		self.labels = labels
		self.data = data
		self.length = length
		self.batch_size = len(labels)
		self.is_training = not test

		if not test:
			self.dropOutRate = self.args.dropOut
		else:
			self.dropOutRate = 1.0

	def step_graph(self, data, length, labels, test):
		"""
		:param data:
		:param length:
		:param labels:
		:param test:
		:return:
		"""
		feed_dict = dict()

		feed_dict[self.labels] = labels
		feed_dict[self.data] = np.asarray(data)
		feed_dict[self.length] = np.asarray(length)
		feed_dict[self.batch_size] = len(labels)
		feed_dict[self.is_training] = not test

		if not test:
			feed_dict[self.dropOutRate] = self.args.dropOut
			ops = (self.optOp, self.loss, self.predictions, self.corrects, self.mask_per_sample, self.true_mask)
		else:
			# during test, do not use drop out!!!!
			feed_dict[self.dropOutRate] = 1.0
			ops = (self.loss, self.predictions, self.corrects, self.mask_per_sample, self.true_mask)

		return ops, feed_dict, labels

	def step(self, batch, test=False, eager=False):

		# [batch_size, max_steps]
		data = []
		# [batch_size]
		length = []
		# [batch_size]
		labels = []

		for sample in batch.samples:
			labels.append(sample.label)

			# train&test, not exceed maxSteps
			if self.args.elmo:
				data.append(sample.words[:self.args.maxSteps])
			else:
				data.append(sample.word_ids[:self.args.maxSteps])
			length.append(sample.length)


		data = np.asarray(data)
		length = np.asarray(length)
		labels = np.asarray(labels)

		if eager:
			return self.step_eager(data=data, length=length, labels=labels, test=test)
		else:
			return self.step_graph(data=data, length=length, labels=labels, test=test)
