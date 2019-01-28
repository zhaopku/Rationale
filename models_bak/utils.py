import os
import tensorflow as tf

def shape(x, dim):
	return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
	if len(inputs.get_shape()) > 3:
		raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

	if len(inputs.get_shape()) == 3:
		batch_size = shape(inputs, 0)
		seqlen = shape(inputs, 1)
		emb_size = shape(inputs, 2)
		current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
	else:
		current_inputs = inputs

	for i in range(num_hidden_layers):
		hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
		hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
		current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

	if dropout is not None:
		current_outputs = tf.nn.dropout(current_outputs, dropout)
	current_inputs = current_outputs

	output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
	output_bias = tf.get_variable("output_bias", [output_size])
	outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

	if len(inputs.get_shape()) == 3:
		outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
	return outputs


def makeSummary(value_dict):
	return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])


def constructFileName(args, prefix=None, createDataSetName=False):

	if createDataSetName:
		file_name = ''
		file_name += prefix + '-'
		file_name += str(args.vocabSize) + '-'
		file_name += str(args.batchSize) + '-'
		file_name += str(args.testBatchSize) + '-'
		file_name += str(args.maxLength) + '.pkl'
		#file_name += str(args.maxSteps) + '.pkl'
		return file_name

	file_name = ''

	file_name += 'hSize_' + str(args.rUnit) + '_' + str(args.hiddenSize)
	file_name += '_steps_' + str(args.maxSteps)
	file_name += '_dep_' + str(args.dependent)
	file_name += '_d_' + str(args.dropOut)

	file_name += '_lr_' + str(args.learningRate)
	file_name += '_bt_' + str(args.batchSize)
	file_name += '_vS_' + str(args.vocabSize)
	file_name += '_pre_' + str(args.preEmbedding)
	file_name += '_elmo_' + str(args.elmo)
	file_name += '_rnnL_' + str(args.rnnLayers)
	file_name += '_theta_' + str(args.theta)
	file_name += '_gamma_' + str(args.gamma)
	file_name += '_t_' + str(args.temperature)
	file_name = os.path.join(prefix, file_name)

	return file_name

def writeInfo(out, args):
	out.write('embeddingSize {}\n'.format(args.embeddingSize))
	out.write('hiddenSize {}\n'.format(args.hiddenSize))

	out.write('dataset {}\n'.format(args.dataset))

	out.write('maxSteps {}\n'.format(args.maxSteps))
	out.write('dropOut {}\n'.format(args.dropOut))

	out.write('learningRate {}\n'.format(args.learningRate))
	out.write('batchSize {}\n'.format(args.batchSize))
	out.write('epochs {}\n'.format(args.epochs))

	out.write('loadModel {}\n'.format(args.loadModel))

	out.write('vocabSize {}\n'.format(args.vocabSize))
	out.write('preEmbeddings {}\n'.format(args.preEmbedding))
	out.write('elmo {}\n'.format(args.elmo))
	out.write('trainElmo {}\n'.format(args.trainElmo))
	out.write('rnnLayers {}\n'.format(args.rnnLayers))

	out.write('theta {}\n'.format(args.theta))
	out.write('gamma {}\n'.format(args.gamma))
	out.write('dependent {}\n'.format(args.dependent))
	out.write('temperature {}\n'.format(args.temperature))
