import tensorflow as tf
import argparse
from models import utils
import os
from models.textData import TextData
from tqdm import tqdm
import pickle as p
from sklearn.metrics import f1_score, precision_recall_fscore_support
from models.model_gumbel import ModelGumbel
import numpy as np
import matplotlib.pyplot as plt

class Train:
	def __init__(self):
		self.args = None

		self.textData = None
		self.model = None
		self.outFile = None
		self.sess = None
		self.saver = None
		self.model_name = None
		self.model_path = None
		self.globalStep = 0
		self.summaryDir = None
		self.testOutFile = None
		self.summaryWriter = None
		self.mergedSummary = None


	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		parser.add_argument('--resultDir', type=str, default='result', help='result directory')
		parser.add_argument('--testDir', type=str, default='test_result')
		# data location
		dataArgs = parser.add_argument_group('Dataset options')

		dataArgs.add_argument('--summaryDir', type=str, default='summaries')
		dataArgs.add_argument('--datasetName', type=str, default='dataset', help='a TextData object')

		dataArgs.add_argument('--dataDir', type=str, default='data', help='dataset directory, save pkl here')
		dataArgs.add_argument('--dataset', type=str, default='rotten')
		dataArgs.add_argument('--trainFile', type=str, default='train.txt')
		dataArgs.add_argument('--valFile', type=str, default='val.txt')
		dataArgs.add_argument('--testFile', type=str, default='test.txt')
		dataArgs.add_argument('--embeddingFile', type=str, default='glove.840B.300d.txt')
		dataArgs.add_argument('--vocabSize', type=int, default=-1, help='vocab size, use the most frequent words')


		# neural network options
		nnArgs = parser.add_argument_group('Network options')
		nnArgs.add_argument('--embeddingSize', type=int, default=300)
		nnArgs.add_argument('--hiddenSize', type=int, default=300)
		nnArgs.add_argument('--rnnLayers', type=int, default=1)
		nnArgs.add_argument('--maxSteps', type=int, default=30)
		nnArgs.add_argument('--nClasses', type=int, default=2)
		nnArgs.add_argument('--dependent', action='store_true', help='two kinds of rationales, only independent is supported at the moment')
		nnArgs.add_argument('--rUnit', type=str, default='lstm', choices=['lstm', 'rcnn'], help='only support lstm at the moment')

		# training options
		trainingArgs = parser.add_argument_group('Training options')
		trainingArgs.add_argument('--rl', action='store_true', help='whether or not to use REINFORCE algorithm')
		trainingArgs.add_argument('--eager', action='store_true', help='turn on eager mode for debugging')
		trainingArgs.add_argument('--modelPath', type=str, default='saved')
		trainingArgs.add_argument('--preEmbedding', action='store_true')
		trainingArgs.add_argument('--elmo', action='store_true')
		trainingArgs.add_argument('--trainElmo', action='store_true')
		trainingArgs.add_argument('--dropOut', type=float, default=1.0, help='dropout rate for RNN (keep prob)')
		trainingArgs.add_argument('--learningRate', type=float, default=0.001, help='learning rate')
		trainingArgs.add_argument('--batchSize', type=int, default=100, help='batch size')
		trainingArgs.add_argument('--epochs', type=int, default=200, help='most training epochs')
		trainingArgs.add_argument('--device', type=str, default='/gpu:0', help='use the first GPU as default')
		trainingArgs.add_argument('--loadModel', action='store_true', help='whether or not to use old models')
		trainingArgs.add_argument('--theta', type=float, default=0.1, help='for #choices')
		trainingArgs.add_argument('--gamma', type=float, default=0.1, help='for continuity')
		trainingArgs.add_argument('--temperature', type=float, default=0.5, help='gumbel softmax temperature')
		trainingArgs.add_argument('--threshold', type=float, default=0.5, help='threshold for producing hard mask')

		return parser.parse_args(args)



	def main(self, args=None):
		print('TensorFlow version {}'.format(tf.VERSION))

		# initialize args
		self.args = self.parse_args(args)

		self.resultDir = os.path.join(self.args.resultDir, self.args.dataset)
		self.summaryDir = os.path.join(self.args.summaryDir, self.args.dataset)
		self.dataDir = os.path.join(self.args.dataDir, self.args.dataset)
		self.testDir = os.path.join(self.args.testDir, self.args.dataset)

		self.outFile = utils.constructFileName(self.args, prefix=self.resultDir)
		self.testFile = utils.constructFileName(self.args, prefix=self.testDir)

		self.args.datasetName = utils.constructFileName(self.args, prefix=self.args.dataset, createDataSetName=True)
		datasetFileName = os.path.join(self.dataDir, self.args.datasetName)

		if not os.path.exists(self.resultDir):
			os.makedirs(self.resultDir)

		if not os.path.exists(self.testDir):
			os.makedirs(self.testDir)

		if not os.path.exists(self.args.modelPath):
			os.makedirs(self.args.modelPath)

		if not os.path.exists(self.summaryDir):
			os.makedirs(self.summaryDir)

		if not os.path.exists(datasetFileName):
			self.textData = TextData(self.args)
			with open(datasetFileName, 'wb') as datasetFile:
				p.dump(self.textData, datasetFile)
			print('dataset created and saved to {}, exiting ...'.format(datasetFileName))
			exit(0)
		else:
			with open(datasetFileName, 'rb') as datasetFile:
				self.textData = p.load(datasetFile)
			print('dataset loaded from {}'.format(datasetFileName))

		# self.statistics()

		sessConfig = tf.ConfigProto(allow_soft_placement=True)
		sessConfig.gpu_options.allow_growth = True

		self.model_path = os.path.join(self.args.modelPath, self.args.dataset)
		self.model_path = utils.constructFileName(self.args, prefix=self.model_path)
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		self.model_name = os.path.join(self.model_path, 'model')

		self.sess = tf.Session(config=sessConfig)
		# summary writer
		self.summaryDir = utils.constructFileName(self.args, prefix=self.summaryDir)
		if self.args.eager:
			tf.enable_eager_execution(config=sessConfig, device_policy=tf.contrib.eager.DEVICE_PLACEMENT_WARN)
			print('eager execution enabled')

		with tf.device(self.args.device):
			if self.args.rl:
				raise NotImplementedError
			else:
				self.model = ModelGumbel(self.args, self.textData, eager=self.args.eager)
				print('Gumbel model created!')

			if self.args.eager:
				self.train_eager()
				exit(0)
			# saver can only be created after we have the model
			self.saver = tf.train.Saver()

			self.summaryWriter = tf.summary.FileWriter(self.summaryDir, self.sess.graph)
			self.mergedSummary = tf.summary.merge_all()

			if self.args.loadModel:
				# load model from disk
				if not os.path.exists(self.model_path):
					print('model does not exist on disk!')
					print(self.model_path)
					exit(-1)

				self.saver.restore(sess=self.sess, save_path=self.model_name)
				print('Variables loaded from disk {}'.format(self.model_name))
			else:
				init = tf.global_variables_initializer()
				# initialize all global variables
				self.sess.run(init)
				print('All variables initialized')

			self.train(self.sess)

	def train_eager(self):
		"""
		for debugging
		:return:
		"""
		for e in range(self.args.epochs):
			trainBatches = self.textData.train_batches

			for idx, nextBatch in enumerate(tqdm(trainBatches)):
				self.model.step(nextBatch, test=False, eager=self.args.eager)
				self.model.buildNetwork()

				print()

	def train(self, sess):
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

		print('Start training')

		out = open(self.outFile, 'w', 1)
		out.write(self.outFile + '\n')
		utils.writeInfo(out, self.args)

		current_valAcc = 0.0
		for e in range(self.args.epochs):
			# training
			#trainBatches = self.textData.get_batches(tag='train')
			trainBatches = self.textData.train_batches
			totalTrainLoss = 0.0

			# cnt of batches
			cnt = 0

			total_samples = 0
			total_corrects = 0

			all_predictions = []
			all_labels = []


			for idx, nextBatch in enumerate(tqdm(trainBatches)):

				cnt += 1
				self.globalStep += 1
				total_samples += nextBatch.batch_size
				# print(idx)

				ops, feed_dict, labels = self.model.step(nextBatch, test=False)
				_, loss, predictions, corrects = sess.run(ops, feed_dict)
				all_predictions.extend(predictions)
				all_labels.extend(labels)
				total_corrects += corrects
				totalTrainLoss += loss

				self.summaryWriter.add_summary(utils.makeSummary({"train_loss": loss}), self.globalStep)

			trainAcc = total_corrects * 1.0 / total_samples

			print('\nepoch = {}, Train, loss = {}, acc = {}'.
				  format(e, totalTrainLoss, trainAcc))

			out.write('\nepoch = {}, loss = {}, acc = {}\n'.
					  format(e, totalTrainLoss, trainAcc))

			out.flush()

			# calculate f1 score for val (weighted/unweighted)
			valAcc, valLoss = self.test(sess, tag='val')
			testAcc, testLoss = self.test(sess, tag='test')


			print('\tVal, loss = {}, acc = {}'.format(valLoss, valAcc))
			out.write('\tVal, loss = {}, acc = {}'.format(valLoss, valAcc))

			print('\tTest, loss = {}, acc = {}'.format(testLoss, valAcc))
			out.write('\tTest, loss = {}, acc = {}'.format(testLoss, valAcc))

			out.flush()

			self.summaryWriter.add_summary(utils.makeSummary({"train_acc": trainAcc}), e)
			self.summaryWriter.add_summary(utils.makeSummary({"val_acc": valAcc}), e)
			self.summaryWriter.add_summary(utils.makeSummary({'test_acc':testAcc}), e)

			self.summaryWriter.add_summary(utils.makeSummary({"val_loss": valLoss}), e)
			self.summaryWriter.add_summary(utils.makeSummary({'test_loss':testLoss}), e)

			if valAcc > current_valAcc:
				current_valAcc = valAcc
				print('New valAcc {} at epoch {}'.format(valAcc, e))
				out.write('New valAcc {} at epoch {}\n'.format(valAcc, e))
				save_path = self.saver.save(sess, save_path=self.model_name)
				print('model saved at {}'.format(save_path))
				out.write('model saved at {}\n'.format(save_path))
			out.flush()
		out.close()


	def write_predictions(self, predictions, tag='weighted'):
		test_file = self.testFile + '_' + tag
		with open(test_file, 'w') as file:
			file.write('id\tturn1\tturn2\tturn3\tlabel\n')
			idx2label = {v: k for k, v in self.textData.label2idx.items()}
			for idx, sample in enumerate(self.textData.test_samples):
				assert idx == sample.id

				file.write(str(idx)+'\t')
				for ind, sent in enumerate(sample.sents):
					file.write(' '.join(sent[:sample.length[ind]]).encode('ascii', 'ignore').decode('ascii')+'\t')
				file.write(idx2label[predictions[idx]]+'\n')
		return test_file

	def test(self, sess, tag='val'):
		if tag == 'val':
			print('Validating\n')
			batches = self.textData.val_batches
		else:
			print('Testing\n')
			batches = self.textData.test_batches

		cnt = 0

		total_samples = 0
		total_corrects = 0
		total_loss = 0.0
		all_predictions = []
		all_labels = []
		all_sample_weights = []
		for idx, nextBatch in enumerate(tqdm(batches)):
			cnt += 1

			total_samples += nextBatch.batch_size
			ops, feed_dict, labels, sample_weights = self.model.step(nextBatch, test=True)

			loss, predictions, corrects = sess.run(ops, feed_dict)
			all_predictions.extend(predictions)
			all_labels.extend(labels)
			all_sample_weights.extend(sample_weights)
			total_loss += loss
			total_corrects += corrects

		acc = total_corrects * 1.0 / total_samples

		if tag == 'test':
			return all_predictions
		else:
			return acc, total_loss
