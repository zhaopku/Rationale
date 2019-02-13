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
import csv
from models.congress_data import CongressData
from models.CourtData import CourtData

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

		dataArgs.add_argument('--dataset', type=str, default='court')

		dataArgs.add_argument('--dataDir', type=str, default='data', help='dataset directory, save pkl here')
		dataArgs.add_argument('--vocabSize', type=int, default=100000, help='vocab size, use the most frequent words')

		# for congress dataset
		dataArgs.add_argument('--trainFile', type=str, default='train.txt')
		dataArgs.add_argument('--valFile', type=str, default='val.txt')
		dataArgs.add_argument('--testFile', type=str, default='test.txt')
		dataArgs.add_argument('--embeddingFile', type=str, default='glove.840B.300d.txt')
		dataArgs.add_argument('--congress_dir', type=str, default='/Users/mengzhao/congress_data/gpo/H')

		# for court dataset


		# neural network options
		nnArgs = parser.add_argument_group('Network options')
		nnArgs.add_argument('--embeddingSize', type=int, default=300)
		nnArgs.add_argument('--hiddenSize', type=int, default=300)
		nnArgs.add_argument('--rnnLayers', type=int, default=1)
		nnArgs.add_argument('--maxSteps', type=int, default=500)
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
		trainingArgs.add_argument('--batchSize', type=int, default=60, help='batch size')
		trainingArgs.add_argument('--epochs', type=int, default=200, help='most training epochs')
		trainingArgs.add_argument('--device', type=str, default='/gpu:0', help='use the first GPU as default')
		trainingArgs.add_argument('--loadModel', action='store_true', help='whether or not to use old models')
		trainingArgs.add_argument('--theta', type=float, default=0.1, help='for #choices')
		trainingArgs.add_argument('--gamma', type=float, default=0.1, help='for continuity')
		trainingArgs.add_argument('--temperature', type=float, default=0.5, help='gumbel softmax temperature')
		trainingArgs.add_argument('--threshold', type=float, default=0.5, help='threshold for producing hard mask')
		trainingArgs.add_argument('--testModel', action='store_true')

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
			if self.args.dataset == 'rotten':
				self.textData = TextData(self.args)
			elif self.args.dataset == 'congress':
				self.textData = CongressData(self.args)
			elif self.args.dataset == 'court':
				self.textData = CourtData(self.args)
			else:
				print('Cannot recognize {}'.format(self.args.dataset))
				raise NotImplementedError

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
				# init = tf.global_variables_initializer()
				# self.sess.run(init)
				print('Variables loaded from disk {}'.format(self.model_name))
				if self.args.testModel:
					self.test_model()
					exit(0)
			else:
				init = tf.global_variables_initializer()
				# initialize all global variables
				self.sess.run(init)
				print('All variables initialized')

			self.train(self.sess)

	def test_model_congress_words(self):
		acc, total_loss, avg_read, all_masks, all_predictions, all_samples = self.test(self.sess, tag='train', mode='test', n_batches=50)
		test_samples = all_samples
		print('acc = {}'.format(acc))
		out_dir = 'all_rationale_words'
		print('out_dir = {}'.format(out_dir))
		if not os.path.exists(out_dir):
			print('Creating Dir {}'.format(out_dir))
			os.makedirs(out_dir)

		for idx, sample in enumerate(test_samples):

			# skip if wrong
			if sample.label != all_predictions[idx]:
				continue

			with open(os.path.join(out_dir, sample.id), 'w') as file:
				masks = all_masks[idx]

				if sample.label == 0:
					file.write('This is from a Democrat\n')
				else:
					file.write('This is from a Republican\n')

				file.write('----- below is the original text -----\n')
				print(sample.id)
				for i, word in enumerate(sample.words[:sample.length]):
					file.write('{}({}) '.format(word, masks[i]))
					if i % 20 == 0 and i != 0:
						file.write('\n')

				file.write('\n\n\n')
				file.write('----- below are rationale words -----\n')
				cnt = 0
				for i, word in enumerate(sample.words[:sample.length]):
					if masks[i] == 1.0 and i != 0:
						file.write('{}\t'.format(word))
						cnt += 1
					if i % 20 == 0:
						file.write('\n')
				file.write('\n\nreads % = ')
				file.write(str(cnt*1.0/sample.length))


	def test_model_congress(self):
		acc, total_loss, avg_read, all_masks, all_predictions, all_samples = self.test(self.sess, tag='train', mode='test', n_batches=5)
		print('acc = {}'.format(acc))
		test_samples = all_samples

		out_dir = '~/rationale_studies'

		for idx, sample in enumerate(test_samples):
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			print(sample.id)
			with open(os.path.join(out_dir, sample.id), 'w') as file:
				masks = all_masks[idx]

				if sample.label == all_predictions[idx]:
					file.write('correct\n')
				else:
					file.write('wrong\n')

				for i , word in enumerate(sample.words[:sample.length]):
					file.write('{}({}) '.format(word, masks[i]))
					if i % 20 == 0:
						file.write('\n')



	def test_model(self):
		# init = tf.global_variables_initializer()
		# self.sess.run(init)
		acc, total_loss, avg_read, all_masks, all_predictions, all_samples = self.test(self.sess, tag='test', mode='test', n_batches=-1)
		test_samples = all_samples
		print(acc)
		exit(0)
		correct = 0
		with open('out.csv', 'w') as csvfile:
			writer = csv.writer(csvfile)
			all_reads = []
			for idx, sample in enumerate(test_samples):
				masks = all_masks[idx]

				row = sample.sentence[:sample.length]
				row.append(str(all_predictions[idx]))
				row.append(str(sample.label))
				if sample.label == all_predictions[idx]:
					correct += 1
					row.append('correct')
				else:
					row.append('wrong')
				writer.writerow(row)
				writer.writerow(masks[:sample.length])
				all_reads.extend(masks[:sample.length])

			csvfile.write('correct = {}\n'.format(correct/len(test_samples)))
			csvfile.write('reads = {}\n'.format(np.sum(all_reads)/len(all_reads)))


	def train_eager(self):
		"""
		for debugging
		:return:
		"""
		for e in range(self.args.epochs):
			trainBatches = self.textData.train_batches

			for idx, nextBatch in enumerate(tqdm(trainBatches)):
				self.model.step(nextBatch, test=True, eager=self.args.eager)
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
			trainBatches = self.textData.get_batches(tag='train')
			#trainBatches = self.textData.train_batches
			totalTrainLoss = 0.0

			# cnt of batches
			cnt = 0

			total_samples = 0
			total_corrects = 0

			all_predictions = []
			all_labels = []

			all_masks = []
			for idx, nextBatch in enumerate(tqdm(trainBatches)):

				cnt += 1
				self.globalStep += 1
				total_samples += nextBatch.batch_size
				# print(idx)

				ops, feed_dict, labels = self.model.step(nextBatch, test=False)
				_, loss, predictions, corrects, mask_per_sample, true_mask = sess.run(ops, feed_dict)
				all_masks.extend(mask_per_sample)
				all_predictions.extend(predictions)
				all_labels.extend(labels)
				total_corrects += corrects
				totalTrainLoss += loss

				self.summaryWriter.add_summary(utils.makeSummary({"train_loss": loss}), self.globalStep)
				#break
			trainAcc = total_corrects * 1.0 / total_samples
			train_avg_read = np.average(all_masks)
			print('\nepoch = {}, Train, loss = {}, acc = {}, avg_read = {}'.
				  format(e, totalTrainLoss, trainAcc, train_avg_read))

			out.write('\nepoch = {}, Train, loss = {}, acc = {}, avg_read = {}\n'.
					  format(e, totalTrainLoss, trainAcc, train_avg_read))

			out.flush()
			#continue
			# calculate f1 score for val (weighted/unweighted)
			valAcc, valLoss, val_avg_read = self.test(sess, tag='val', out=out)
			testAcc, testLoss, test_avg_read = self.test(sess, tag='test', out=out)

			out.flush()

			self.summaryWriter.add_summary(utils.makeSummary({"train_acc": trainAcc}), e)
			self.summaryWriter.add_summary(utils.makeSummary({"val_acc": valAcc}), e)
			self.summaryWriter.add_summary(utils.makeSummary({'test_acc':testAcc}), e)

			self.summaryWriter.add_summary(utils.makeSummary({"val_loss": valLoss}), e)
			self.summaryWriter.add_summary(utils.makeSummary({'test_loss':testLoss}), e)
			self.summaryWriter.add_summary(utils.makeSummary({'train_avg_read':train_avg_read}), e)
			self.summaryWriter.add_summary(utils.makeSummary({'val_avg_read':val_avg_read}), e)
			self.summaryWriter.add_summary(utils.makeSummary({'test_avg_read':test_avg_read}), e)

			if valAcc > current_valAcc:
				current_valAcc = valAcc
				print('New valAcc {} at epoch {}'.format(valAcc, e))
				out.write('New valAcc {} at epoch {}\n'.format(valAcc, e))
				save_path = self.saver.save(sess, save_path=self.model_name)
				print('model saved at {}'.format(save_path))
				out.write('model saved at {}\n'.format(save_path))
			out.flush()
		out.close()


	def write_predictions(self, predictions):
		# TODO
		#   add prediction code
		pass

	def test(self, sess, tag='val', out=None, mode=None, n_batches=2):
		if tag == 'val':
			print('Validating\n')
			batches = self.textData.val_batches
		elif tag == 'train':
			print('Testing on training data')
			batches = self.textData.train_batches
		else:
			print('Testing\n')
			batches = self.textData.test_batches

		if mode == 'test':
			if n_batches > 0:
				batches = batches[:n_batches]

		cnt = 0

		total_samples = 0
		total_corrects = 0
		total_loss = 0.0
		all_predictions = []
		all_labels = []
		all_masks = []
		all_true_masks = []
		all_samples = []
		for idx, nextBatch in enumerate(tqdm(batches)):
			cnt += 1

			total_samples += nextBatch.batch_size
			ops, feed_dict, labels = self.model.step(nextBatch, test=True)
			all_samples.extend(nextBatch.samples)

			loss, predictions, corrects, mask_per_sample, true_mask = sess.run(ops, feed_dict)
			all_masks.extend(mask_per_sample)
			all_true_masks.extend(true_mask)
			all_predictions.extend(predictions)
			all_labels.extend(labels)
			total_loss += loss
			total_corrects += corrects
			# if idx == 2:
			# 	break

		acc = total_corrects * 1.0 / total_samples
		avg_read = np.average(all_masks)

		if mode == 'test':
			return acc, total_loss, avg_read, all_true_masks, all_predictions, all_samples

		print('\t{}, loss = {}, acc = {}, avg_read = {}'.format(tag, total_loss, acc, avg_read))
		out.write('\t{}, loss = {}, acc = {}, avg_read = {}\n'.format(tag, total_loss, acc, avg_read))

		return acc, total_loss, avg_read
