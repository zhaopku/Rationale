import os
from collections import defaultdict, Counter
import numpy as np
import random
from tqdm import tqdm
from models.congress_utils import Sample
from models.data_utils import Batch
import nltk
import csv


class CourtData:
	def __init__(self, args):
		self.args = args

		self.meta_file_name = 'data/court/data.csv'

		#note: use 20k most frequent words
		self.UNK_WORD = 'unk'
		self.PAD_WORD = '<pad>'
		self.BLANK = '<blank>'
		self.NEWLINE = '<newline>'

		# list of batches
		self.train_batches = []
		self.val_batches = []
		self.test_batches = []

		self.word2id = {}
		self.id2word = {}

		self.train_samples = None
		self.valid_samples = None
		self.test_samples = None
		self.preTrainedEmbedding = None

		self.train_samples, self.valid_samples, self.test_samples = self._create_data()

		# [num_batch, batch_size, maxStep]
		self.train_batches = self._create_batch(self.train_samples)
		self.val_batches = self._create_batch(self.valid_samples)

		# note: test_batches is none here
		self.test_batches = self._create_batch(self.test_samples)

		print('Dataset created')

	def getVocabularySize(self):
		assert len(self.word2id) == len(self.id2word)
		return len(self.word2id)

	def _create_batch(self, all_samples, tag='test'):
		all_batches = []
		if tag == 'train':
			random.shuffle(all_samples)
		if all_samples is None:
			return all_batches

		num_batch = len(all_samples)//self.args.batchSize + 1
		for i in range(num_batch):
			samples = all_samples[i*self.args.batchSize:(i+1)*self.args.batchSize]

			if len(samples) == 0:
				continue

			batch = Batch(samples)
			all_batches.append(batch)

		return all_batches

	def divide(self, words):

		n_groups = len(words) // self.args.maxSteps + 1
		group_words = []

		for i in range(n_groups):
			group_words.append(words[self.args.maxSteps*i:self.args.maxSteps*(i+1)])

		return group_words


	def _create_samples(self, file_path, tag=1):
		path = os.path.join(file_path, str(tag))
		file_names = os.listdir(path)
		original_lengths = []
		oov_cnt = 0
		cnt = 0
		all_samples = []
		for idx, file_name in enumerate(tqdm(file_names, desc=file_path)):
			# if idx == 50:
			# 	break
			with open(os.path.join(path, file_name), 'r') as file:
				lines = file.readlines()
				texts = ''

				for line in lines:
					if len(line.strip()) <= 1:
						continue
					texts += line.strip()
				all_words = nltk.word_tokenize(texts)

				group_words = self.divide(all_words)

				for sub_id, words in enumerate(group_words):
					word_ids = []
					original_lengths.append(len(words))

					words = words[:self.args.maxSteps]
					length = len(words)
					if length == 0:
						continue
					cnt += length
					for word in words:
						if word in self.word2id.keys():
							id_ = self.word2id[word]
						else:
							id_ = self.word2id[self.UNK_WORD]
							# print('Check!')
						if id_ == self.word2id[self.UNK_WORD] and word != self.UNK_WORD:
							oov_cnt += 1
						word_ids.append(id_)
					while len(word_ids) < self.args.maxSteps:
						word_ids.append(self.word2id[self.PAD_WORD])
					while len(words) < self.args.maxSteps:
						words.append(self.PAD_WORD)

					sample = Sample(word_ids=word_ids, words=words, label=tag-1, length=length, id=file_name+str(sub_id))
					all_samples.append(sample)

		print(len(original_lengths))
		print(np.average(original_lengths))
		print('{} of samples within {} words'.format(np.sum(np.asarray(original_lengths) < self.args.maxSteps)/len(original_lengths), self.args.maxSteps))

		return all_samples, oov_cnt, cnt

	def create_embeddings(self):
		words = self.word2id.keys()

		glove_embed = {}

		with open('vectors.txt', 'r') as glove:
			lines = glove.readlines()
			for line in tqdm(lines, desc='glove'):
				splits = line.split()
				word = splits[0]
				if len(splits) > 301:
					word = ''.join(splits[0:len(splits) - 300])
					splits[1:] = splits[len(splits) - 300:]
				if word not in words:
					continue
				embed = [float(s) for s in splits[1:]]
				glove_embed[word] = embed

		embeds = []
		for word_id in range(len(self.id2word)):
			word = self.id2word[word_id]
			if word in glove_embed.keys():
				embed = glove_embed[word]
			else:
				embed = glove_embed[self.UNK_WORD]
				self.word2id[word] = self.word2id[self.UNK_WORD]
			embeds.append(embed)

		embeds = np.asarray(embeds)

		return embeds

	def _create_data(self):

		train_path = '/Users/mengzhao/court_data/train'
		val_path = '/Users/mengzhao/court_data/val'
		test_path = '/Users/mengzhao/court_data/test'

		print('Building vocabularies for {} dataset'.format(self.args.dataset))
		self.word2id, self.id2word = self._build_vocab(train_path, val_path, test_path)

		if self.args.dataset != 'ag':
			print('Creating pretrained embeddings!')
			# self.preTrainedEmbedding = self.create_embeddings()

		print('Building training samples!')
		train_samples1, train_oov1, train_cnt1 = self._create_samples(train_path, tag=1)
		val_samples1, val_oov1, val_cnt1 = self._create_samples(val_path, tag=1)
		test_samples1, test_oov1, test_cnt1 = self._create_samples(test_path, tag=1)

		train_samples2, train_oov2, train_cnt2 = self._create_samples(train_path, tag=2)
		val_samples2, val_oov2, val_cnt2 = self._create_samples(val_path, tag=2)
		test_samples2, test_oov2, test_cnt2 = self._create_samples(test_path, tag=2)

		train_samples = train_samples1 + train_samples2
		val_samples = val_samples1 + val_samples2
		test_samples = test_samples1 + test_samples2

		train_oov = train_oov1 + train_oov2
		val_oov = val_oov1 + val_oov2
		test_oov = test_oov1 + test_oov2

		train_cnt = train_cnt1 + train_cnt2
		val_cnt = val_cnt1 + val_cnt2
		test_cnt = test_cnt1 + test_cnt2


		print('OOV rate for train = {:.2%}'.format(train_oov*1.0/train_cnt))
		print('OOV rate for val = {:.2%}'.format(val_oov*1.0/val_cnt))
		print('OOV rate for test = {:.2%}'.format(test_oov*1.0/test_cnt))

		return train_samples, val_samples, test_samples

	@staticmethod
	def _read_sents(path):
		filenames = os.listdir(path)
		all_words = []
		for idx, filename in enumerate(tqdm(filenames, desc=path)):
			# if idx == 50:
			# 	break
			with open(os.path.join(path, filename), 'r') as file:
				lines = file.readlines()
				texts = ''

				for line in lines:
					if len(line.strip()) <= 1:
						continue
					texts += line.strip()
				words = nltk.word_tokenize(texts)
			all_words.extend(words)
		return all_words

	def _build_vocab(self, train_path, val_path, test_path):
		all_train_words = self._read_sents(os.path.join(train_path, '1'))
		all_val_words = self._read_sents(os.path.join(val_path, '1'))
		all_test_words = self._read_sents(os.path.join(test_path, '1'))

		all_train_words2 = self._read_sents(os.path.join(train_path, '2'))
		all_val_words2 = self._read_sents(os.path.join(val_path, '2'))
		all_test_words2 = self._read_sents(os.path.join(test_path, '2'))

		all_words = all_train_words + all_val_words + all_test_words
		all_words += all_train_words2 + all_val_words2 + all_test_words2

		print(len(list(set(all_words))))

		counter = Counter(all_words)

		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

		# keep the most frequent vocabSize words, including the special tokens
		# -1 means we have no limits on the number of words
		if self.args.vocabSize != -1:
			count_pairs = count_pairs[0:self.args.vocabSize-2]

		count_pairs.append((self.UNK_WORD, 100000))
		count_pairs.append((self.PAD_WORD, 100000))

		if self.args.vocabSize != -1:
			assert len(count_pairs) == self.args.vocabSize

		words, _ = list(zip(*count_pairs))
		word_to_id = dict(zip(words, range(len(words))))

		id_to_word = {v: k for k, v in word_to_id.items()}

		return word_to_id, id_to_word

	def get_batches(self, tag='train'):
		if tag == 'train':
			return self._create_batch(self.train_samples, tag='train')
		elif tag == 'val':
			return self.val_batches
		else:
			return self.test_batches
