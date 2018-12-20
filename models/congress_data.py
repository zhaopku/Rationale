import os
from collections import defaultdict, Counter
import numpy as np
import random
from tqdm import tqdm
from models.congress_utils import Sample
from models.data_utils import Batch
import nltk
import pickle as p

class CongressData:
	def __init__(self, args):
		self.args = args

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

		self.id2label = self.get_labels()

		self.train_samples, self.valid_samples, self.test_samples = self._create_data()

		# [num_batch, batch_size, maxStep]
		self.train_batches = self._create_batch(self.train_samples)
		self.val_batches = self._create_batch(self.valid_samples)

		# note: test_batches is none here
		self.test_batches = self._create_batch(self.test_samples)

		print('Dataset created')

	@staticmethod
	def get_labels():
		with open('./data/congress/speakerid2party.pkl', 'rb') as file:
			id2label = p.load(file)

		return id2label

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

	def _create_samples(self, file_path):

		oov_cnt = 0
		cnt = 0
		with open(file_path, 'r') as file:
			lines = file.readlines()
			all_samples = []
			for idx, line in enumerate(tqdm(lines)):
				# if idx == 100000:
				#     break
				line = line.strip()
				label = int(line[-1])
				line = line[0:-1].strip()

				words = nltk.word_tokenize(line)
				word_ids = []

				words = words[:self.args.maxSteps]
				length = len(words)
				cnt += length
				for word in words:
					if word in self.word2id.keys():
						id_ = self.word2id[word]
					else:
						id_ = self.word2id[self.UNK_WORD]
						print('Check!')
					if id_ == self.word2id[self.UNK_WORD] and word != self.UNK_WORD:
						oov_cnt += 1
					word_ids.append(id_)
				while len(word_ids) < self.args.maxSteps:
					word_ids.append(self.word2id[self.PAD_WORD])
				while len(words) < self.args.maxSteps:
					words.append(self.PAD_WORD)
				sample = Sample(data=word_ids, words=words,
								steps=self.args.maxSteps, label=label, length=length)
				all_samples.append(sample)

		return all_samples, oov_cnt, cnt

	def create_embeddings(self):
		words = self.word2id.keys()

		glove_embed = {}

		with open(self.args.embeddingFile, 'r') as glove:
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

	def process_single_file(self, texts):
		sample = Sample()

		words = nltk.word_tokenize(texts.strip())
		words = words[:self.args.maxSteps]
		sample.length = len(words)
		while len(words) < self.args.maxSteps:
			words.append(self.PAD_WORD)

		sample.words = words

		for word in words:
			if word in self.word2id.keys():
				sample.word_ids.append(self.word2id[word])
			else:
				self.word2id[word] = len(self.word2id.keys())
				sample.word_ids.append(self.word2id[word])

		return sample

	def _create_data(self):

		all_samples = []

		subdirs = os.listdir(self.args.congress_dir)[:1000]
		cnt_illegal_label = 0
		cnt_illegal_id = 0
		cnt_too_short = 0
		for subdir in tqdm(subdirs):
			if subdir.startswith('.'):
				continue
			subfiles = os.listdir(os.path.join(self.args.congress_dir, subdir))
			for subfile in subfiles:
				with open(os.path.join(self.args.congress_dir, subdir, subfile), 'r') as file:
					lines = file.readlines()
					texts = ' '.join(lines)
					sample = self.process_single_file(texts=texts)
					sample.id = subfile.strip()
					#sample.id = '_'.join(subfile.split('_')[:4]).strip('.txt')

					label = '_'
					for k, v in self.id2label.items():
						if sample.id.startswith(k.strip()):
							label = v
							break

					if label == '_':
						#print('illegal sample id {}'.format(sample.id))
						cnt_illegal_id += 1
						continue

					if label == 'Democrat':
						sample.label = 0
					elif label == 'Republican':
						sample.label = 1
					else:
						#print('illegal sample label {}'.format(label))
						cnt_illegal_label += 1
						continue

					if sample.length < 10:
						cnt_too_short += 1
						continue

					all_samples.append(sample)

		print('illegal label = {}'.format(cnt_illegal_label))
		print('illegal id = {}'.format(cnt_illegal_id))
		print('too short = {}'.format(cnt_too_short))



		n_samples = len(all_samples)
		n_train = int(n_samples*0.8)
		n_val = int((n_samples - n_train) / 2)

		random.shuffle(all_samples)

		train_samples = all_samples[0:n_train]
		val_samples = all_samples[n_train:n_train+n_val]
		test_samples = all_samples[n_train+n_val:]

		print('Totally {} samples'.format(len(all_samples)))
		print('# training samples = {}, val samples = {}, test samples = {}'.
		      format(len(train_samples), len(val_samples), len(test_samples)))

		self.word2id[self.UNK_WORD] = len(self.word2id)
		self.id2word = {v: k for k, v in self.word2id.items()}

		if self.args.dataset != 'ag':
			print('Creating pretrained embeddings!')
			self.preTrainedEmbedding = self.create_embeddings()

		return train_samples, val_samples, test_samples

	def get_batches(self, tag='train'):
		if tag == 'train':
			return self._create_batch(self.train_samples, tag='train')
		elif tag == 'val':
			return self.val_batches
		else:
			return self.test_batches

"""
illegal label = 530
illegal id = 50667
too short (10) = 19780
Totally 140653 samples
# training samples = 112522, val samples = 14065, test samples = 14066

"""