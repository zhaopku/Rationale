from tqdm import tqdm
import os
import pickle as p
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

n_clusters = [2, 3, 4, 5, 6, 7, 8, 10]

def read_embeddings(rep_words, dem_words):
	rep_words = {k:1 for k in rep_words}
	dem_words = {k:1 for k in dem_words}

	rep_embeds = {}
	dem_embeds = {}

	with open('../glove.840B.300d.txt', 'r') as glove:
		lines = glove.readlines()
		for line in tqdm(lines, desc='glove'):
			splits = line.split()
			word = splits[0]
			if len(splits) > 301:
				word = ''.join(splits[0:len(splits) - 300])
				splits[1:] = splits[len(splits) - 300:]
			embed = [float(s) for s in splits[1:]]

			if word.lower() in rep_words.keys():
				rep_embeds[word.lower()] = embed
			if word.lower() in dem_words.keys():
				dem_embeds[word.lower()] = embed

	return rep_embeds, dem_embeds

def read_words():
	in_dir = '/Users/mengzhao/all_rationale_words'
	file_names = os.listdir(in_dir)

	rep_words = []
	dem_words = []

	for file_name in tqdm(file_names):
		with open(os.path.join(in_dir, file_name), 'r') as file:
			lines = file.readlines()
			if lines[0].find('Republican') != -1:
				label = 'rep'
			else:
				label = 'dem'

			flag = False
			for line in lines:
				if line.find('below are rationale words') != -1:
					flag = True
					continue
				if flag:

					if len(line.strip()) == 0:
						continue
					if line.startswith('reads % ='):
						break

					words = line.strip().split('\t')
					words = [w.lower() for w in words]

					if label == 'rep':
						rep_words.extend(words)
					else:
						dem_words.extend(words)
	rep_words = list(set(rep_words))
	dem_words = list(set(dem_words))

	print('rep_words = {}'.format(len(rep_words)))
	print('dem_words = {}'.format(len(dem_words)))

	return rep_words, dem_words


def save(words, labels, tag='rep', n=5, n_rep=None):
	out_dir = 'results/cluster_results_' + str(n)
	if not os.path.exists(os.path.join(out_dir, tag)):
		os.makedirs(os.path.join(out_dir, tag))

	cluster_to_words = defaultdict(list)

	assert len(words) == len(labels)

	for idx in range(len(words)):
		if tag != 'combined':
			cluster_to_words[labels[idx]].append(words[idx])
		else:
			if idx < n_rep:
				cluster_to_words[labels[idx]].append(words[idx]+'\t#\t'+'rep')
			else:
				cluster_to_words[labels[idx]].append(words[idx]+'\t#\t'+'dem')

	for cluster_id, words in cluster_to_words.items():
		with open(os.path.join(out_dir, tag, str(cluster_id)+'.txt'), 'w') as file:
			for word in words:
				file.write(word+'\n')


def analyze_separate(rep_embeds_values, dem_embeds_values, rep_embeds_keys, dem_embeds_keys):

	for n in n_clusters:

		rep_kmeans = KMeans(n_clusters=n, n_init=8, n_jobs=-1).fit(rep_embeds_values)
		dem_kmeans = KMeans(n_clusters=n, n_init=8, n_jobs=-1).fit(dem_embeds_values)

		print('saving separated {} clusters'.format(n))

		save(rep_embeds_keys, rep_kmeans.labels_, tag='rep', n=n)
		save(dem_embeds_keys, dem_kmeans.labels_, tag='dem', n=n)

		assert len(rep_kmeans.labels_) == len(rep_embeds.items())
		assert len(dem_kmeans.labels_) == len(dem_embeds.items())


def analyze_combine(rep_embeds_values, dem_embeds_values, rep_embeds_keys, dem_embeds_keys):

	for n in n_clusters:

		kmeans = KMeans(n_clusters=n, n_init=8, n_jobs=-1).fit(rep_embeds_values+dem_embeds_values)

		print('saving combined {} clusters'.format(n))

		save(rep_embeds_keys+dem_embeds_keys, kmeans.labels_, tag='combined', n=n, n_rep=len(rep_embeds_keys))

		assert len(kmeans.labels_) == len(rep_embeds.items()) + len(dem_embeds.items())

if __name__ == '__main__':

	glove_embed_file = 'glove.pkl'

	if os.path.exists(glove_embed_file):
		with open(glove_embed_file, 'rb') as file:
			rep_embeds, dem_embeds = p.load(file)
			print('glove loaded from {}'.format(glove_embed_file))

	else:
		rep_words, dem_words = read_words()
		glove_embed = read_embeddings(rep_words, dem_words)
		with open(glove_embed_file, 'wb') as file:
			p.dump(glove_embed, file)
			print('glove saved to {}'.format(glove_embed_file))
			exit(0)

	print('# rep_embeds = {}'.format(len(rep_embeds.items())))
	print('# dem_embeds = {}'.format(len(dem_embeds.items())))

	rep_embeds_keys = []
	rep_embeds_values = []
	for k, v in rep_embeds.items():
		rep_embeds_keys.append(k)
		rep_embeds_values.append(v)

	dem_embeds_keys = []
	dem_embeds_values = []
	for k, v in dem_embeds.items():
		dem_embeds_keys.append(k)
		dem_embeds_values.append(v)

	analyze_separate(rep_embeds_keys=rep_embeds_keys, rep_embeds_values=rep_embeds_values,
	                 dem_embeds_keys=dem_embeds_keys, dem_embeds_values=dem_embeds_values)

	analyze_combine(rep_embeds_keys=rep_embeds_keys, rep_embeds_values=rep_embeds_values,
	                 dem_embeds_keys=dem_embeds_keys, dem_embeds_values=dem_embeds_values)
