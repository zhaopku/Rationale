import pickle as p
from tqdm import tqdm
import os
import argparse
from collections import defaultdict
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.style
import matplotlib



from matplotlib import interactive
interactive(True)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='combined')
parser.add_argument('--clusters', type=int, default=4)
parser.add_argument('--party', type=str, default='rep')
parser.add_argument('--sample', type=float, default=0.05, help='sample 5% of the data for plotting')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

def read_embeddings():
	if args.mode == 'combined':
		folder = os.path.join('results', 'cluster_results_'+str(args.clusters), args.mode)
	else:
		folder = os.path.join('results', 'cluster_results_'+str(args.clusters), args.party)

	cluster_to_words = defaultdict(list)
	cluster_to_embeddings = defaultdict(list)
	cluster_to_labels = defaultdict(list)
	for cluster_id in range(args.clusters):
		with open(os.path.join(folder, str(cluster_id)+'.txt'), 'r') as file:
			lines = file.readlines()
			for line in lines:
				if args.mode == 'combined':
					word = line.split('\t#\t')[0].strip()
					label = line.split('\t#\t')[1].strip()
				else:
					word = line.strip()
					label = args.party

				random_number = np.random.uniform()

				if random_number >= args.sample:
					continue

				cluster_to_words[cluster_id].append(word)
				cluster_to_labels[cluster_id].append(label)
				cluster_to_embeddings[cluster_id].append(glove_embed[word])

	return cluster_to_words, cluster_to_embeddings, folder, cluster_to_labels

def plot_results(reduced_embeddings, folder, cluster_to_length, cluster_to_labels):
	cluster_to_reduced = defaultdict(list)

	accumulate = 0
	for cluster_id, length in cluster_to_length.items():
		cluster_to_reduced[cluster_id] = reduced_embeddings[accumulate:accumulate+length, :]
		accumulate += length
	matplotlib.rcParams['figure.figsize'] = [16.0, 15.0]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	cmaps = [
		'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
		'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
		'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


	for cluster_id in range(args.clusters):

		# plot dem

		ax.scatter(cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'dem', 0],
		           cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'dem', 1],
		             cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'dem', 2],
		           c=cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'dem', 2],
		           marker='1',
		           cmap=cmaps[cluster_id], label = 'cluster {}, {}'.format(cluster_id, 'democrats'))

		# plot rep
		ax.scatter(cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'rep', 0],
		           cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'rep', 1],
		             cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'rep', 2],
		           c=cluster_to_reduced[cluster_id][np.asarray(cluster_to_labels[cluster_id]) == 'rep', 2],
		           marker='.',
		           cmap=cmaps[cluster_id], label = 'cluster {}, {}'.format(cluster_id, 'republicans'))
	ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
	plt.tight_layout()
	plt.show()
	if args.interactive:
		plt.pause(0.1)

		input('---------- Press enter to quit. ----------')
		plt.close()
	else:
		plt.savefig(os.path.join(folder, 'fig.png'))

def statistics(cluster_to_words, cluster_to_embeddings, folder, cluster_to_labels):
	# cluster_to_labels = np.asarray(cluster_to_labels)
	if args.interactive:
		file_name = 'tmp_stat.txt'
	else:
		file_name = 'stat.txt'
	with open(os.path.join(folder, file_name), 'w') as file:
		total_words = np.sum([len(v) for k, v in cluster_to_words.items() ])
		for cluster_id in range(args.clusters):
			print('cluster {} has {} words, {} % of {}'.format(cluster_id, len(cluster_to_labels[cluster_id]),
			                                                   len(cluster_to_words[cluster_id])*100.0/total_words,
			                                                   total_words))
			file.write('cluster {} has {} words, {} % of {}\n'.format(cluster_id, len(cluster_to_labels[cluster_id]),
			                                                   len(cluster_to_words[cluster_id])*100.0/total_words,
			                                                   total_words))
			if args.mode == 'combined':
				print('\t {} rep words, {} %'.format(np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'rep'),
				                                   np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'rep')*100.0
				                                   /len(cluster_to_labels[cluster_id])))
				print('\t {} dem words, {} %'.format(np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'dem'),
				                                   np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'dem')*100.0
				                                   /len(cluster_to_labels[cluster_id])))
				file.write('\t {} rep words, {} %\n'.format(np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'rep'),
				                                   np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'rep')*100.0
				                                   /len(cluster_to_labels[cluster_id])))
				file.write('\t {} dem words, {} %\n'.format(np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'dem'),
				                                   np.sum(np.asarray(cluster_to_labels[cluster_id]) == 'dem')*100.0
				                                   /len(cluster_to_labels[cluster_id])))


def vis():
	cluster_to_words, cluster_to_embeddings, folder, cluster_to_labels = read_embeddings()

	statistics(cluster_to_words, cluster_to_embeddings, folder, cluster_to_labels)

	cluster_to_length = {}
	all_embeddings = []

	for n in range(args.clusters):
		assert len(cluster_to_words[n]) == len(cluster_to_embeddings[n])
		cluster_to_length[n] = len(cluster_to_words[n])
		all_embeddings.extend(cluster_to_embeddings[n])

	all_embeddings = np.asarray(all_embeddings)

	pca = PCA(n_components=3)

	reduced_embeddings = pca.fit_transform(all_embeddings)

	plot_results(reduced_embeddings, folder, cluster_to_length, cluster_to_labels)

	print()


if __name__ == '__main__':
	glove_embed_file = 'glove.pkl'
	if os.path.exists(glove_embed_file):
		with open(glove_embed_file, 'rb') as file:
			rep_embeds, dem_embeds = p.load(file)
			print('glove loaded from {}'.format(glove_embed_file))
	else:
		print('{} does not exist'.format(glove_embed_file))
		exit(-1)

	glove_embed = dict()
	glove_embed.update(rep_embeds)
	glove_embed.update(dem_embeds)

	vis()


"""
# rep_embeds = 37372, 48.78%
# dem_embeds = 39246, 51.22%
"""
