import pickle as p
from tqdm import tqdm

with open('data/congress/congress--1-20-500.pkl', 'rb') as file:
	data = p.load(file)
	d = 0
	r = 0
	n_samples = len(data.train_samples)

	for sample in tqdm(data.train_samples):
		if sample.label == 0:
			d += 1
		elif sample.label == 1:
			r += 1
		else:
			print('illegal label = {}'.format(sample.label))


	print('Democrats = {}, {}'.format(d, d/n_samples))
	print('Republican = {}, {}'.format(r, r/n_samples))


"""
in training set:

Democrats = 55084, 0.489539823323439
Republican = 57438, 0.5104601766765611
"""