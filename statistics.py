import pickle as p
from tqdm import tqdm

all_length = []
import matplotlib.pyplot as plt

with open('data/congress/congress--1-20-500.pkl', 'rb') as file:
	data = p.load(file)
	d = 0
	r = 0
	n_samples = len(data.test_samples)

	for sample in tqdm(data.test_samples):
		all_length.append(sample.length)
		if sample.label == 0:
			d += 1
		elif sample.label == 1:
			r += 1
		else:
			print('illegal label = {}'.format(sample.label))


	print('Democrats = {}, {}'.format(d, d/n_samples))
	print('Republican = {}, {}'.format(r, r/n_samples))

plt.hist(all_length)
plt.show()

"""
in training set:

Democrats = 55084, 0.489539823323439
Republican = 57438, 0.5104601766765611

in val set:
Democrats = 6804, 0.4837539992890153
Republican = 7261, 0.5162460007109847

in test set:
Democrats = 6761, 0.4806625906441064
Republican = 7305, 0.5193374093558937

"""