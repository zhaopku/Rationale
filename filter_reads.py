import os
from shutil import copyfile
from tqdm import tqdm

from_path = '/Users/mengzhao/filter_court/all_rationale_words_court'
to_path = '/Users/mengzhao/filter_court/filtered'

file_names = os.listdir(from_path)

for file_name in tqdm(file_names):
	with open(os.path.join(from_path, file_name), 'r') as file:
		lines = file.readlines()

		last_line = lines[-1].split()

		percent = float(last_line[-1])

		if percent > 0.05 and len(lines) > 30:
			copyfile(os.path.join(from_path, file_name), os.path.join(to_path, file_name))
