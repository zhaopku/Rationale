import os
from tqdm import tqdm
from collections import defaultdict

from_path = '/Users/mengzhao/filter_court/all_rationale_words_court'

file_names = os.listdir(from_path)

rationale_cnt = defaultdict(int)

for file_name in tqdm(file_names):
	with open(os.path.join(from_path, file_name), 'r') as file:
		lines = file.readlines()

		begin = False
		for line in lines:
			if line.find('----- below are rationale words -----') != -1:
				begin = True
				continue

			if not begin:
				continue

			if line.startswith('reads % ='):
				break

			if len(line.strip()) < 2:
				continue
			line = line.strip()
			splits = line.split('<SEP>')

			for split in splits:
				if split.strip() == '<SEP>':
					continue
				rationale_cnt[split.strip()] += 1

rationale_cnt = ((k, v) for k, v in rationale_cnt.items())

rationale_cnt = sorted(rationale_cnt, key=lambda x: (-x[1], x[0]))

with open('rationales_cnt.txt', 'w') as file:
	for (k, v) in rationale_cnt:
		file.write('{}, {}\n'.format(k, v))

print()

