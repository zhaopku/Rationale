modes = ['combined', 'separated']
clusters = [2, 3, 4, 5, 6, 7, 8, 10]
party = ['dem', 'rep']

for m in modes:
	for c in clusters:
		for p in party:
			print('python visualize.py --mode {} --clusters {} --party {} --sample 1.0 &&'.format(m, c, p))
			if m == 'combined':
				break