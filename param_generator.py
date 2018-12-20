LEONHARD = True
# ./test.sh > out 2>&1 &
hiddensizes = [100]
lrs = [0.001]
dropouts = [0.8]
thetas = [0.01, 0.1, 1.0, 10.0]
gammas = [0.01, 0.1, 1.0, 10.0]

cnt = 0
# --skimloss
for theta in thetas:
	for gamma in gammas:
		for h in hiddensizes:
			for lr in lrs:
				for d in dropouts:
					if cnt == 0:
						print('module load python_gpu/3.6.4 &&')
						print('module load cuda/9.0.176 cudnn/7.3 &&')
					print('bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize {}'
					      ' --maxSteps 30 --dropOut {} --batchSize 100 --preEmbedding --elmo --learningRate {} --epochs 100 --theta {} --gamma {} &&'
					      .format(h, d, lr, theta, gamma))
					cnt += 1
					if cnt % 13 == 0:
						print('sleep 7200 &&')
						print('module load python_gpu/3.6.4 &&')
						print('module load cuda/9.0.176 cudnn/7.3 &&')


