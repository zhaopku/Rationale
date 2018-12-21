module load python_gpu/3.6.4 &&
module load cuda/9.0.176 cudnn/7.3 &&
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 0.01 --loadModel --testModel