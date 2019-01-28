module load python_gpu/3.6.4 &&
module load cuda/9.0.176 cudnn/7.3 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 200 --maxSteps 500 --dropOut 0.8 --batchSize 150 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 0.01
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 0.1 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 1.0 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 0.01 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 0.1 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 1.0 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 0.01 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 0.1 &&
bsub -W 48:00 -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 1.0 &&


New valAcc 0.7499466761464628 at epoch 2
model saved at saved/congress/hSize_lstm_100_steps_500_dep_False_d_0.8_lr_0.001_bt_20_vS_-1_pre_True_elmo_False_rnnL_1_theta_0.1_gamma_1.0_t_0.5/model

module load python_gpu/3.6.4 &&
module load cuda/9.0.176 cudnn/7.3 &&
bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" python main.py --hiddenSize 200 --maxSteps 500 --dropOut 0.8 --batchSize 150 --preEmbedding --learningRate 0.001 --theta 0.01 --gamma 0.01 --loadModel --testModel

--hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 0.1 --loadModel --testModel