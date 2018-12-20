module load python_gpu/3.6.4 &&
module load cuda/9.0.176 cudnn/7.3 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 0.01 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 0.1 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 1.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 10.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.01 --gamma 100.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 0.01 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 0.1 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 1.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 10.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.1 --gamma 100.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 0.01 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 0.1 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 1.0 &&
sleep 7200 &&
module load python_gpu/3.6.4 &&
module load cuda/9.0.176 cudnn/7.3 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 10.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 1.0 --gamma 100.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 10.0 --gamma 0.01 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 10.0 --gamma 0.1 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 10.0 --gamma 1.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 10.0 --gamma 10.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 10.0 --gamma 100.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 100.0 --gamma 0.01 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 100.0 --gamma 0.1 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 100.0 --gamma 1.0 &&
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --hiddenSize 100 --maxSteps 500 --dropOut 0.8 --batchSize 20 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 0.00001 --gamma 0.00001
bsub -n 3 -R  "rusage[mem=2048,ngpus_excl_p=1]" python main.py --loadModel --hiddenSize 100 --maxSteps 30 --dropOut 0.8 --batchSize 100 --preEmbedding --elmo --learningRate 0.001 --epochs 100 --theta 100.0 --gamma 100.0
