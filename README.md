# Rationale Prediction

TensorFlow reimplementation of [Rationalizing Neural Predictions](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf)

Using Gumbel Softmax instead of REINFORCE.

## Requirements
    1. TensorFlow 1.12
    2. tqdm

## Usage
    See models/train.py for details of commandline options.

## Dataset
    1. Rotten Tomatoes;
    2. Congress Dataset (ETH internal only);
    3. BeerReview Dataset (TODO).

## Result
    See out.csv for selected rationales (labeled with 1).