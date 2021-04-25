# A Minimal Cognitive Architecture
## Features
* [BriCA](https://github.com/wbap/BriCA1)
* OpenAI Gym
* Perceptual Module + Motor Module
* Autoencoder for Perceptual Module
* PPO for Motor Module
* Curriculum Learning

For the details, please read [this article](https://rondelionai.blogspot.com/2021/04/minimal-cognitive-architecture.html).

## How to Install
* Clone the repository
* BriCA1
    * Follow the instruction [here](http://wbap.github.io/BriCA1/tutorial/introduction.html#installing).
* Cerenaut Core
    * Follow the instruction [here](https://github.com/Cerenaut/cerenaut-pt-core).
* pip install the following
    * OpenAI Gym
        * pip install gym
    * [TensorForce](https://github.com/tensorforce/tensorforce)
        * pip install tensorforce
    * TensorFlow (used in TensorForce and TensorBoard)
        * pip install tensorflow

## To train a perceptual model

### Dumping perceptual data

```
$ python minimal_CA.py 2 --config minimal_CA.json --episode_count 250 --max_steps 1500 --dump dump.txt

```

This feeds the Gym environment observation directly to the reinforcement learnig to cause learning, while dumping the observation to the dump file.

### Training a perceptual model

```
$ python gym_test_simple.py --dataset dump.txt --save-model saved_model.pt --config minimal_CA.json

```
This is to train the simple autoencoder from cerenaut_core and to output the model file.

## Reinforcement learning with the trained perceptual model

```
$ python minimal_CA.py 2 --config minimal_CA.json --episode_count 50 --max_steps 1500 --model saved_model.pt
```

