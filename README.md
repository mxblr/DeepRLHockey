# DeepRLHockey

This repository contains my complimentary code for a 
[Reinforcement Learning Lecture](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/distributed-intelligence/home/) 
I took at University of Tübingen in 2019. Given the recent advent in Reinforcement Learning (RL) for training LLMs, I thought 
I'd give this another shot.
      
## Original task
The original task of the assignment was to implement and train a deep RL Agent to successfully compete in the 
[Laser-Hockey-Environment](https://github.com/martius-lab/laser-hockey-env). Additionally, we used the 
[Pendulum environment](https://gymnasium.farama.org/environments/classic_control/pendulum/) provided by OpenAIs `gym` 
package (now [gymnasium](https://gymnasium.farama.org/)).

## Soft-Actor-Critic
Back then, I chose to implement the [Soft-Actor-Critic]((https://arxiv.org/abs/1801.01290)) algorithm by Tuomas Haarnoja, 
Aurick Zhou, Pieter Abbeel and Sergey Levine. 

## Original code 
My implementation heavily leaned on OpenAIs SpinningUp 
[code](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/sac.py), which was implemented in 
Tensorflow V1. You can now find it under 'src/v1'.

## Reimplementation
In 2025 I chose to go with a PyTorch reimplementation, which you can find in `src/v2`.


