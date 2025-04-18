# DeepRLHockey

This repository contains my complimentary code for a 
[Reinforcement Learning Lecture](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/distributed-intelligence/home/) 
me and my lab partner took at University of Tübingen in 2019. Given the recent advent in Reinforcement Learning (RL) for training LLMs, I thought 
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
Tensorflow V1. You can now find it under 'src/v1'. **I did not test the original code, it is at it was when I wrote 
it.** I do not think that it will be running out of the box.

## Reimplementation
In 2025 I chose to go with a PyTorch reimplementation, which you can find in `src/v2`. Notice, that the details of SAC 
appear to have changed over time (no specific value function networks, ...), however I chose to go with the original 
version I implemented during my student times.

You can find an example on how to train SAC on `pendulum-v1` in `examples/train_sac_pendulum.py` to run it, make sure 
to install all required dependencies, then execute:

**Linux/macOS**
```bash
PYTHONPATH=. python examples/train_sac_pendulum.py
```
**Windows (CMD)**
```
set PYTHONPATH=.
python examples\train_sac_pendulum.py
```

### Experiments
#### SAC
The model trained in `examples\train_sac_pendulum.py` was saved under `examples\models\sac_agent.pt`. It works reasonably well, see here: 
![](assets/sac_on_pendulum.gif)

#### Gemini - Multimodal LLM
Additionally, I evaluated a range of Gemini models in `examples\eval_multimodal_llm_pendulum.py`. I tested 
`gemini-2.0-flash`, `gemini-2.5-flash-preview-04-17`, `gemini-1.5-pro`. Additionally, I tried to add some few shot 
examples to the context window. Either by sampling random actions or by sampling actions from 
`examples\models\sac_agent.pt`. 

So far none of the models worked. 
