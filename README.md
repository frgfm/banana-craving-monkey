# Banana craving monkey
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/77f12eeafe1c4e2bae444876625dc3f2)](https://www.codacy.com/manual/fg/banana-craving-monkey?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/banana-craving-monkey&amp;utm_campaign=Badge_Grade) [![CircleCI](https://circleci.com/gh/frgfm/banana-craving-monkey.svg?style=shield)](https://circleci.com/gh/frgfm/banana-craving-monkey)

This repository is an implementation of DQN agent for the navigation project of Udacity Deep Reinforcement Learning nanodegree, in the banana collection game provided by unity.

![banana-gif](<https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif>)



## Table of Contents

- [Environment](#environment)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)



## Environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- **0** - move forward.
- **1** - move backward.
- **2** - turn left.
- **3** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)
- [ml-agents](https://github.com/Unity-Technologies/ml-agents) v0.4 (check this [release](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0b) if you encounter issues)

### Installation

You can install the project requirements as follows:

```shell
git clone https://github.com/frgfm/banana-craving-monkey.git
cd banana-craving-monkey
pip install -r requirements.txt
```

Download the environment build corresponding to your OS

- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then extract the archive in the project folder.



If you wish to use the agent trained by repo owner, you can download the model parameters as follows:

```shell
wget https://github.com/frgfm/banana-craving-monkey/releases/download/v0.1.0/dqn_fixed_target.pt
```



## Usage

### Training

All training arguments can be found using the `--help` flag:

```shell
python train.py --help
```

Below you can find an example to train your agent:

```shell
python train.py --deterministic --lr 5e-4 --no-graphics --eps-decay 0.98 --eps-end 0.02
```

### Evaluation

You can use an existing model's checkpoint to evaluate your agent as follows:

```shell
python evaluate.py --checkpoint ./dqn_fixed_target.pt
```



## Credits

This implementation is vastly based on the following papers:

- [DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience replay](https://arxiv.org/abs/1511.05952)



## License

Distributed under the MIT License. See `LICENSE` for more information.