# Banana craving monkey
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/77f12eeafe1c4e2bae444876625dc3f2)](https://www.codacy.com/manual/fg/banana-craving-monkey?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/banana-craving-monkey&amp;utm_campaign=Badge_Grade) [![CircleCI](https://circleci.com/gh/frgfm/banana-craving-monkey.svg?style=shield)](https://circleci.com/gh/frgfm/banana-craving-monkey)

This repository is an implementation of DQN agent for the navigation project of Udacity Deep Reinforcement Learning nanodegree, in the banana collection game provided by unity.



## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)



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

### Environment rules

This build is similar to the Unity Banana collector discrete environment:

- 37 states
- 4 actions
- solving condition: the agent reaches an average score of 13.0 over 100 episodes

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



## License

Distributed under the MIT License. See `LICENSE` for more information.