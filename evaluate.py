#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from pathlib import Path
from unityagents import UnityEnvironment
from agent import Agent


def evaluate_agent(agent, env):
    """Agent training function

    Args:
        agent: agent to train
        env: environment to interact with

    Returns:
        score (float): total score of episode
    """
    brain_name = env.brain_names[0]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # get the current state
    state = env_info.vector_observations[0]
    # initialize the score
    score = 0
    while True:
        action = agent.act(state)

        # Perform action in the environment
        env_info = env.step(action)[brain_name]
        # Get next state, reward and completion boolean
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        # Update overall score
        score += reward
        state = next_state
        if done:
            break

    return score


def main(args):

    env = UnityEnvironment(file_name=args.env_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('Number of states:', state_size)

    # Instantiate trained agent
    lin_feats = [args.lin_feats] * (1 + args.nb_hidden)
    agent = Agent(state_size, action_size,
                  train=False,
                  lin_feats=lin_feats,
                  bn=args.bn,
                  device=args.device)

    # Load checkpoint
    model_folder = Path(args.model_folder)
    if model_folder.is_dir() and model_folder.joinpath('model.pt').is_file():
        state_dict = torch.load(model_folder.joinpath('model.pt'), map_location=args.device)
    else:
        raise FileNotFoundError(f"unable to locate {args.checkpoint}/model.pt")
    agent.qnetwork_local.load_state_dict(state_dict)

    # Evaluation run
    score = evaluate_agent(agent, env)

    print(f"Evaluation score: {score}")
    env.close()


if __name__ == "__main__":
    import argparse
    # Environment
    parser = argparse.ArgumentParser(description='Banana craving agent training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no-graphics", dest="no_graphics",
                        help="Should graphical environment be disabled",
                        action="store_true")
    # Input / Output
    parser.add_argument('--env-path', default='./Banana_Linux/Banana.x86_64',
                        help='path to executable unity environment')
    parser.add_argument('--model-folder', default='./outputs', type=str,
                        help='folder where model checkpoint is located')
    # Architecture
    parser.add_argument('--lin-feats', default=64, type=int, help='number of nodes in hidden layers')
    parser.add_argument('--nb-hidden', default=1, type=int, help='number of hidden layers')
    parser.add_argument("--bn", dest="bn",
                        help="should batch norms be added after hidden layers",
                        action="store_true")
    # Device
    parser.add_argument('--device', default=None, help='device')
    args = parser.parse_args()

    main(args)
