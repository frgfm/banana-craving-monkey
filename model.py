#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model using Fixed Q-targets

    Args:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        lin_features (int, optional): number of nodes in hidden layers
        bn (bool, optional): whether batch norm should be added after hidden layer
        dropout_prob (float, optional): dropout probability of hidden layers
    """

    def __init__(self, state_size, action_size, lin_features=64, bn=False, dropout_prob=0.):

        super(QNetwork, self).__init__()

        # Number of nodes in FC layers
        if isinstance(lin_features, list):
            lin_features = [state_size] + lin_features + [action_size]
        elif isinstance(lin_features, int):
            lin_features = [state_size, lin_features, action_size]

        dropout_probs = [dropout_prob / 2] * (len(lin_features) - 2) + [dropout_prob, None]

        layers = []
        for idx, out_feats in enumerate(lin_features[1:]):
            layers.append(nn.Linear(lin_features[idx], out_feats))
            # Don't addd anything after last Linear layer
            if idx + 1 < len(lin_features):
                layers.append(nn.ReLU())
                # Add batch norm
                if bn:
                    layers.append(nn.BatchNorm1d(out_feats))
                # Add dropout
                if dropout_probs[idx] > 0:
                    layers.append(nn.Dropout(dropout_probs[idx]))

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        return self.model(state)
