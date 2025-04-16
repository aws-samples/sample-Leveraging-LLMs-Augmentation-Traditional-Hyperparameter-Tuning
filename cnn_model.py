#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################
"""
Created on Fri Mar 28 15:08:58 2025

@author: R. Pivovar
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    An extremely simple CNN with a single convolutional layer
    followed by a fully connected layer for classification.
    Uses sigmoid activation instead of ReLU.
    """
    def __init__(self):
        super(CNN, self).__init__()

        # Single convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # With no pooling, feature size remains 32x32, so: 32x32x16 = 16384
        self.fc = nn.Linear(16384, 10)

    def forward(self, x):
        # Single convolutional layer with sigmoid activation
        x = torch.sigmoid(self.conv1(x))

        # No pooling, so dimensions stay the same

        # Flatten for fully connected layer
        x = torch.flatten(x, 1)

        # Classification layer
        x = self.fc(x)

        return x
