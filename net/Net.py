#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class Net(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(3, 16, 7, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(16, 16, 7, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(16, 16, 7, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(16, 16, 7, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(16, 16, 7, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(inplace=True)
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(16, out_channel, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.conv5(data)
        data = self.final(data)
        return data
