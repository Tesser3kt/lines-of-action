import torch
import torch.nn as nn
import torch.nn.functional as F
from game.board import Board


ROW_COUNT = 8
COL_COUNT = 8
ACTION_SIZE = (ROW_COUNT * COL_COUNT) ** 2


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, board: Board, num_resblocks: int, num_hidden: int, device: str):
        super().__init__()
        self.device = device
        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.back_bone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resblocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ROW_COUNT * COL_COUNT, ACTION_SIZE),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * ROW_COUNT * COL_COUNT, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x):
        x = self.start_block(x)
        for res_block in self.back_bone:
            x = res_block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value
