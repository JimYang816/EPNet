"""Coordinate-shared LSTM optimizer, paper section III-C / eqs. (16)--(19)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMOptimizer(nn.Module):
    """Two stacked LSTM cells, 5 units each; separate state per coordinate/task."""
    def __init__(self, hidden_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.first = nn.LSTMCell(1, hidden_size)
        self.second = nn.LSTMCell(hidden_size, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)
        self.double()

    def forward(self, gradient, state=None):
        value = gradient.detach().reshape(-1, 1)  # dashed edges in Fig. 3
        if state is None:
            zero = value.new_zeros((value.shape[0], self.hidden_size))
            state = ((zero, zero), (zero, zero))
        first = self.first(value, state[0])
        second = self.second(first[0], state[1])
        return self.readout(second[0]).reshape(gradient.shape), (first, second)


def train_meta_optimizer(*, layers=5, epochs=100, tasks=20, steps=20, learning_rate=1e-3,
                         seed=20261027, device="cpu", callback=None):
    controls = (layers, epochs, tasks, steps)
    if (any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in controls)
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid meta optimizer controls")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = LSTMOptimizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        w = torch.randn((tasks, layers, layers), generator=generator, dtype=torch.float64, device=device)
        q = torch.randn((tasks, layers), generator=generator, dtype=torch.float64, device=device)
        beta = torch.ones_like(q, requires_grad=True)
        state = None
        for _ in range(steps):
            objective = ((w@beta.unsqueeze(-1)).squeeze(-1)-q).square().sum()
            gradient, = torch.autograd.grad(objective, beta)
            delta, state = model(gradient, state)
            beta = beta+delta
        loss = ((w@beta.unsqueeze(-1)).squeeze(-1)-q).square().sum()/(tasks*layers)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite meta training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if callback is not None:
            callback(epoch+1, history[-1])
    return model.eval().requires_grad_(False), history
