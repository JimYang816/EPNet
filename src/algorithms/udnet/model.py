"""UDNet equations (13)--(15), DOI 10.1109/TVT.2022.3230143.

Only likelihood statistics and a physical carrier mask enter forward().
The class order follows this repository's bits: 00, 01, 10, 11.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class UDNet(nn.Module):
    def __init__(self, layers=22, window_size=32, hidden_size=48):
        super().__init__()
        if (layers, window_size, hidden_size) != (22, 32, 48):
            raise ValueError("UDNet requires 22 independent 192 -> 48 -> 128 layers")
        self.config = dict(layers=layers, window_size=window_size, hidden_size=hidden_size)
        self.hidden = nn.ModuleList(nn.Linear(6*window_size, hidden_size) for _ in range(layers))
        self.classifiers = nn.ModuleList(nn.Linear(hidden_size, 4*window_size) for _ in range(layers))
        self.register_buffer("constellation", torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])/math.sqrt(2))

    def forward(self, gram, matched, zf, data_mask):
        """Return [batch, 22, 32, 4] probabilities, zero on non-data slots.

        Real coordinates are [Re(all 32), Im(all 32)]. Lambda is one and
        absorbed into the unrestricted linear weights. Residual hidden states
        are added after Tanh; probability residuals are renormalized per slot.
        """
        size = self.config["window_size"]
        if (gram.ndim != 3 or gram.shape[1:] != (2*size, 2*size)
                or matched.shape != gram.shape[:2] or zf.shape != matched.shape
                or data_mask.shape != (len(gram), size) or data_mask.dtype != torch.bool):
            raise ValueError("Invalid UDNet statistics/mask dimensions")
        weight = self.hidden[0].weight
        if any(x.dtype != weight.dtype or x.device != weight.device for x in (gram, matched, zf)):
            raise ValueError("UDNet statistics must match network dtype and device")
        if data_mask.device != weight.device:
            raise ValueError("UDNet mask must be on the network device")
        coordinate_mask = torch.cat((data_mask, data_mask), dim=-1)
        state = zf*coordinate_mask
        matched = matched*coordinate_mask
        hidden_state = probabilities = None
        outputs = []
        for hidden, classifier in zip(self.hidden, self.classifiers):
            product = (gram@state.unsqueeze(-1)).squeeze(-1)*coordinate_mask
            update = torch.tanh(hidden(torch.cat((state, matched, product), dim=-1)))
            hidden_state = update if hidden_state is None else update+hidden_state
            update_p = classifier(hidden_state).reshape(-1, size, 4).softmax(dim=-1)
            probabilities = update_p if probabilities is None else update_p+probabilities
            probabilities = probabilities/probabilities.sum(dim=-1, keepdim=True)
            probabilities = probabilities*data_mask.unsqueeze(-1)
            outputs.append(probabilities)
            expectation = probabilities@self.constellation
            state = torch.cat((expectation[..., 0], expectation[..., 1]), dim=-1)*coordinate_mask
        return torch.stack(outputs, dim=1)
