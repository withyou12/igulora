# -*- coding: utf-8 -*-
"""
@Copyright: 2025 Mr. Cui.
@License：the Apache License, Version 2.0
@Author：Mr. Cui,
@version
@Date：
@Desc: lora svd format
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List


class SVDLoRAModule(nn.Linear):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
            self,
            linear_module: nn.Linear,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = False,
            **kwargs
    ):
        super().__init__(in_features, out_features)

        self.weight = linear_module.weight
        self.bias = linear_module.bias

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.fan_in_fan_out = fan_in_fan_out

        self.merge_weights = merge_weights
        self.merged = False
        if self.r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.lora_E = nn.Parameter(
                0.1 * self.weight.new_ones(r, 1)
            )

            self.lora_mask = nn.Parameter(
                self.weight.new_ones(r, 1),
                requires_grad=False
            )

            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False

        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.weight_coeff = nn.Parameter(
            self.weight.new_ones(1), requires_grad=False
        )

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            tmp_1 = self.lora_dropout(x) @ self.lora_A.T
            tmp_2 = tmp_1 @ torch.diag((self.lora_E * self.lora_mask).squeeze())
            tmp_3 = tmp_2 @ self.lora_B.T
            result += tmp_3 * self.scaling / (self.ranknum + 1e-5) * self.weight_coeff**3

        return result