"""Fit the head diagonal from context-weighted activation moments and the BF16 embedding."""

import json
import math

from common import open_model

def quarter_log2(a, b):
    """Round log2(a/b)/4 to nearest even using exact binary-rational comparisons."""
    an, ad = float(a).as_integer_ratio()
    bn, bd = float(b).as_integer_ratio()
    numerator, denominator = an * bd, ad * bn

    def compare(exponent):
        left = numerator << max(0, -exponent)
        right = denominator << max(0, exponent)
        return (left > right) - (left < right)

    floor = numerator.bit_length() - denominator.bit_length()
    lower = (floor - (compare(floor) < 0)) // 4
    result = compare(4 * lower + 2)
    return lower + (result > 0 or (result == 0 and lower % 2 != 0))


def head_balance(weights, moments, out):
    import torch

    activation = json.loads(moments.read_text())["second_moment"]
    with open_model(weights) as checkpoint:
        embedding = checkpoint.get_slice("model.language_model.embed_tokens.weight")
        rows, columns = embedding.get_shape()
        if len(activation) != columns or not all(math.isfinite(a) and a >= 0 for a in activation):
            raise ValueError("expected one finite, nonnegative head moment per channel")
        weight = torch.zeros(columns, dtype=torch.float64)
        for first in range(0, rows, 1024):
            weight += embedding[first:first+1024].double().square_().sum(0)
    weight /= rows
    exponents = {i: quarter_log2(a, b) for i, (a, b) in enumerate(zip(activation, weight.tolist()))
                 if a > 0 and b > 0}
    if not exponents:
        raise ValueError("no channels have positive weight and activation moments")
    center = sorted(exponents.values())[(len(exponents) - 1) // 2]
    diagonal = [exponents.get(i, center) - center for i in range(columns)]
    (out / "head4_balance_exponents.i8").write_bytes(
        b"".join(power.to_bytes(1, "little", signed=True) for power in diagonal)
    )
