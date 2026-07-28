import importlib
from importlib.metadata import version

import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    device = torch.device("cuda")
    assert torch.cuda.get_device_capability(device)[0] >= 9
    return device


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("flash-attn-3") == "3.0.0b1+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize(
    "module_name", ["flash_attn_3", "flash_attn_3._C", "flash_attn_interface"]
)
def test_native_module(device: torch.device, module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hopper_flash_attention(
    device: torch.device, causal: bool, dtype: torch.dtype
) -> None:
    from flash_attn_interface import flash_attn_func

    torch.manual_seed(0)
    query, key, value = (
        torch.randn((2, 32, 4, 64), device=device, dtype=dtype, requires_grad=True)
        for _ in range(3)
    )
    actual = flash_attn_func(query, key, value, causal=causal)
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2).float(),
            key.transpose(1, 2).float(),
            value.transpose(1, 2).float(),
            is_causal=causal,
        )
        .transpose(1, 2)
        .to(dtype)
    )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    actual.square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
