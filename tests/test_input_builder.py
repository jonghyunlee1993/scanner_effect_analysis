import numpy as np
import torch

from prenorm.models import Canonicalizer, DifferentiableInputBuilder, SetStyleEncoder


def stain_artifact(tmp_path):
    path = tmp_path / "stain.npz"
    matrix = np.asarray(
        [[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]], dtype=np.float32
    )
    np.savez(path, matrix=matrix, concentration_scale=np.asarray([2.0, 2.0], np.float32))
    return path


def test_full_and_structure_channel_contract(tmp_path):
    builder = DifferentiableInputBuilder(stain_artifact(tmp_path), mode="full")
    rgb = torch.rand(2, 3, 32, 32) * 2 - 1
    full = builder(rgb)
    structure = builder(rgb, mode="structure")
    assert full.shape == (2, 8, 32, 32)
    assert structure.shape == (2, 5, 32, 32)
    assert torch.allclose(full[:, 3:], structure)
    assert structure.min() >= 0 and structure.max() <= 1


def test_all_derived_channels_backpropagate_to_rgb(tmp_path):
    builder = DifferentiableInputBuilder(stain_artifact(tmp_path))
    rgb = (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_()
    builder(rgb)[:, 3:].mean().backward()
    assert rgb.grad is not None
    assert torch.count_nonzero(rgb.grad) > 0


def test_group_dropout_is_shared_and_preserves_structure(tmp_path):
    builder = DifferentiableInputBuilder(stain_artifact(tmp_path), group_dropout_p=1.0)
    mask = builder.sample_group_mask(4)
    assert mask[:, (1, 2, 4)].any(dim=1).all()
    rgb = torch.rand(4, 3, 3, 16, 16) * 2 - 1
    output = builder(rgb, group_mask=mask)
    for coordinate in range(4):
        dropped_channels = output[coordinate].abs().sum(dim=(-2, -1)) == 0
        assert torch.equal(dropped_channels[0], dropped_channels[1])


def test_set_style_pooling_is_permutation_invariant():
    encoder = SetStyleEncoder(style_dim=8, ndf=8, n_downsample=2)
    context = torch.rand(2, 4, 3, 32, 32)
    present = torch.tensor([[True, True, False, True], [True, True, True, True]])
    permutation = torch.tensor([3, 0, 2, 1])
    first = encoder(context, present)
    second = encoder(context[:, permutation], present[:, permutation])
    assert torch.allclose(first, second, atol=1e-6)


def test_canonicalize_does_not_call_style_encoder(tmp_path):
    model = Canonicalizer(
        ["at2", "other"],
        "at2",
        stain_artifact(tmp_path),
        content_ch=64,
        style_dim=8,
        ngf=16,
        n_res=1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("style encoder entered canonical inference")

    model.style_encoder.forward = forbidden
    output = model.canonicalize(torch.rand(2, 3, 32, 32) * 2 - 1)
    assert output.shape == (2, 3, 32, 32)
    assert output.min() >= -1 and output.max() <= 1
