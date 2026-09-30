import torch

from scripts.probe_perceptual_geometry import distribution_metrics


def test_distribution_metrics_are_zero_for_identical_logits():
    logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 0.0]])
    metrics = distribution_metrics(logits, logits.clone(), torch.tensor([1, 2]))
    assert torch.allclose(metrics["forward_kl"], torch.zeros(2), atol=1e-7)
    assert torch.allclose(metrics["reverse_kl"], torch.zeros(2), atol=1e-7)
    assert torch.allclose(metrics["js"], torch.zeros(2), atol=1e-7)
    assert torch.equal(metrics["top1_flip"], torch.zeros(2))


def test_distribution_metrics_detect_changed_distribution_and_realized_drop():
    clean = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    view = torch.tensor([[0.0, 4.0], [0.0, 4.0]])
    metrics = distribution_metrics(clean, view, torch.tensor([0, 1]))
    assert metrics["forward_kl"][0] > 0
    assert metrics["reverse_kl"][0] > 0
    assert metrics["js"][0] > 0
    assert metrics["realized_logprob_drop"][0] > 0
    assert metrics["top1_flip"].tolist() == [1.0, 0.0]
