import torch

from src.draft_transformer import DraftTransformer


def test_transformer_masks_used_heroes_and_supports_empty_history():
    model = DraftTransformer(
        hero_count=4, patch_count=1, max_action_order=3, team_count=2,
        d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0,
    )
    model.eval()
    logits, used = model(
        torch.tensor([[0, 2, -1], [-1, -1, -1]], dtype=torch.int16),
        torch.tensor([[0, 3, -1], [-1, -1, -1]], dtype=torch.int8),
        torch.tensor([1, 0]), torch.tensor([2, 0]),
        torch.tensor([0, 1]), torch.tensor([1, 0]),
        torch.tensor([[1, 2], [0, 0]]),
    )
    assert logits.shape == (2, 4)
    assert used.tolist() == [[True, False, True, False], [False, False, False, False]]
    assert logits[0, 0].item() < -1e8
    assert logits[0, 2].item() < -1e8
    assert torch.isfinite(logits[1]).all()


def test_transformer_rejects_incompatible_attention_dimensions():
    try:
        DraftTransformer(hero_count=4, patch_count=1, max_action_order=3, d_model=10, nhead=4)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected incompatible dimensions to fail")
