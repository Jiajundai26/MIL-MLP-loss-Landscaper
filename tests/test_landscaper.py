"""Tests for MIL/MLP loss-landscape framework.

These tests are intentionally lightweight (CPU-only, tiny models) so they
run quickly in CI without a GPU.
"""

from __future__ import annotations

import copy
from typing import List

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import package under test
# ---------------------------------------------------------------------------
from landscaper import (
    AttentionMILMLP,
    EmbeddingBagDataset,
    LandscapeConfig,
    LandscapeEvaluator,
    MILClassificationAdapter,
    MLPAdapter,
    StandardClassificationAdapter,
    apply_perturbation,
    filter_normalize_direction,
    padded_bag_collate,
    random_direction,
    run_landscape,
    sample_1d,
    sample_2d,
    variable_length_collate,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def small_mil_model() -> AttentionMILMLP:
    torch.manual_seed(0)
    return AttentionMILMLP(d_in=8, d_hidden=16, n_classes=2)


@pytest.fixture()
def small_mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))


@pytest.fixture()
def embedding_bag_dataset() -> EmbeddingBagDataset:
    torch.manual_seed(1)
    bags = [torch.randn(n, 8) for n in [4, 6, 5, 3, 7]]
    labels = [torch.tensor(i % 2) for i in range(5)]
    return EmbeddingBagDataset(bags, labels)


# ===========================================================================
# EmbeddingBagDataset
# ===========================================================================


class TestEmbeddingBagDataset:
    def test_length(self, embedding_bag_dataset):
        assert len(embedding_bag_dataset) == 5

    def test_item_keys(self, embedding_bag_dataset):
        item = embedding_bag_dataset[0]
        assert "bag" in item and "label" in item

    def test_label_is_tensor(self, embedding_bag_dataset):
        item = embedding_bag_dataset[0]
        assert isinstance(item["label"], torch.Tensor)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            EmbeddingBagDataset([torch.randn(3, 8)], [0, 1])


# ===========================================================================
# Collate functions
# ===========================================================================


class TestCollate:
    def test_variable_length_collate(self, embedding_bag_dataset):
        samples = [embedding_bag_dataset[i] for i in range(3)]
        batch = variable_length_collate(samples)
        assert isinstance(batch["bag"], list)
        assert len(batch["bag"]) == 3
        assert batch["label"].shape == (3,)

    def test_padded_bag_collate(self, embedding_bag_dataset):
        samples = [embedding_bag_dataset[i] for i in range(3)]
        batch = padded_bag_collate(samples)
        B = 3
        n_max = max(s["bag"].shape[0] for s in samples)
        assert batch["bag"].shape == (B, n_max, 8)
        assert batch["label"].shape == (B,)
        assert batch["mask"].shape == (B, n_max)
        # Real instances should be marked True
        assert batch["mask"][0, 0].item() is True


# ===========================================================================
# AttentionMILMLP forward
# ===========================================================================


class TestAttentionMILMLP:
    def test_output_keys(self, small_mil_model):
        bag = torch.randn(5, 8)
        out = small_mil_model(bag)
        assert "logits" in out
        assert "attention" in out
        assert "bag_embedding" in out

    def test_logits_shape(self, small_mil_model):
        bag = torch.randn(7, 8)
        out = small_mil_model(bag)
        assert out["logits"].shape == (1, 2)

    def test_attention_sums_to_one(self, small_mil_model):
        bag = torch.randn(6, 8)
        out = small_mil_model(bag)
        assert torch.isclose(out["attention"].sum(), torch.tensor(1.0), atol=1e-5)

    def test_bag_embedding_shape(self, small_mil_model):
        bag = torch.randn(4, 8)
        out = small_mil_model(bag)
        assert out["bag_embedding"].shape == (16,)


# ===========================================================================
# MILClassificationAdapter
# ===========================================================================


class TestMILClassificationAdapter:
    def setup_method(self):
        self.adapter = MILClassificationAdapter()
        self.device = torch.device("cpu")

    def test_compute_loss_single_bag(self, small_mil_model):
        batch = {"bag": torch.randn(5, 8), "label": torch.tensor(1)}
        loss = self.adapter.compute_loss(small_mil_model, batch, self.device)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_compute_loss_batched_bags(self, small_mil_model):
        batch = {"bag": torch.randn(3, 5, 8), "label": torch.tensor([0, 1, 0])}
        loss = self.adapter.compute_loss(small_mil_model, batch, self.device)
        assert loss.ndim == 0

    def test_compute_loss_variable_length(self, small_mil_model):
        bags = [torch.randn(n, 8) for n in [4, 6, 3]]
        labels = torch.tensor([0, 1, 0])
        batch = {"bag": bags, "label": labels}
        loss = self.adapter.compute_loss(small_mil_model, batch, self.device)
        assert loss.ndim == 0

    def test_batch_size_single(self):
        batch = {"bag": torch.randn(5, 8), "label": torch.tensor(0)}
        assert self.adapter.batch_size(batch) == 1

    def test_batch_size_batched(self):
        batch = {"bag": torch.randn(3, 5, 8), "label": torch.tensor([0, 1, 0])}
        assert self.adapter.batch_size(batch) == 3

    def test_batch_size_list(self):
        bags = [torch.randn(n, 8) for n in [4, 6, 3]]
        batch = {"bag": bags, "label": torch.tensor([0, 1, 0])}
        assert self.adapter.batch_size(batch) == 3

    def test_move_batch_to_device(self, small_mil_model):
        batch = {"bag": torch.randn(5, 8), "label": torch.tensor(1)}
        moved = self.adapter.move_batch_to_device(batch, self.device)
        assert moved["bag"].device.type == "cpu"


# ===========================================================================
# MLPAdapter
# ===========================================================================


class TestMLPAdapter:
    def setup_method(self):
        self.adapter = MLPAdapter()
        self.device = torch.device("cpu")

    def test_compute_loss_tuple_batch(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        loss = self.adapter.compute_loss(small_mlp, (x, y), self.device)
        assert loss.ndim == 0

    def test_compute_loss_dict_batch(self, small_mlp):
        batch = {"input": torch.randn(4, 8), "label": torch.randint(0, 2, (4,))}
        loss = self.adapter.compute_loss(small_mlp, batch, self.device)
        assert loss.ndim == 0

    def test_batch_size(self):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        assert self.adapter.batch_size((x, y)) == 4


# ===========================================================================
# StandardClassificationAdapter
# ===========================================================================


class TestStandardClassificationAdapter:
    def setup_method(self):
        self.adapter = StandardClassificationAdapter()
        self.device = torch.device("cpu")

    def test_compute_loss_tuple(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        loss = self.adapter.compute_loss(small_mlp, (x, y), self.device)
        assert loss.ndim == 0

    def test_batch_size(self):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        assert self.adapter.batch_size((x, y)) == 4


# ===========================================================================
# Directions
# ===========================================================================


class TestDirections:
    def test_random_direction_keys(self, small_mlp):
        d = random_direction(small_mlp)
        param_names = {n for n, _ in small_mlp.named_parameters()}
        assert set(d.keys()) == param_names

    def test_random_direction_shapes(self, small_mlp):
        d = random_direction(small_mlp)
        for name, param in small_mlp.named_parameters():
            assert d[name].shape == param.shape

    def test_param_filter_zeros_excluded(self, small_mil_model):
        # Only perturb the "cls" layer
        d = random_direction(small_mil_model,
                              param_filter=lambda n, p: n.startswith("cls"))
        for name, delta in d.items():
            if name.startswith("cls"):
                assert delta.norm() > 0, f"{name} should be non-zero"
            else:
                assert delta.norm() == 0, f"{name} should be zero"

    def test_apply_perturbation_changes_params(self, small_mlp):
        base_state = copy.deepcopy(small_mlp.state_dict())
        d = random_direction(small_mlp)
        apply_perturbation(small_mlp, base_state, [(d, 1.0)])
        for name, param in small_mlp.named_parameters():
            if d[name].norm() > 0:
                assert not torch.allclose(param.data, base_state[name]), name

    def test_apply_perturbation_zero_alpha_restores(self, small_mlp):
        base_state = copy.deepcopy(small_mlp.state_dict())
        d = random_direction(small_mlp)
        apply_perturbation(small_mlp, base_state, [(d, 0.0)])
        for name, param in small_mlp.named_parameters():
            assert torch.allclose(param.data, base_state[name])

    def test_filter_normalize_direction(self, small_mlp):
        d = random_direction(small_mlp, normalize=False)
        nd = filter_normalize_direction(d, small_mlp)
        param_dict = dict(small_mlp.named_parameters())
        for name, delta in nd.items():
            if name in param_dict:
                p_norm = param_dict[name].data.norm()
                d_norm = delta.norm()
                if p_norm > 0 and d[name].norm() > 0:
                    assert torch.isclose(d_norm, p_norm, rtol=1e-4), (
                        f"Layer {name}: expected norm {p_norm}, got {d_norm}"
                    )


# ===========================================================================
# LandscapeEvaluator
# ===========================================================================


class TestLandscapeEvaluator:
    def _make_evaluator(self, model, adapter, batches):
        return LandscapeEvaluator(
            model=model,
            adapter=adapter,
            dataloader=batches,
            device=torch.device("cpu"),
        )

    def test_evaluate_loss_returns_float(self, small_mil_model, embedding_bag_dataset):
        dl = DataLoader(embedding_bag_dataset, batch_size=1,
                        collate_fn=variable_length_collate)
        ev = self._make_evaluator(small_mil_model, MILClassificationAdapter(), dl)
        loss = ev.evaluate_loss()
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_loss_max_batches(self, small_mil_model, embedding_bag_dataset):
        dl = DataLoader(embedding_bag_dataset, batch_size=1,
                        collate_fn=variable_length_collate)
        ev = self._make_evaluator(small_mil_model, MILClassificationAdapter(), dl)
        loss_all = ev.evaluate_loss()
        loss_1 = ev.evaluate_loss(max_batches=1)
        # Both should be positive finite floats; they can differ
        assert loss_1 > 0

    def test_evaluate_loss_mlp(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y), (x, y)]
        ev = self._make_evaluator(small_mlp, MLPAdapter(), dl)
        loss = ev.evaluate_loss()
        assert loss > 0


# ===========================================================================
# Samplers
# ===========================================================================


class TestSamplers:
    def _make_simple_setup(self, model, adapter, batches):
        evaluator = LandscapeEvaluator(
            model=model,
            adapter=adapter,
            dataloader=batches,
            device=torch.device("cpu"),
        )
        d = random_direction(model)
        return evaluator, d

    def test_sample_1d_length(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        ev, d = self._make_simple_setup(small_mlp, MLPAdapter(), dl)
        alphas = [-0.5, 0.0, 0.5]
        losses = sample_1d(small_mlp, d, alphas, ev)
        assert len(losses) == 3

    def test_sample_1d_positive_losses(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        ev, d = self._make_simple_setup(small_mlp, MLPAdapter(), dl)
        losses = sample_1d(small_mlp, d, [-0.5, 0.0, 0.5], ev)
        assert all(l > 0 for l in losses)

    def test_sample_2d_shape(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        ev = LandscapeEvaluator(small_mlp, MLPAdapter(), dl, torch.device("cpu"))
        d1 = random_direction(small_mlp)
        d2 = random_direction(small_mlp)
        alphas = [-0.5, 0.0, 0.5]
        betas = [-0.5, 0.0, 0.5]
        Z = sample_2d(small_mlp, d1, d2, alphas, betas, ev)
        assert Z.shape == (3, 3)

    def test_sample_1d_does_not_mutate_model(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        ev, d = self._make_simple_setup(small_mlp, MLPAdapter(), dl)
        original_state = copy.deepcopy(small_mlp.state_dict())
        sample_1d(small_mlp, d, [-0.5, 0.0, 0.5], ev)
        for name, param in small_mlp.named_parameters():
            assert torch.allclose(param.data, original_state[name]), name


# ===========================================================================
# run_landscape
# ===========================================================================


class TestRunLandscape:
    def test_run_landscape_1d_mil(self, small_mil_model, embedding_bag_dataset):
        dl = DataLoader(embedding_bag_dataset, batch_size=1,
                        collate_fn=variable_length_collate)
        cfg = LandscapeConfig(n_points_1d=3, fixed_seed=42)
        result = run_landscape(
            model=small_mil_model,
            adapter=MILClassificationAdapter(),
            dataloader=dl,
            mode="1d",
            config=cfg,
            device="cpu",
        )
        assert result["mode"] == "1d"
        assert len(result["alphas"]) == 3
        assert len(result["losses"]) == 3
        assert all(l > 0 for l in result["losses"])

    def test_run_landscape_2d_mlp(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        cfg = LandscapeConfig(n_points_2d_alpha=3, n_points_2d_beta=3, fixed_seed=0)
        result = run_landscape(
            model=small_mlp,
            adapter=MLPAdapter(),
            dataloader=dl,
            mode="2d",
            config=cfg,
            device="cpu",
        )
        assert result["mode"] == "2d"
        assert result["losses"].shape == (3, 3)

    def test_run_landscape_param_filter(self, small_mil_model, embedding_bag_dataset):
        """Head-only landscape: only 'cls' parameters should be perturbed."""
        dl = DataLoader(embedding_bag_dataset, batch_size=1,
                        collate_fn=variable_length_collate)
        cfg = LandscapeConfig(n_points_1d=3, fixed_seed=7)
        result = run_landscape(
            model=small_mil_model,
            adapter=MILClassificationAdapter(),
            dataloader=dl,
            mode="1d",
            param_filter=lambda n, p: n.startswith("cls"),
            config=cfg,
            device="cpu",
        )
        assert len(result["losses"]) == 3
        # Non-cls parameters in dir1 should be all-zero
        for name, delta in result["dir1"].items():
            if not name.startswith("cls"):
                assert delta.norm() == 0, f"{name} should be zero in head-only direction"

    def test_run_landscape_does_not_mutate_model(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        original_state = copy.deepcopy(small_mlp.state_dict())
        cfg = LandscapeConfig(n_points_1d=3, fixed_seed=1)
        run_landscape(
            model=small_mlp, adapter=MLPAdapter(), dataloader=dl,
            mode="1d", config=cfg, device="cpu",
        )
        for name, param in small_mlp.named_parameters():
            assert torch.allclose(param.data, original_state[name]), name

    def test_run_landscape_2d_mil(self, small_mil_model, embedding_bag_dataset):
        dl = DataLoader(embedding_bag_dataset, batch_size=1,
                        collate_fn=variable_length_collate)
        cfg = LandscapeConfig(n_points_2d_alpha=2, n_points_2d_beta=2, fixed_seed=3)
        result = run_landscape(
            model=small_mil_model,
            adapter=MILClassificationAdapter(),
            dataloader=dl,
            mode="2d",
            config=cfg,
            device="cpu",
        )
        assert result["losses"].shape == (2, 2)
        
    def test_run_landscape_indexed_grid(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        cfg = LandscapeConfig(grid_mode="indexed", steps=4, distance=0.5, fixed_seed=0)
        result = run_landscape(
            model=small_mlp,
            adapter=MLPAdapter(),
            dataloader=dl,
            mode="2d",
            config=cfg,
            device="cpu",
        )
        assert result["alphas"] == pytest.approx([-0.5, -1 / 6, 1 / 6, 0.5], rel=1e-5)
        assert result["betas"] == pytest.approx([-0.5, -1 / 6, 1 / 6, 0.5], rel=1e-5)
        assert result["losses"].shape == (4, 4)

    def test_run_landscape_hessian_mode(self, small_mlp):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        dl = [(x, y)]
        cfg = LandscapeConfig(
            direction_mode="hessian",
            hessian_top_n=2,
            hessian_power_iters=2,
            n_points_2d_alpha=2,
            n_points_2d_beta=2,
        )
        result = run_landscape(
            model=small_mlp,
            adapter=MLPAdapter(),
            dataloader=dl,
            mode="2d",
            config=cfg,
            device="cpu",
        )
        assert result["losses"].shape == (2, 2)

