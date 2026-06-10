from __future__ import annotations

import torch
from torch import nn

from nba_forecasting.config import apply_overrides, load_config
from nba_forecasting.metrics import min_ade_loss
from nba_forecasting.models import build_model
from nba_forecasting.models.common import ForecastOutput, forecast_model
from nba_forecasting.models.mart import MLP, ParallelMARTDecoder
from tests.conftest import make_config, make_context


class _ReferenceMARTDecoder(nn.Module):
    """Single-mode decoder kept for equivalence checks against ParallelMARTDecoder."""

    def __init__(self, model_dim: int, horizon_size: int, decoder_hidden_dim: int, multiplier: int = 3) -> None:
        super().__init__()
        self.horizon_size = horizon_size
        self.decoder_mlp = MLP(
            model_dim * multiplier,
            horizon_size * 2,
            hidden_size=(decoder_hidden_dim, decoder_hidden_dim // 2),
        )

    def forward(self, final_feature: torch.Tensor, cur_location: torch.Tensor) -> torch.Tensor:
        outputs = self.decoder_mlp(final_feature).view(-1, self.horizon_size, 2)
        return outputs + cur_location


def test_mart_model_returns_expected_shape() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)
    future = torch.randn(2, 12, 11, 2)
    output = model(
        context,
        future_xy=future,
        teacher_forcing_prob=0.5,
        scheduled_sampling_prob=0.25,
    )
    assert isinstance(output, ForecastOutput)
    assert tuple(output.point.shape) == (2, 12, 11, 2)
    assert torch.isfinite(output.point).all()


def test_mart_sample_forward_returns_reference_style_modes() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    config.model.mart.mart_sample_k = 3
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)

    modes = model.sample_forward(context)
    output = forecast_model(model, context)

    assert tuple(modes.shape) == (2, 12, 11, 3, 2)
    assert output.modes is not None
    assert tuple(output.modes.shape) == (2, 12, 11, 3, 2)
    torch.testing.assert_close(output.point, modes.mean(dim=3))


def test_mart_learned_point_aggregation_returns_mode_logits_and_gradients() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    config.model.mart.mart_sample_k = 3
    config.model.mart.mart_point_aggregation = "learned_softmax"
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)

    output = forecast_model(model, context)
    loss = output.point.square().mean()
    assert output.mode_logits is not None
    assert tuple(output.mode_logits.shape) == (2, 11, 3)
    assert output.modes is not None
    assert tuple(output.point.shape) == (2, 12, 11, 2)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mart_gaussian_mode_distribution_returns_mdn_params_and_gradients() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    config.model.mart.mart_sample_k = 3
    config.model.mart.mart_mode_distribution = "gaussian"
    config.model.mart.mart_point_aggregation = "distribution_softmax"
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)

    output = forecast_model(model, context)
    assert output.distribution is not None
    assert output.modes is not None
    assert output.mode_logits is None
    assert tuple(output.distribution.means.shape) == (2, 12, 11, 3, 2)
    assert tuple(output.distribution.log_sigma.shape) == (2, 12, 11, 3, 2)
    assert tuple(output.distribution.rho.shape) == (2, 12, 11, 3)
    assert tuple(output.distribution.log_pi.shape) == (2, 11, 3)
    torch.testing.assert_close(output.distribution.means, output.modes)
    weights = output.distribution.log_pi.exp().view(2, 1, 11, 3, 1)
    torch.testing.assert_close(output.point, (output.modes * weights).sum(dim=3))

    loss = (
        output.distribution.log_sigma.square().mean() + output.distribution.rho.square().mean() + output.distribution.log_pi.square().mean()
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mart_distribution_detach_features_blocks_encoder_gradients() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    config.model.mart.mart_sample_k = 3
    config.model.mart.mart_mode_distribution = "gaussian"
    config.model.mart.mart_distribution_detach_features = True
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)

    output = forecast_model(model, context)
    assert output.distribution is not None
    loss = output.distribution.log_sigma.square().mean() + output.distribution.log_pi.square().mean()
    loss.backward()

    assert model.distribution_head is not None
    assert any(parameter.grad is not None for parameter in model.distribution_head.parameters())
    assert model.input_fc.weight.grad is None


def test_mart_type_conditioned_decoder_returns_reference_style_modes() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    config.model.mart.mart_sample_k = 3
    config.model.mart.mart_decoder_variant = "type_conditioned"
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)

    output = forecast_model(model, context)

    assert output.modes is not None
    assert tuple(output.modes.shape) == (2, 12, 11, 3, 2)
    assert tuple(output.point.shape) == (2, 12, 11, 2)
    assert torch.isfinite(output.modes).all()


def test_sample_forecaster_uses_structured_compiled_forward() -> None:
    class CompiledLikeWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = module
            self.forward_calls = 0

        def forward(self, *args: object, **kwargs: object) -> ForecastOutput:
            self.forward_calls += 1
            return self._orig_mod(*args, **kwargs)

    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    config.model.mart.mart_sample_k = 3
    model = build_model(config)
    wrapped = CompiledLikeWrapper(model)
    context = make_context(batch_size=2, use_velocity=True)

    output = forecast_model(wrapped, context)

    assert output.modes is not None
    assert tuple(output.modes.shape) == (2, 12, 11, 3, 2)
    assert wrapped.forward_calls == 1


def test_parallel_mart_decoder_matches_independent_decoders_and_gradients() -> None:
    torch.manual_seed(2)
    sample_k = 3
    model_dim = 5
    horizon = 4
    hidden = 8
    multiplier = 3
    references = torch.nn.ModuleList(_ReferenceMARTDecoder(model_dim, horizon, hidden, multiplier=multiplier) for _ in range(sample_k))
    parallel = ParallelMARTDecoder(model_dim, horizon, hidden, sample_k=sample_k, multiplier=multiplier)
    with torch.no_grad():
        for sample_idx, reference in enumerate(references):
            for layer_idx, layer in enumerate(reference.decoder_mlp.layers):
                parallel.weights[layer_idx][sample_idx].copy_(layer.weight)
                parallel.biases[layer_idx][sample_idx].copy_(layer.bias)

    feature_ref = torch.randn(7, model_dim * multiplier, requires_grad=True)
    feature_parallel = feature_ref.detach().clone().requires_grad_(True)
    cur_ref = torch.randn(7, 1, 2, requires_grad=True)
    cur_parallel = cur_ref.detach().clone().requires_grad_(True)

    reference_out = torch.stack([head(feature_ref, cur_ref) for head in references], dim=1)
    parallel_out = parallel(feature_parallel, cur_parallel)

    torch.testing.assert_close(parallel_out, reference_out, rtol=1e-6, atol=1e-6)
    reference_out.square().sum().backward()
    parallel_out.square().sum().backward()
    torch.testing.assert_close(feature_parallel.grad, feature_ref.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cur_parallel.grad, cur_ref.grad, rtol=1e-6, atol=1e-6)
    for sample_idx, reference in enumerate(references):
        for layer_idx, layer in enumerate(reference.decoder_mlp.layers):
            torch.testing.assert_close(parallel.weights[layer_idx].grad[sample_idx], layer.weight.grad, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(parallel.biases[layer_idx].grad[sample_idx], layer.bias.grad, rtol=1e-6, atol=1e-6)


def test_min_ade_loss_uses_best_mart_sample() -> None:
    target = torch.zeros(1, 2, 1, 2)
    modes = torch.tensor([[[[[10.0, 0.0], [0.0, 0.0]]], [[[10.0, 0.0], [0.0, 0.0]]]]])

    loss = min_ade_loss(modes, target)

    assert float(loss.item()) == 0.0


def test_mart_hypergraph_forward_backward() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 64
    config.model.common.message_dim = 128
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 128
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)
    target = torch.randn(2, 12, 11, 2)

    prediction = forecast_model(model, context).point
    from nba_forecasting.metrics import normalized_mse_loss

    loss = normalized_mse_loss(prediction, target)
    loss.backward()

    assert tuple(prediction.shape) == (2, 12, 11, 2)
    assert torch.isfinite(prediction).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mart_softmin_raw_loss_smoke() -> None:
    config, _ = load_config("mart")
    config = apply_overrides(
        config,
        {
            "model.common.hidden_dim": 64,
            "model.common.message_dim": 128,
            "model.common.num_layers": 1,
            "model.mart.mart_sample_k": 3,
            "loss.sample_loss_type": "softmin_ade",
            "loss.sample_loss_space": "raw",
            "loss.softmin_temperature": 2.0,
        },
    )
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)
    target = torch.randn(2, 12, 11, 2)

    output = forecast_model(model, context)
    loss = min_ade_loss(output.modes, target) if output.modes is not None else output.point.square().mean()
    loss.backward()

    assert output.modes is not None
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
