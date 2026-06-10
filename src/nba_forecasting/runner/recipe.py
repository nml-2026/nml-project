from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader

from ..config import LossSection
from ..data import Normalizer, _denormalize_predictions
from ..metrics import (
    ball_plausibility_loss,
    best_mode_indices,
    boundary_loss,
    min_ade_loss,
    mixture_nll,
    normalized_mse_loss,
    smoothness_loss,
    softmin_ade_loss,
)
from ..models.builders import TrainingLossPolicy
from ..models.common import ForecastOutput, MixtureParams, forecast_model
from .io import (
    ExponentialMovingAverage,
    _cuda_autocast,
    _mark_compiled_cuda_step_begin,
    _model_parameter_l2,
    _progress_bar,
    _progress_disabled,
    tqdm,
)


class TimingCallback(Protocol):
    def __call__(self, event: str, **payload: object) -> None: ...


def _mixture_params_for_loss(params: MixtureParams, *, detach_means: bool) -> MixtureParams:
    if detach_means:
        return MixtureParams(
            means=params.means.detach(),
            log_sigma=params.log_sigma,
            rho=params.rho,
            log_pi=params.log_pi,
        )
    return params


def _assert_mixture_finite(
    params: MixtureParams,
    *,
    epoch: int | None,
    batch_idx: int,
) -> None:
    for name, tensor in (
        ("means", params.means),
        ("log_sigma", params.log_sigma),
        ("rho", params.rho),
        ("log_pi", params.log_pi),
    ):
        if torch.isfinite(tensor).all():
            continue
        finite = tensor[torch.isfinite(tensor)]
        detail = f"epoch={epoch} batch={batch_idx} tensor=distribution.{name} shape={tuple(tensor.shape)}"
        if finite.numel() > 0:
            detail += f" finite_min={finite.min().item():.6g} finite_max={finite.max().item():.6g}"
        raise FloatingPointError(f"Non-finite mixture parameter during training: {detail}")


@dataclass(slots=True)
class ForecastTrainingRecipe:
    loss_config: LossSection
    normalizer: Normalizer
    device: torch.device
    grad_clip: float
    mixed_precision: bool
    training_policy: TrainingLossPolicy = TrainingLossPolicy()

    def _raw_loss_scale(self, reference: torch.Tensor) -> torch.Tensor:
        if self.loss_config.raw_loss_scale_ft > 0.0:
            return torch.tensor(self.loss_config.raw_loss_scale_ft, dtype=reference.dtype, device=reference.device)
        sigma = self.normalizer.sigma.to(device=reference.device, dtype=reference.dtype)
        return torch.linalg.vector_norm(sigma).clamp_min(1e-6)

    @staticmethod
    def _scheduled_weight(base_weight: float, *, epoch: int | None, start_epoch: int, warmup_epochs: int) -> float:
        if base_weight <= 0.0:
            return 0.0
        if epoch is None:
            return base_weight
        if epoch < start_epoch:
            return 0.0
        if warmup_epochs <= 0:
            return base_weight
        progress = min(1.0, max(0.0, (epoch - start_epoch + 1) / warmup_epochs))
        return base_weight * progress

    @staticmethod
    def _empty_loss_components() -> dict[str, float]:
        return {
            "mse": 0.0,
            "smooth": 0.0,
            "boundary": 0.0,
            "ball": 0.0,
            "sample": 0.0,
            "point": 0.0,
            "nll": 0.0,
            "oracle_student": 0.0,
        }

    def _physics_penalties(
        self,
        prediction: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        components = {"smooth": 0.0, "boundary": 0.0, "ball": 0.0}
        needs_raw = self.loss_config.lambda_smooth > 0.0 or self.loss_config.lambda_boundary > 0.0 or self.loss_config.lambda_ball > 0.0
        if not needs_raw:
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype), components
        total = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        prediction_raw = _denormalize_predictions(prediction, self.normalizer)
        if self.loss_config.lambda_smooth > 0.0:
            smooth = smoothness_loss(prediction_raw)
            components["smooth"] = float(smooth.item())
            total = total + self.loss_config.lambda_smooth * smooth
        if self.loss_config.lambda_boundary > 0.0:
            boundary = boundary_loss(
                prediction_raw,
                court_x_half=self.loss_config.court_x_half,
                court_y_half=self.loss_config.court_y_half,
                margin=self.loss_config.boundary_margin,
            )
            components["boundary"] = float(boundary.item())
            total = total + self.loss_config.lambda_boundary * boundary
        if self.loss_config.lambda_ball > 0.0:
            ball = ball_plausibility_loss(
                prediction_raw,
                static=batch["static"].to(self.device),
                max_dist=self.loss_config.ball_max_dist,
            )
            components["ball"] = float(ball.item())
            total = total + self.loss_config.lambda_ball * ball
        return total, components

    def compute_training_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mse = normalized_mse_loss(prediction, target)
        physics, physics_components = self._physics_penalties(prediction, batch)
        total = mse + physics
        components = self._empty_loss_components()
        components["mse"] = float(mse.item())
        components.update(physics_components)
        return total, components

    def compute_distribution_loss(
        self,
        output: ForecastOutput,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if output.distribution is None:
            raise RuntimeError("Distribution params must be set before computing distribution loss.")
        params = _mixture_params_for_loss(
            output.distribution,
            detach_means=self.loss_config.mixture_nll_detach_means,
        )
        nll = mixture_nll(params, target)
        loss = nll
        oracle_student_loss = output.point.sum() * 0.0
        oracle_student_weight = self._scheduled_weight(
            self.loss_config.oracle_student_loss_weight,
            epoch=epoch,
            start_epoch=self.loss_config.oracle_student_start_epoch,
            warmup_epochs=self.loss_config.oracle_student_warmup_epochs,
        )
        if oracle_student_weight > 0.0:
            if output.oracle_student is None:
                raise RuntimeError("oracle_student must be set when loss.oracle_student_loss_weight > 0.")
            with torch.no_grad():
                best = best_mode_indices(params.means.detach(), target)
                index = best.view(best.shape[0], 1, best.shape[1], 1, 1).expand(
                    -1,
                    params.means.shape[1],
                    -1,
                    -1,
                    params.means.shape[-1],
                )
                teacher = torch.gather(params.means.detach(), 3, index).squeeze(3)
            oracle_student_loss = normalized_mse_loss(output.oracle_student, teacher)
            loss = loss + oracle_student_weight * oracle_student_loss
        physics, physics_components = self._physics_penalties(output.point, batch)
        loss = loss + physics
        with torch.no_grad():
            point_mse = normalized_mse_loss(output.point, target)
        components = self._empty_loss_components()
        components["mse"] = float(point_mse.item())
        components.update(physics_components)
        components["nll"] = float(nll.item())
        components["oracle_student"] = float(oracle_student_loss.item())
        return loss, components

    def compute_sample_loss(
        self,
        output: ForecastOutput,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if output.modes is None:
            raise RuntimeError("Sample modes must be set before computing sample loss.")
        if self.loss_config.sample_loss_type == "mixture_nll":
            return self.compute_distribution_loss(output, target, batch, epoch=epoch)
        modes_for_loss = output.modes
        target_for_loss = target
        point_for_loss = output.point
        sample_scale = torch.ones((), dtype=target.dtype, device=target.device)
        point_scale = torch.ones((), dtype=target.dtype, device=target.device)
        if self.loss_config.sample_loss_space == "raw":
            modes_for_loss = _denormalize_predictions(output.modes, self.normalizer)
            target_for_loss = _denormalize_predictions(target, self.normalizer)
            point_for_loss = _denormalize_predictions(output.point, self.normalizer)
            raw_scale = self._raw_loss_scale(target_for_loss)
            sample_scale = raw_scale
            point_scale = raw_scale.square()
        elif self.loss_config.sample_loss_space != "normalized":
            raise ValueError("loss.sample_loss_space must be one of: normalized, raw.")

        if self.loss_config.sample_loss_type == "hard_min_ade":
            sample_loss = min_ade_loss(modes_for_loss, target_for_loss)
        elif self.loss_config.sample_loss_type == "softmin_ade":
            sample_loss = softmin_ade_loss(
                modes_for_loss,
                target_for_loss,
                temperature=self.loss_config.softmin_temperature,
            )
        else:
            raise ValueError("loss.sample_loss_type must be one of: hard_min_ade, softmin_ade, mixture_nll.")
        sample_loss = sample_loss / sample_scale
        loss = sample_loss
        point_loss = normalized_mse_loss(point_for_loss, target_for_loss) / point_scale
        point_weight = self._scheduled_weight(
            self.loss_config.point_loss_weight,
            epoch=epoch,
            start_epoch=self.loss_config.point_loss_start_epoch,
            warmup_epochs=self.loss_config.point_loss_warmup_epochs,
        )
        if point_weight > 0.0:
            loss = loss + point_weight * point_loss
        nll_loss = output.point.sum() * 0.0
        nll_weight = self._scheduled_weight(
            self.loss_config.mixture_nll_weight,
            epoch=epoch,
            start_epoch=self.loss_config.mixture_nll_start_epoch,
            warmup_epochs=self.loss_config.mixture_nll_warmup_epochs,
        )
        if nll_weight > 0.0:
            if output.distribution is None:
                raise RuntimeError("distribution params must be set when loss.mixture_nll_weight > 0.")
            distribution = _mixture_params_for_loss(
                output.distribution,
                detach_means=self.loss_config.mixture_nll_detach_means,
            )
            nll_loss = mixture_nll(distribution, target)
            loss = loss + nll_weight * nll_loss
        if output.auxiliary_loss is not None:
            loss = loss + output.auxiliary_loss
        physics, physics_components = self._physics_penalties(output.point, batch)
        loss = loss + physics
        with torch.no_grad():
            point_mse = normalized_mse_loss(output.point, target)
        components = self._empty_loss_components()
        components["mse"] = float(point_mse.item())
        components.update(physics_components)
        components["sample"] = float(sample_loss.item())
        components["point"] = float(point_loss.item())
        components["nll"] = float(nll_loss.item())
        return loss, components

    def train_one_epoch(
        self,
        *,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        teacher_forcing_prob: float,
        scheduled_sampling_prob: float,
        ema: ExponentialMovingAverage | None = None,
        max_batches: int | None = None,
        scaler: Any | None = None,
        timing_callback: TimingCallback | None = None,
        epoch: int | None = None,
    ) -> dict[str, float]:
        model.train()
        totals = {
            "loss": 0.0,
            "mse": 0.0,
            "smooth": 0.0,
            "boundary": 0.0,
            "ball": 0.0,
            "sample": 0.0,
            "point": 0.0,
            "nll": 0.0,
            "oracle_student": 0.0,
        }
        steps = 0
        total_batches = max_batches if max_batches is not None else len(dataloader)
        iterator = _progress_bar(
            enumerate(dataloader, start=1),
            total=total_batches,
            desc="train",
            leave=False,
        )
        for batch_idx, batch in iterator:
            if batch_idx == 1 and timing_callback is not None:
                timing_callback("first_step_start")
            context = batch["context"].to(self.device)
            target = batch["future_xy"].to(self.device)

            with _cuda_autocast(self.mixed_precision):
                _mark_compiled_cuda_step_begin(model, self.device)
                mixture_point_estimate = (
                    self.training_policy.mixture_point_estimate if self.training_policy.primary == "mixture_nll" else None
                )
                output = forecast_model(
                    model,
                    context,
                    future_xy=target,
                    teacher_forcing_prob=teacher_forcing_prob,
                    scheduled_sampling_prob=scheduled_sampling_prob,
                    mdn_point_estimate=mixture_point_estimate,
                )
                if self.training_policy.primary == "mixture_nll":
                    if output.distribution is None:
                        raise RuntimeError("Mixture training requires distribution params in the model output.")
                    _assert_mixture_finite(output.distribution, epoch=epoch, batch_idx=batch_idx)
                    loss, components = self.compute_distribution_loss(output, target, batch, epoch=epoch)
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(
                            f"Non-finite mixture loss during training: epoch={epoch} batch={batch_idx} loss={loss.item()}"
                        )
                elif self.training_policy.primary == "sample_modes":
                    if output.modes is None:
                        raise RuntimeError("Sample-mode training requires modes in the model output.")
                    if not torch.isfinite(output.modes).all():
                        raise FloatingPointError("Model produced NaN or Inf sampled trajectories during training.")
                    loss, components = self.compute_sample_loss(output, target, batch, epoch=epoch)
                else:
                    if not torch.isfinite(output.point).all():
                        raise FloatingPointError("Model produced NaN or Inf predictions during training.")
                    loss, components = self.compute_training_loss(output.point, target, batch)
            optimizer.zero_grad()
            if scaler is not None and self.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            if batch_idx == 1 and timing_callback is not None:
                timing_callback("first_forward_backward_optimizer_step_done", model_param_l2=_model_parameter_l2(model))
            totals["loss"] += float(loss.item())
            for key, value in components.items():
                totals[key] += value
            steps += 1
            if tqdm is not None and not _progress_disabled():
                iterator.set_postfix(loss=f"{loss.item():.4f}")
            if max_batches is not None and batch_idx >= max_batches:
                break
        return {key: value / max(steps, 1) for key, value in totals.items()}
