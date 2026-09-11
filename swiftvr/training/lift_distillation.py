"""LIFT-style coarse-to-fine endpoint distillation for compact SwiftVR DiTs.

This module adapts the output-level LIFT objective from CVPR 2026 to SwiftVR's
single endpoint velocity prediction.  It remains strictly teacher-only: no GT or
compressed decoder enters the training objective.

For each sample we fit the teacher endpoint velocity as

    v_T ~= beta_0 + beta_1 * v_S

with differentiable closed-form ordinary least squares.  Coarse alignment drives
(beta_0, beta_1) toward the identity mapping (0, 1); fine refinement minimizes
the remaining regression residual.  The adaptive weight follows the paper's
coarse-to-fine rule w = 1 - min(1, L_coarse).

This is intentionally LIFT-only, not PLACE: spatial/error-grouped PLACE is kept
for a later ablation after the global coarse-to-fine behavior is validated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lift_velocity_terms(
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return per-batch LIFT diagnostics for endpoint velocity tensors.

    Regression coefficients are estimated independently per sample over all
    channel/spatiotemporal elements.  Reductions are performed in FP32.
    """

    if student_velocity.shape != teacher_velocity.shape:
        raise ValueError(
            f"Velocity shape mismatch: {tuple(student_velocity.shape)} vs "
            f"{tuple(teacher_velocity.shape)}"
        )
    if student_velocity.ndim < 2:
        raise ValueError("Velocity tensors must include batch and feature dimensions")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    student = student_velocity.float().flatten(1)
    teacher = teacher_velocity.detach().float().flatten(1)

    student_mean = student.mean(dim=1, keepdim=True)
    teacher_mean = teacher.mean(dim=1, keepdim=True)
    student_centered = student - student_mean
    teacher_centered = teacher - teacher_mean

    covariance = (student_centered * teacher_centered).mean(dim=1, keepdim=True)
    variance = student_centered.square().mean(dim=1, keepdim=True)
    beta1 = covariance / variance.clamp_min(float(epsilon))
    beta0 = teacher_mean - beta1 * student_mean

    coarse_per_sample = beta0.squeeze(1).abs() + (beta1.squeeze(1) - 1.0).abs()
    fitted = beta0 + beta1 * student
    fine_per_sample = (teacher - fitted).square().mean(dim=1)

    # LIFT's adaptive transition: hard residual fitting is suppressed while the
    # coarse identity mismatch is large, then increases automatically toward 1.
    fine_weight_per_sample = 1.0 - coarse_per_sample.clamp(min=0.0, max=1.0)
    lift_per_sample = coarse_per_sample + fine_weight_per_sample * fine_per_sample

    return {
        "lift_loss": lift_per_sample.mean(),
        "lift_coarse_loss": coarse_per_sample.mean(),
        "lift_fine_loss": fine_per_sample.mean(),
        "lift_fine_weight": fine_weight_per_sample.mean(),
        "lift_beta0_abs": beta0.squeeze(1).abs().mean(),
        "lift_beta1": beta1.squeeze(1).mean(),
        "lift_beta1_abs_error": (beta1.squeeze(1) - 1.0).abs().mean(),
    }


def lift_velocity_distillation_objective(
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    *,
    velocity_mse_weight: float = 1.0,
    velocity_cosine_weight: float = 1.0,
    lift_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Direct endpoint KD plus LIFT coarse-to-fine teacher guidance."""

    for name, value in (
        ("velocity_mse_weight", velocity_mse_weight),
        ("velocity_cosine_weight", velocity_cosine_weight),
        ("lift_weight", lift_weight),
    ):
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    if not any(float(value) > 0 for value in (velocity_mse_weight, velocity_cosine_weight, lift_weight)):
        raise ValueError("At least one distillation weight must be nonzero")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if student_velocity.shape != teacher_velocity.shape:
        raise ValueError(
            f"Velocity shape mismatch: {tuple(student_velocity.shape)} vs "
            f"{tuple(teacher_velocity.shape)}"
        )

    student = student_velocity.float()
    teacher = teacher_velocity.detach().float()
    raw_mse = F.mse_loss(student, teacher)
    teacher_power = teacher.square().mean().detach()
    normalized_mse = raw_mse / teacher_power.clamp_min(float(epsilon))
    cosine = F.cosine_similarity(
        student.flatten(1), teacher.flatten(1), dim=1, eps=float(epsilon)
    ).mean()
    cosine_loss = 1.0 - cosine

    lift = lift_velocity_terms(student_velocity, teacher_velocity, epsilon=epsilon)
    total = (
        float(velocity_mse_weight) * normalized_mse
        + float(velocity_cosine_weight) * cosine_loss
        + float(lift_weight) * lift["lift_loss"]
    )
    return {
        "loss": total,
        "velocity_mse": raw_mse,
        "velocity_normalized_mse": normalized_mse,
        "velocity_cosine": cosine,
        "velocity_cosine_loss": cosine_loss,
        "teacher_velocity_power": teacher_power,
        **lift,
    }
