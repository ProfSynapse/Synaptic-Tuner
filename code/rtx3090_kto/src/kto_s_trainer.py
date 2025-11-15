#!/usr/bin/env python3
"""
KTO-S Trainer Implementation
Adds SIGN correction to standard KTO for stable KL divergence.

Based on research paper showing KTO has early instability when training
from base models. The SIGN correction fixes gradient scaling so higher-KL
responses don't get stronger updates.
"""

import torch
import torch.nn.functional as F
from trl import KTOTrainer
from typing import Tuple


class KTOSTrainer(KTOTrainer):
    """
    KTO trainer with SIGN correction for stable KL divergence.

    Overrides the kto_loss method to add SIGN correction when enabled.
    """

    def __init__(self, *args, use_sign_correction: bool = True, **kwargs):
        """Initialize KTO-S trainer with optional SIGN correction."""
        super().__init__(*args, **kwargs)
        self.use_sign_correction = use_sign_correction

        if self.use_sign_correction:
            print("\n" + "=" * 80)
            print("🔬 KTO-S MODE ENABLED")
            print("=" * 80)
            print("Using SIGN correction for stable KL divergence")
            print("Expected: KL stays < 0.1 through early training")
            print("=" * 80 + "\n")
        else:
            print("\n⚠️  STANDARD KTO MODE (KTO-S disabled)\n")

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute KTO loss with optional SIGN correction.

        Standard KTO:
            KL = policy - reference
            loss_chosen = -log_sigmoid(beta * (reward_chosen - KL_chosen))
            loss_rejected = -log_sigmoid(beta * (KL_rejected - reward_rejected))

        KTO-S (with SIGN correction):
            S_chosen = sign(reward_chosen)
            S_rejected = sign(reward_rejected)
            loss_chosen = -log_sigmoid(beta * (reward_chosen + S_chosen * KL_chosen))
            loss_rejected = -log_sigmoid(beta * (S_rejected * KL_rejected - reward_rejected))
        """
        # Compute KL divergences
        KL_chosen = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        KL_rejected = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        # Compute rewards per sample
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps

        if self.use_sign_correction:
            # KTO-S: Apply SIGN correction (from paper)
            # Desirable:   σ(β * (r + S*z_0))
            # Undesirable: σ(β * (-S*z_0 - r))
            S_chosen = torch.sign(chosen_rewards)
            S_rejected = torch.sign(rejected_rewards)

            chosen_losses = -F.logsigmoid(
                self.beta * (chosen_rewards + S_chosen * KL_chosen)
            )
            rejected_losses = -F.logsigmoid(
                self.beta * (-S_rejected * KL_rejected - rejected_rewards)
            )
        else:
            # Standard KTO: Original formulation
            chosen_losses = -F.logsigmoid(
                self.beta * (chosen_rewards - KL_chosen)
            )
            rejected_losses = -F.logsigmoid(
                self.beta * (KL_rejected - rejected_rewards)
            )

        # Weight and combine losses
        chosen_loss = (chosen_losses * self.desirable_weight).mean()
        rejected_loss = (rejected_losses * self.undesirable_weight).mean()
        loss = chosen_loss + rejected_loss

        # Return: loss, chosen_rewards (full tensor), rejected_rewards (full tensor), KL
        # Don't use .mean() - parent class needs the full tensors
        return loss, chosen_rewards, rejected_rewards, KL_chosen
