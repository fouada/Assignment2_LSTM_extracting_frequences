"""
Adversarial Robustness Testing for LSTM Models
INNOVATION: Test model against adversarial perturbations

Author: Professional ML Engineering Team
Date: November 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AdversarialTester:
    """
    Test model robustness against adversarial attacks.
    
    INNOVATION HIGHLIGHTS:
    - Evaluates model security and robustness
    - Identifies model vulnerabilities
    - Applicable to real-world safety-critical systems
    - Multiple attack strategies
    
    Key Features:
    - FGSM (Fast Gradient Sign Method) attack
    - PGD (Projected Gradient Descent) attack
    - Random noise baseline
    - Robustness metrics and visualization
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device
    ):
        """
        Initialize Adversarial Tester.
        
        Args:
            model: Model to test
            criterion: Loss function
            device: Device to run on
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        
        # Track attack results
        self.attack_history: List[Dict] = []
        
        logger.info("AdversarialTester initialized")
    
    def fgsm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        targeted: bool = False,
        target_value: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Creates adversarial examples by adding scaled gradients.
        
        Args:
            x: Input tensor
            y: True target
            epsilon: Perturbation magnitude
            targeted: Whether attack is targeted
            target_value: Target value for targeted attack
            
        Returns:
            Adversarial examples and attack info
        """
        self.model.eval()
        
        # Ensure requires_grad
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x_adv)
        
        # Compute loss
        if targeted and target_value is not None:
            # Targeted attack: minimize distance to target
            target = torch.full_like(output, target_value)
            loss = -self.criterion(output, target)
        else:
            # Untargeted attack: maximize loss
            loss = self.criterion(output, y)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        grad_sign = x_adv.grad.sign()
        
        # Create adversarial example
        x_adv = x + epsilon * grad_sign
        x_adv = x_adv.detach()
        
        # Get adversarial prediction
        with torch.no_grad():
            output_adv = self.model(x_adv)
        
        # Compute metrics
        original_loss = self.criterion(output, y).item()
        adversarial_loss = self.criterion(output_adv, y).item()
        perturbation_norm = torch.norm(x_adv - x).item()
        
        attack_info = {
            'attack_type': 'FGSM',
            'epsilon': epsilon,
            'targeted': targeted,
            'original_loss': original_loss,
            'adversarial_loss': adversarial_loss,
            'loss_increase': adversarial_loss - original_loss,
            'perturbation_norm': perturbation_norm,
            'success': adversarial_loss > original_loss if not targeted else adversarial_loss < original_loss
        }
        
        self.attack_history.append(attack_info)
        
        return x_adv, attack_info
    
    def pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iter: int = 40,
        random_start: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Projected Gradient Descent (PGD) attack.
        
        Iterative attack that is stronger than FGSM.
        
        Args:
            x: Input tensor
            y: True target
            epsilon: Max perturbation magnitude
            alpha: Step size
            num_iter: Number of iterations
            random_start: Whether to start from random point
            
        Returns:
            Adversarial examples and attack info
        """
        self.model.eval()
        
        # Initialize adversarial example
        if random_start:
            # Start from random point in epsilon ball
            x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, x.min().item(), x.max().item())
        else:
            x_adv = x.clone()
        
        original_output = self.model(x).detach()
        original_loss = self.criterion(original_output, y).item()
        
        # Iterative attack
        for i in range(num_iter):
            x_adv.requires_grad = True
            
            # Forward pass
            output = self.model(x_adv)
            loss = self.criterion(output, y)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            with torch.no_grad():
                # Gradient ascent step
                x_adv = x_adv + alpha * x_adv.grad.sign()
                
                # Project back to epsilon ball around x
                perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = x + perturbation
                
                # Clamp to valid range
                x_adv = torch.clamp(x_adv, x.min().item(), x.max().item())
        
        # Final evaluation
        with torch.no_grad():
            output_adv = self.model(x_adv)
            adversarial_loss = self.criterion(output_adv, y).item()
        
        perturbation_norm = torch.norm(x_adv - x).item()
        
        attack_info = {
            'attack_type': 'PGD',
            'epsilon': epsilon,
            'alpha': alpha,
            'num_iter': num_iter,
            'original_loss': original_loss,
            'adversarial_loss': adversarial_loss,
            'loss_increase': adversarial_loss - original_loss,
            'perturbation_norm': perturbation_norm,
            'success': adversarial_loss > original_loss * 1.1  # 10% increase threshold
        }
        
        self.attack_history.append(attack_info)
        
        return x_adv, attack_info
    
    def random_noise_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        noise_type: str = 'uniform'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Random noise attack (baseline).
        
        Args:
            x: Input tensor
            y: True target
            epsilon: Noise magnitude
            noise_type: 'uniform' or 'gaussian'
            
        Returns:
            Noisy examples and attack info
        """
        if noise_type == 'uniform':
            noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
        elif noise_type == 'gaussian':
            noise = torch.randn_like(x) * epsilon
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, x.min().item(), x.max().item())
        
        # Evaluate
        with torch.no_grad():
            original_output = self.model(x)
            noisy_output = self.model(x_noisy)
            
            original_loss = self.criterion(original_output, y).item()
            noisy_loss = self.criterion(noisy_output, y).item()
        
        attack_info = {
            'attack_type': f'Random_{noise_type}',
            'epsilon': epsilon,
            'original_loss': original_loss,
            'adversarial_loss': noisy_loss,
            'loss_increase': noisy_loss - original_loss,
            'perturbation_norm': epsilon,
            'success': noisy_loss > original_loss
        }
        
        self.attack_history.append(attack_info)
        
        return x_noisy, attack_info
    
    def robustness_evaluation(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        epsilons: List[float] = [0.01, 0.05, 0.1, 0.2],
        attacks: List[str] = ['fgsm', 'pgd', 'random']
    ) -> Dict:
        """
        Comprehensive robustness evaluation across multiple attacks and epsilons.
        
        Args:
            x_batch: Batch of inputs
            y_batch: Batch of targets
            epsilons: List of perturbation magnitudes to test
            attacks: List of attack types to test
            
        Returns:
            Robustness metrics
        """
        logger.info("Running comprehensive robustness evaluation...")
        
        results = {
            'epsilons': epsilons,
            'attacks': attacks,
            'success_rates': {},
            'avg_loss_increase': {},
            'avg_perturbation': {}
        }
        
        for attack_type in attacks:
            results['success_rates'][attack_type] = []
            results['avg_loss_increase'][attack_type] = []
            results['avg_perturbation'][attack_type] = []
            
            for epsilon in epsilons:
                successes = 0
                loss_increases = []
                perturbations = []
                
                for i in range(len(x_batch)):
                    x = x_batch[i:i+1]
                    y = y_batch[i:i+1]
                    
                    # Run attack
                    if attack_type == 'fgsm':
                        _, info = self.fgsm_attack(x, y, epsilon=epsilon)
                    elif attack_type == 'pgd':
                        _, info = self.pgd_attack(x, y, epsilon=epsilon, num_iter=20)
                    elif attack_type == 'random':
                        _, info = self.random_noise_attack(x, y, epsilon=epsilon)
                    
                    if info['success']:
                        successes += 1
                    loss_increases.append(info['loss_increase'])
                    perturbations.append(info['perturbation_norm'])
                
                # Aggregate results
                success_rate = successes / len(x_batch)
                avg_loss_increase = np.mean(loss_increases)
                avg_perturbation = np.mean(perturbations)
                
                results['success_rates'][attack_type].append(success_rate)
                results['avg_loss_increase'][attack_type].append(avg_loss_increase)
                results['avg_perturbation'][attack_type].append(avg_perturbation)
                
                logger.info(f"{attack_type.upper()} (ε={epsilon}): Success rate: {success_rate:.1%}, Avg loss increase: {avg_loss_increase:.6f}")
        
        return results
    
    def visualize_adversarial_examples(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        save_path: Optional[str] = None
    ):
        """
        Visualize original vs. adversarial examples.
        
        Args:
            x: Input tensor
            y: True target
            epsilon: Perturbation magnitude
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        # Generate adversarial examples
        x_fgsm, info_fgsm = self.fgsm_attack(x, y, epsilon=epsilon)
        x_pgd, info_pgd = self.pgd_attack(x, y, epsilon=epsilon, num_iter=20)
        x_random, info_random = self.random_noise_attack(x, y, epsilon=epsilon)
        
        # Get predictions
        with torch.no_grad():
            pred_original = self.model(x)
            pred_fgsm = self.model(x_fgsm)
            pred_pgd = self.model(x_pgd)
            pred_random = self.model(x_random)
        
        # Convert to numpy
        x_np = x.cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()
        pred_original_np = pred_original.cpu().numpy().flatten()
        pred_fgsm_np = pred_fgsm.cpu().numpy().flatten()
        pred_pgd_np = pred_pgd.cpu().numpy().flatten()
        pred_random_np = pred_random.cpu().numpy().flatten()
        
        # Limit to first 200 samples for clarity
        n_plot = min(200, len(x_np))
        time_steps = np.arange(n_plot)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Original
        axes[0, 0].plot(time_steps, y_np[:n_plot], 'g-', label='Ground Truth', linewidth=2)
        axes[0, 0].plot(time_steps, pred_original_np[:n_plot], 'b--', label='Prediction', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_title(f'Original (Loss: {info_fgsm["original_loss"]:.6f})', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # FGSM
        axes[0, 1].plot(time_steps, y_np[:n_plot], 'g-', label='Ground Truth', linewidth=2)
        axes[0, 1].plot(time_steps, pred_fgsm_np[:n_plot], 'r--', label='Adversarial', linewidth=1.5, alpha=0.7)
        axes[0, 1].set_title(f'FGSM Attack (Loss: {info_fgsm["adversarial_loss"]:.6f}, +{info_fgsm["loss_increase"]:.6f})', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # PGD
        axes[1, 0].plot(time_steps, y_np[:n_plot], 'g-', label='Ground Truth', linewidth=2)
        axes[1, 0].plot(time_steps, pred_pgd_np[:n_plot], 'm--', label='Adversarial', linewidth=1.5, alpha=0.7)
        axes[1, 0].set_title(f'PGD Attack (Loss: {info_pgd["adversarial_loss"]:.6f}, +{info_pgd["loss_increase"]:.6f})', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Random Noise
        axes[1, 1].plot(time_steps, y_np[:n_plot], 'g-', label='Ground Truth', linewidth=2)
        axes[1, 1].plot(time_steps, pred_random_np[:n_plot], 'orange', linestyle='--', label='Noisy', linewidth=1.5, alpha=0.7)
        axes[1, 1].set_title(f'Random Noise (Loss: {info_random["adversarial_loss"]:.6f}, +{info_random["loss_increase"]:.6f})', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
        
        plt.suptitle(f'Adversarial Robustness Test (ε={epsilon})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Adversarial examples visualization saved to {save_path}")
        
        return fig
    
    def plot_robustness_curve(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot robustness curves showing attack success rate vs. epsilon.
        
        Args:
            results: Results from robustness_evaluation
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        epsilons = results['epsilons']
        attacks = results['attacks']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Success rate plot
        for attack in attacks:
            success_rates = results['success_rates'][attack]
            ax1.plot(epsilons, success_rates, marker='o', linewidth=2, label=attack.upper(), markersize=8)
        
        ax1.set_xlabel('Perturbation Magnitude (ε)', fontsize=12)
        ax1.set_ylabel('Attack Success Rate', fontsize=12)
        ax1.set_title('Model Vulnerability to Adversarial Attacks', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Loss increase plot
        for attack in attacks:
            loss_increases = results['avg_loss_increase'][attack]
            ax2.plot(epsilons, loss_increases, marker='s', linewidth=2, label=attack.upper(), markersize=8)
        
        ax2.set_xlabel('Perturbation Magnitude (ε)', fontsize=12)
        ax2.set_ylabel('Average Loss Increase', fontsize=12)
        ax2.set_title('Impact of Adversarial Perturbations', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Robustness curve saved to {save_path}")
        
        return fig

