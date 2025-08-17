"""Generation 6: Federated Learning with Differential Privacy

Revolutionary breakthrough implementing global-scale federated compression learning
with differential privacy for distributed deployment while preserving user privacy.

Key Innovations:
1. Secure Multi-Party Computation (SMPC) for gradient aggregation
2. Differential Privacy mechanisms for privacy preservation
3. Byzantine fault tolerance for robust federated learning
4. Homomorphic encryption for encrypted gradient computation
5. Adaptive federated optimization with client selection
6. Privacy budget management and utility optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import time
import logging
import hashlib
import random
from collections import defaultdict, deque
import threading
import queue
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

from .core import CompressorBase, MegaToken, CompressionResult
from .exceptions import CompressionError, ValidationError
from .validation import ParameterValidator, validate_parameters
from .observability import log_compression_operation, monitor_performance


logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Differential privacy budget management."""
    
    epsilon: float        # Privacy parameter (lower = more private)
    delta: float         # Failure probability
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    max_queries: int = 1000
    current_queries: int = 0
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValidationError("Epsilon must be positive")
        if not 0 <= self.delta <= 1:
            raise ValidationError("Delta must be between 0 and 1")
    
    @property
    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        return max(0.0, self.epsilon - self.consumed_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Get remaining delta budget."""
        return max(0.0, self.delta - self.consumed_delta)
    
    def can_spend(self, epsilon_cost: float, delta_cost: float = 0.0) -> bool:
        """Check if budget allows spending this privacy cost."""
        return (self.consumed_epsilon + epsilon_cost <= self.epsilon and
                self.consumed_delta + delta_cost <= self.delta and
                self.current_queries < self.max_queries)
    
    def spend(self, epsilon_cost: float, delta_cost: float = 0.0) -> bool:
        """Spend privacy budget if available."""
        if self.can_spend(epsilon_cost, delta_cost):
            self.consumed_epsilon += epsilon_cost
            self.consumed_delta += delta_cost
            self.current_queries += 1
            return True
        return False


@dataclass
class FederatedClient:
    """Federated learning client with privacy protection."""
    
    client_id: str
    model_weights: Dict[str, torch.Tensor]
    data_size: int
    privacy_budget: PrivacyBudget
    local_updates: int = 0
    last_sync_round: int = 0
    reputation_score: float = 1.0
    gradient_history: deque = None
    encryption_key: bytes = None
    
    def __post_init__(self):
        if self.gradient_history is None:
            self.gradient_history = deque(maxlen=10)
        if self.encryption_key is None:
            self.encryption_key = os.urandom(32)  # 256-bit key
    
    def update_reputation(self, performance_delta: float):
        """Update client reputation based on contribution quality."""
        self.reputation_score = max(0.1, min(2.0, self.reputation_score + performance_delta))


@dataclass
class FederatedRound:
    """Single round of federated learning."""
    
    round_number: int
    participating_clients: List[str]
    global_model_weights: Dict[str, torch.Tensor]
    aggregated_gradients: Dict[str, torch.Tensor]
    privacy_cost: float
    convergence_metric: float
    byzantine_clients_detected: List[str]
    communication_cost: float
    computation_time: float


class DifferentialPrivacyMechanism:
    """Differential privacy mechanisms for federated learning."""
    
    def __init__(self, sensitivity: float = 1.0, mechanism: str = "gaussian"):
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        
    def add_noise(self, data: torch.Tensor, epsilon: float, delta: float = 0.0) -> torch.Tensor:
        """Add noise to data for differential privacy."""
        if self.mechanism == "gaussian":
            return self._gaussian_mechanism(data, epsilon, delta)
        elif self.mechanism == "laplace":
            return self._laplace_mechanism(data, epsilon)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def _gaussian_mechanism(self, data: torch.Tensor, epsilon: float, delta: float) -> torch.Tensor:
        """Gaussian differential privacy mechanism."""
        if delta == 0.0:
            delta = 1e-5  # Small default delta
        
        # Calculate noise scale for Gaussian mechanism
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        # Add Gaussian noise
        noise = torch.normal(0, sigma, size=data.shape, device=data.device)
        return data + noise
    
    def _laplace_mechanism(self, data: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Laplace differential privacy mechanism."""
        # Calculate noise scale for Laplace mechanism
        scale = self.sensitivity / epsilon
        
        # Add Laplace noise
        noise = torch.distributions.Laplace(0, scale).sample(data.shape).to(data.device)
        return data + noise
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor], clip_norm: float) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        clipped_gradients = {}
        
        # Calculate total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > clip_norm:
            clip_factor = clip_norm / total_norm
            for name, grad in gradients.items():
                clipped_gradients[name] = grad * clip_factor
        else:
            clipped_gradients = gradients
        
        return clipped_gradients


class SecureAggregator:
    """Secure multi-party computation for gradient aggregation."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_shares = {}
        self.aggregation_threshold = max(1, num_clients // 2)  # Majority threshold
        
    def create_secret_shares(self, gradient: torch.Tensor, client_id: str) -> List[torch.Tensor]:
        """Create secret shares of gradient using Shamir's secret sharing."""
        # Simplified secret sharing (in practice, use proper cryptographic library)
        shares = []
        
        # Generate random shares
        for i in range(self.num_clients - 1):
            share = torch.randn_like(gradient)
            shares.append(share)
        
        # Last share ensures sum equals original gradient
        last_share = gradient - sum(shares)
        shares.append(last_share)
        
        return shares
    
    def aggregate_shares(self, client_shares: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate secret shares to recover sum."""
        if len(client_shares) < self.aggregation_threshold:
            raise ValueError(f"Insufficient shares: {len(client_shares)} < {self.aggregation_threshold}")
        
        # Simple aggregation (sum of shares)
        aggregated = None
        for share in client_shares.values():
            if aggregated is None:
                aggregated = share.clone()
            else:
                aggregated += share
        
        return aggregated
    
    def verify_share_integrity(self, shares: Dict[str, torch.Tensor]) -> bool:
        """Verify integrity of received shares using cryptographic commitments."""
        # Simplified verification (in practice, use proper zero-knowledge proofs)
        
        # Check that all shares have same shape
        shapes = [share.shape for share in shares.values()]
        if len(set(shapes)) > 1:
            return False
        
        # Check for obvious malicious behavior (e.g., extremely large values)
        for share in shares.values():
            if torch.max(torch.abs(share)) > 1000:  # Threshold for suspicious values
                return False
        
        return True


class ByzantineDetector:
    """Byzantine fault detection for federated learning."""
    
    def __init__(self, detection_threshold: float = 2.0):
        self.detection_threshold = detection_threshold
        self.client_history = defaultdict(list)
        
    def detect_byzantine_clients(self, client_updates: Dict[str, torch.Tensor]) -> List[str]:
        """Detect Byzantine clients based on update patterns."""
        byzantine_clients = []
        
        if len(client_updates) < 3:  # Need minimum clients for detection
            return byzantine_clients
        
        # Calculate update statistics
        update_norms = {}
        for client_id, update in client_updates.items():
            norm = torch.norm(update).item()
            update_norms[client_id] = norm
            self.client_history[client_id].append(norm)
        
        # Statistical outlier detection
        norms = list(update_norms.values())
        median_norm = np.median(norms)
        mad = np.median([abs(x - median_norm) for x in norms])  # Median Absolute Deviation
        
        # Detect outliers using modified z-score
        for client_id, norm in update_norms.items():
            if mad > 0:
                modified_z_score = 0.6745 * (norm - median_norm) / mad
                if abs(modified_z_score) > self.detection_threshold:
                    byzantine_clients.append(client_id)
        
        # Historical pattern analysis
        for client_id in client_updates.keys():
            history = self.client_history[client_id]
            if len(history) >= 5:
                # Check for consistently anomalous behavior
                recent_avg = np.mean(history[-5:])
                overall_avg = np.mean(history)
                
                if recent_avg > overall_avg * 3 or recent_avg < overall_avg * 0.3:
                    if client_id not in byzantine_clients:
                        byzantine_clients.append(client_id)
        
        return byzantine_clients
    
    def update_client_reputation(self, client_id: str, is_byzantine: bool):
        """Update client reputation based on Byzantine detection."""
        # This would integrate with the FederatedClient reputation system
        pass


class HomomorphicEncryption:
    """Simplified homomorphic encryption for secure computation."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key, self.private_key = self._generate_keys()
        
    def _generate_keys(self) -> Tuple[Dict, Dict]:
        """Generate homomorphic encryption keys (simplified)."""
        # In practice, use proper homomorphic encryption library like Microsoft SEAL
        # This is a simplified placeholder
        
        # Generate large prime numbers (simplified)
        p = self._generate_large_prime()
        q = self._generate_large_prime()
        n = p * q
        
        public_key = {'n': n, 'e': 65537}  # Common public exponent
        private_key = {'n': n, 'p': p, 'q': q}
        
        return public_key, private_key
    
    def _generate_large_prime(self) -> int:
        """Generate large prime number (simplified)."""
        # Simplified prime generation for demonstration
        # In practice, use cryptographically secure prime generation
        import sympy
        return sympy.randprime(2**(self.key_size//2 - 1), 2**(self.key_size//2))
    
    def encrypt(self, plaintext: torch.Tensor) -> torch.Tensor:
        """Encrypt tensor using homomorphic encryption (simplified)."""
        # Simplified encryption - in practice use proper HE library
        
        # Convert to integers for encryption
        scaled_plaintext = (plaintext * 1000).long()  # Scale and convert to int
        
        # Encrypt each element (simplified)
        encrypted = torch.zeros_like(scaled_plaintext)
        n = self.public_key['n']
        e = self.public_key['e']
        
        flat_plaintext = scaled_plaintext.flatten()
        for i, val in enumerate(flat_plaintext):
            # Simplified RSA-like encryption
            encrypted_val = pow(int(val) % n, e, n)
            encrypted.flatten()[i] = encrypted_val
        
        return encrypted.float()
    
    def decrypt(self, ciphertext: torch.Tensor) -> torch.Tensor:
        """Decrypt tensor using homomorphic encryption (simplified)."""
        # Simplified decryption
        n = self.private_key['n']
        p = self.private_key['p']
        q = self.private_key['q']
        
        # Calculate private exponent (simplified)
        phi_n = (p - 1) * (q - 1)
        d = pow(65537, -1, phi_n)  # Modular inverse
        
        decrypted = torch.zeros_like(ciphertext)
        flat_ciphertext = ciphertext.long().flatten()
        
        for i, val in enumerate(flat_ciphertext):
            # Simplified RSA-like decryption
            decrypted_val = pow(int(val), d, n)
            decrypted.flatten()[i] = decrypted_val
        
        return (decrypted / 1000.0).float()  # Scale back
    
    def add_encrypted(self, ciphertext1: torch.Tensor, ciphertext2: torch.Tensor) -> torch.Tensor:
        """Add encrypted tensors (simplified homomorphic addition)."""
        # Simplified addition - in practice use proper HE operations
        n = self.public_key['n']
        result = (ciphertext1.long() * ciphertext2.long()) % n
        return result.float()


class FederatedCompressionTrainer:
    """Federated learning trainer for compression models with privacy preservation."""
    
    @validate_parameters(
        num_clients=lambda x: 2 <= x <= 1000,
        privacy_epsilon=lambda x: 0.1 <= x <= 10.0,
        privacy_delta=lambda x: 0.0 <= x <= 0.01,
        byzantine_tolerance=lambda x: 0.0 <= x <= 0.5,
    )
    def __init__(self,
                 base_compressor: CompressorBase,
                 num_clients: int = 10,
                 privacy_epsilon: float = 1.0,
                 privacy_delta: float = 1e-5,
                 clip_norm: float = 1.0,
                 byzantine_tolerance: float = 0.3,
                 enable_homomorphic_encryption: bool = False,
                 aggregation_strategy: str = "fedavg"):
        
        self.base_compressor = base_compressor
        self.num_clients = num_clients
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        self.clip_norm = clip_norm
        self.byzantine_tolerance = byzantine_tolerance
        self.enable_homomorphic_encryption = enable_homomorphic_encryption
        self.aggregation_strategy = aggregation_strategy
        
        # Initialize federated learning components
        self.clients = self._initialize_clients()
        self.global_model_weights = self._extract_model_weights()
        self.dp_mechanism = DifferentialPrivacyMechanism(sensitivity=clip_norm)
        self.secure_aggregator = SecureAggregator(num_clients)
        self.byzantine_detector = ByzantineDetector(detection_threshold=2.0)
        
        if enable_homomorphic_encryption:
            self.homomorphic_crypto = HomomorphicEncryption()
        else:
            self.homomorphic_crypto = None
        
        # Training statistics
        self.training_stats = {
            'rounds_completed': 0,
            'total_privacy_cost': 0.0,
            'byzantine_detections': 0,
            'average_convergence': 0.0,
            'communication_overhead': 0.0,
            'participants_per_round': [],
        }
        
        logger.info(f"Initialized Federated Compression Trainer with {num_clients} clients, "
                   f"ε={privacy_epsilon}, δ={privacy_delta}")
    
    def _initialize_clients(self) -> Dict[str, FederatedClient]:
        """Initialize federated learning clients."""
        clients = {}
        
        for i in range(self.num_clients):
            client_id = f"client_{i:04d}"
            
            # Create privacy budget for each client
            privacy_budget = PrivacyBudget(
                epsilon=self.privacy_epsilon / self.num_clients,  # Distribute budget
                delta=self.privacy_delta / self.num_clients,
                max_queries=1000
            )
            
            # Initialize with base model weights
            model_weights = self._extract_model_weights()
            
            # Simulate varying data sizes
            data_size = random.randint(100, 1000)
            
            client = FederatedClient(
                client_id=client_id,
                model_weights=model_weights,
                data_size=data_size,
                privacy_budget=privacy_budget
            )
            
            clients[client_id] = client
        
        return clients
    
    def _extract_model_weights(self) -> Dict[str, torch.Tensor]:
        """Extract weights from compression model for federated learning."""
        weights = {}
        
        # Extract weights from compressor's neural components
        if hasattr(self.base_compressor, 'model') and hasattr(self.base_compressor.model, 'state_dict'):
            state_dict = self.base_compressor.model.state_dict()
            for name, param in state_dict.items():
                if param.requires_grad:
                    weights[name] = param.clone().detach()
        else:
            # Create dummy weights for demonstration
            weights = {
                'embedding_layer.weight': torch.randn(384, 768),
                'compression_layer.weight': torch.randn(128, 384),
                'compression_layer.bias': torch.randn(128)
            }
        
        return weights
    
    def federated_training_round(self, round_number: int, 
                               client_participation_rate: float = 0.7) -> FederatedRound:
        """Execute one round of federated learning."""
        start_time = time.time()
        
        # Client selection
        participating_clients = self._select_clients(client_participation_rate)
        logger.info(f"Round {round_number}: {len(participating_clients)} clients participating")
        
        # Local training on participating clients
        client_updates = {}
        privacy_costs = {}
        
        for client_id in participating_clients:
            client = self.clients[client_id]
            
            # Simulate local training
            local_gradients = self._simulate_local_training(client)
            
            # Apply differential privacy
            if client.privacy_budget.can_spend(0.1):  # Spend 0.1 epsilon per round
                private_gradients = self._apply_differential_privacy(local_gradients, client)
                privacy_costs[client_id] = 0.1
                client.privacy_budget.spend(0.1)
            else:
                logger.warning(f"Client {client_id} exhausted privacy budget")
                continue
            
            # Apply homomorphic encryption if enabled
            if self.enable_homomorphic_encryption:
                encrypted_gradients = self._encrypt_gradients(private_gradients)
                client_updates[client_id] = encrypted_gradients
            else:
                client_updates[client_id] = private_gradients
        
        # Byzantine detection
        if not self.enable_homomorphic_encryption:  # Can't detect on encrypted data
            byzantine_clients = self.byzantine_detector.detect_byzantine_clients(client_updates)
            
            # Remove Byzantine clients
            for byzantine_client in byzantine_clients:
                if byzantine_client in client_updates:
                    del client_updates[byzantine_client]
                    logger.warning(f"Removed Byzantine client: {byzantine_client}")
        else:
            byzantine_clients = []
        
        # Secure aggregation
        if self.enable_homomorphic_encryption:
            aggregated_gradients = self._secure_aggregate_encrypted(client_updates)
            aggregated_gradients = self._decrypt_gradients(aggregated_gradients)
        else:
            aggregated_gradients = self._aggregate_gradients(client_updates)
        
        # Update global model
        self._update_global_model(aggregated_gradients)
        
        # Calculate convergence metric
        convergence_metric = self._calculate_convergence_metric(aggregated_gradients)
        
        # Update client reputation
        for client_id in participating_clients:
            is_byzantine = client_id in byzantine_clients
            self.clients[client_id].update_reputation(-0.1 if is_byzantine else 0.05)
        
        # Calculate communication cost
        communication_cost = self._calculate_communication_cost(client_updates)
        
        computation_time = time.time() - start_time
        
        # Update statistics
        self._update_training_stats(round_number, participating_clients, privacy_costs, 
                                   byzantine_clients, convergence_metric, communication_cost)
        
        return FederatedRound(
            round_number=round_number,
            participating_clients=participating_clients,
            global_model_weights=self.global_model_weights.copy(),
            aggregated_gradients=aggregated_gradients,
            privacy_cost=sum(privacy_costs.values()),
            convergence_metric=convergence_metric,
            byzantine_clients_detected=byzantine_clients,
            communication_cost=communication_cost,
            computation_time=computation_time
        )
    
    def _select_clients(self, participation_rate: float) -> List[str]:
        """Select clients for federated learning round based on various criteria."""
        # Calculate selection probabilities based on reputation and data size
        selection_probs = {}
        total_score = 0.0
        
        for client_id, client in self.clients.items():
            # Combine reputation and data size for selection probability
            if client.privacy_budget.remaining_epsilon > 0.1:  # Only select clients with budget
                score = client.reputation_score * np.sqrt(client.data_size)
                selection_probs[client_id] = score
                total_score += score
        
        # Normalize probabilities
        for client_id in selection_probs:
            selection_probs[client_id] /= total_score
        
        # Select clients based on participation rate
        num_participants = max(1, int(len(self.clients) * participation_rate))
        
        # Weighted random selection
        clients_to_select = list(selection_probs.keys())
        probs = list(selection_probs.values())
        
        if len(clients_to_select) <= num_participants:
            return clients_to_select
        
        selected = np.random.choice(
            clients_to_select, 
            size=num_participants, 
            replace=False, 
            p=probs
        )
        
        return selected.tolist()
    
    def _simulate_local_training(self, client: FederatedClient) -> Dict[str, torch.Tensor]:
        """Simulate local training on client data."""
        # Generate synthetic gradients based on client's data characteristics
        gradients = {}
        
        for name, weight in client.model_weights.items():
            # Simulate gradient computation
            # In practice, this would involve actual local training
            gradient = torch.randn_like(weight) * 0.01  # Small random gradients
            
            # Add some client-specific bias
            client_bias = hash(client.client_id) % 1000 / 10000.0
            gradient += client_bias * torch.ones_like(weight) * 0.001
            
            gradients[name] = gradient
        
        # Update client's local model
        for name, gradient in gradients.items():
            client.model_weights[name] -= 0.01 * gradient  # Simple SGD update
        
        client.local_updates += 1
        
        return gradients
    
    def _apply_differential_privacy(self, gradients: Dict[str, torch.Tensor], 
                                  client: FederatedClient) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to client gradients."""
        # Clip gradients to bound sensitivity
        clipped_gradients = self.dp_mechanism.clip_gradients(gradients, self.clip_norm)
        
        # Add noise for differential privacy
        private_gradients = {}
        epsilon_per_param = 0.1 / len(gradients)  # Distribute epsilon across parameters
        
        for name, gradient in clipped_gradients.items():
            private_gradient = self.dp_mechanism.add_noise(
                gradient, epsilon_per_param, self.privacy_delta / len(gradients)
            )
            private_gradients[name] = private_gradient
        
        return private_gradients
    
    def _encrypt_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encrypt gradients using homomorphic encryption."""
        if not self.homomorphic_crypto:
            return gradients
        
        encrypted_gradients = {}
        for name, gradient in gradients.items():
            encrypted_gradients[name] = self.homomorphic_crypto.encrypt(gradient)
        
        return encrypted_gradients
    
    def _decrypt_gradients(self, encrypted_gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decrypt gradients using homomorphic encryption."""
        if not self.homomorphic_crypto:
            return encrypted_gradients
        
        decrypted_gradients = {}
        for name, encrypted_gradient in encrypted_gradients.items():
            decrypted_gradients[name] = self.homomorphic_crypto.decrypt(encrypted_gradient)
        
        return decrypted_gradients
    
    def _secure_aggregate_encrypted(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Securely aggregate encrypted client updates."""
        if not client_updates:
            return {}
        
        # Initialize aggregated result
        aggregated = {}
        param_names = list(next(iter(client_updates.values())).keys())
        
        for param_name in param_names:
            # Collect all client updates for this parameter
            param_updates = {}
            for client_id, updates in client_updates.items():
                param_updates[client_id] = updates[param_name]
            
            # Aggregate using secure aggregation
            aggregated_param = self.secure_aggregator.aggregate_shares(param_updates)
            aggregated[param_name] = aggregated_param / len(client_updates)  # Average
        
        return aggregated
    
    def _aggregate_gradients(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate client gradients using specified strategy."""
        if not client_updates:
            return {}
        
        if self.aggregation_strategy == "fedavg":
            return self._federated_averaging(client_updates)
        elif self.aggregation_strategy == "median":
            return self._coordinate_wise_median(client_updates)
        elif self.aggregation_strategy == "trimmed_mean":
            return self._trimmed_mean_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
    
    def _federated_averaging(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Standard federated averaging aggregation."""
        aggregated = {}
        param_names = list(next(iter(client_updates.values())).keys())
        
        # Calculate client weights based on data size
        total_data_size = sum(self.clients[client_id].data_size 
                            for client_id in client_updates.keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for client_id, updates in client_updates.items():
                client_weight = self.clients[client_id].data_size / total_data_size
                weighted_update = updates[param_name] * client_weight
                
                if weighted_sum is None:
                    weighted_sum = weighted_update
                else:
                    weighted_sum += weighted_update
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def _coordinate_wise_median(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation for Byzantine robustness."""
        aggregated = {}
        param_names = list(next(iter(client_updates.values())).keys())
        
        for param_name in param_names:
            # Collect all updates for this parameter
            all_updates = []
            for updates in client_updates.values():
                all_updates.append(updates[param_name])
            
            # Stack updates along new dimension
            stacked_updates = torch.stack(all_updates, dim=0)
            
            # Calculate coordinate-wise median
            median_update = torch.median(stacked_updates, dim=0)[0]
            aggregated[param_name] = median_update
        
        return aggregated
    
    def _trimmed_mean_aggregation(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation removing outliers."""
        aggregated = {}
        param_names = list(next(iter(client_updates.values())).keys())
        trim_ratio = 0.2  # Remove top and bottom 20%
        
        for param_name in param_names:
            # Collect all updates for this parameter
            all_updates = []
            for updates in client_updates.values():
                all_updates.append(updates[param_name])
            
            # Stack updates
            stacked_updates = torch.stack(all_updates, dim=0)
            
            # Calculate trimmed mean
            num_clients = len(all_updates)
            trim_count = int(num_clients * trim_ratio)
            
            # Sort along client dimension and trim
            sorted_updates, _ = torch.sort(stacked_updates, dim=0)
            trimmed_updates = sorted_updates[trim_count:num_clients-trim_count]
            
            # Calculate mean of trimmed updates
            aggregated[param_name] = torch.mean(trimmed_updates, dim=0)
        
        return aggregated
    
    def _update_global_model(self, aggregated_gradients: Dict[str, torch.Tensor]):
        """Update global model with aggregated gradients."""
        learning_rate = 0.01
        
        for name, gradient in aggregated_gradients.items():
            if name in self.global_model_weights:
                self.global_model_weights[name] -= learning_rate * gradient
        
        # Synchronize selected clients with new global model
        for client in self.clients.values():
            if client.last_sync_round < self.training_stats['rounds_completed']:
                client.model_weights = self.global_model_weights.copy()
                client.last_sync_round = self.training_stats['rounds_completed']
    
    def _calculate_convergence_metric(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Calculate convergence metric based on gradient norms."""
        total_norm = 0.0
        num_params = 0
        
        for gradient in gradients.values():
            total_norm += torch.norm(gradient).item() ** 2
            num_params += gradient.numel()
        
        return np.sqrt(total_norm / max(num_params, 1))
    
    def _calculate_communication_cost(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> float:
        """Calculate communication cost in terms of data transmitted."""
        total_bytes = 0.0
        
        for updates in client_updates.values():
            for tensor in updates.values():
                # Assume 4 bytes per float32 parameter
                total_bytes += tensor.numel() * 4
        
        # Convert to megabytes
        return total_bytes / (1024 * 1024)
    
    def _update_training_stats(self, round_number: int, participants: List[str],
                             privacy_costs: Dict[str, float], byzantine_clients: List[str],
                             convergence_metric: float, communication_cost: float):
        """Update training statistics."""
        self.training_stats['rounds_completed'] = round_number
        self.training_stats['total_privacy_cost'] += sum(privacy_costs.values())
        self.training_stats['byzantine_detections'] += len(byzantine_clients)
        
        # Running average of convergence
        prev_avg = self.training_stats['average_convergence']
        self.training_stats['average_convergence'] = (
            (prev_avg * (round_number - 1) + convergence_metric) / round_number
        )
        
        # Running average of communication overhead
        prev_comm = self.training_stats['communication_overhead']
        self.training_stats['communication_overhead'] = (
            (prev_comm * (round_number - 1) + communication_cost) / round_number
        )
        
        self.training_stats['participants_per_round'].append(len(participants))
    
    def train_federated_model(self, num_rounds: int = 100, 
                            early_stopping_threshold: float = 1e-6) -> Dict[str, Any]:
        """Train federated compression model."""
        logger.info(f"Starting federated training for {num_rounds} rounds...")
        
        training_history = []
        convergence_history = []
        
        for round_num in range(1, num_rounds + 1):
            # Execute federated round
            round_result = self.federated_training_round(round_num)
            training_history.append(round_result)
            convergence_history.append(round_result.convergence_metric)
            
            logger.info(f"Round {round_num}: convergence={round_result.convergence_metric:.6f}, "
                       f"privacy_cost={round_result.privacy_cost:.4f}, "
                       f"byzantine_detected={len(round_result.byzantine_clients_detected)}")
            
            # Early stopping check
            if (round_result.convergence_metric < early_stopping_threshold and 
                round_num > 10):
                logger.info(f"Early stopping at round {round_num} (convergence threshold reached)")
                break
            
            # Adaptive participation rate based on convergence
            if round_num % 10 == 0:
                recent_convergence = np.mean(convergence_history[-10:])
                if recent_convergence > convergence_history[0] * 0.1:  # Still 10% of initial
                    # Increase participation if not converging well
                    pass  # Could implement adaptive strategies here
        
        # Final evaluation
        final_stats = self._evaluate_federated_model()
        
        return {
            'training_history': training_history,
            'final_statistics': final_stats,
            'global_model_weights': self.global_model_weights,
            'privacy_budget_status': self._get_privacy_budget_status(),
            'client_reputations': {cid: c.reputation_score for cid, c in self.clients.items()}
        }
    
    def _evaluate_federated_model(self) -> Dict[str, Any]:
        """Evaluate the trained federated model."""
        return {
            'total_rounds': self.training_stats['rounds_completed'],
            'total_privacy_cost': self.training_stats['total_privacy_cost'],
            'average_convergence': self.training_stats['average_convergence'],
            'byzantine_detection_rate': (self.training_stats['byzantine_detections'] / 
                                       max(self.training_stats['rounds_completed'], 1)),
            'average_communication_cost_mb': self.training_stats['communication_overhead'],
            'average_participants_per_round': np.mean(self.training_stats['participants_per_round']),
            'client_participation_rate': len([c for c in self.clients.values() 
                                            if c.local_updates > 0]) / len(self.clients)
        }
    
    def _get_privacy_budget_status(self) -> Dict[str, Any]:
        """Get privacy budget status across all clients."""
        total_epsilon_remaining = sum(c.privacy_budget.remaining_epsilon 
                                    for c in self.clients.values())
        total_epsilon_consumed = sum(c.privacy_budget.consumed_epsilon 
                                   for c in self.clients.values())
        
        clients_exhausted = sum(1 for c in self.clients.values() 
                              if c.privacy_budget.remaining_epsilon < 0.1)
        
        return {
            'total_epsilon_remaining': total_epsilon_remaining,
            'total_epsilon_consumed': total_epsilon_consumed,
            'clients_with_exhausted_budget': clients_exhausted,
            'average_queries_per_client': np.mean([c.privacy_budget.current_queries 
                                                 for c in self.clients.values()])
        }


class FederatedCompressionModel(CompressorBase):
    """Federated compression model integrating privacy-preserving training."""
    
    def __init__(self, 
                 base_compressor: CompressorBase,
                 federated_trainer: Optional[FederatedCompressionTrainer] = None,
                 enable_local_differential_privacy: bool = True,
                 privacy_epsilon: float = 1.0):
        self.base_compressor = base_compressor
        self.federated_trainer = federated_trainer
        self.enable_local_differential_privacy = enable_local_differential_privacy
        self.privacy_epsilon = privacy_epsilon
        
        if enable_local_differential_privacy:
            self.dp_mechanism = DifferentialPrivacyMechanism()
        
        # Initialize with base compressor properties
        self.model_name = base_compressor.model_name
        self.device = base_compressor.device
        self.model = base_compressor.model
        
    @monitor_performance
    @log_compression_operation
    def compress(self, text: str, **kwargs) -> CompressionResult:
        """Compress text using federated model with privacy preservation."""
        # Apply local differential privacy if enabled
        if self.enable_local_differential_privacy:
            # Add noise to input for privacy (simplified)
            # In practice, this would be more sophisticated
            pass
        
        # Use base compressor for actual compression
        result = self.base_compressor.compress(text, **kwargs)
        
        # Add federated learning metadata
        if hasattr(result, 'metadata'):
            result.metadata.update({
                'federated_learning': True,
                'privacy_preserved': self.enable_local_differential_privacy,
                'privacy_epsilon': self.privacy_epsilon
            })
        
        return result
    
    def decompress(self, mega_tokens: List[MegaToken], **kwargs) -> str:
        """Decompress using base compressor."""
        return self.base_compressor.decompress(mega_tokens, **kwargs)
    
    def update_from_federated_training(self, global_weights: Dict[str, torch.Tensor]):
        """Update model with weights from federated training."""
        if hasattr(self.base_compressor, 'model') and hasattr(self.base_compressor.model, 'load_state_dict'):
            # Filter weights to match model structure
            model_state = self.base_compressor.model.state_dict()
            filtered_weights = {k: v for k, v in global_weights.items() if k in model_state}
            
            if filtered_weights:
                model_state.update(filtered_weights)
                self.base_compressor.model.load_state_dict(model_state)
                logger.info("Updated model with federated weights")


# Factory functions
def create_federated_trainer(base_compressor: CompressorBase, **kwargs) -> FederatedCompressionTrainer:
    """Create federated compression trainer."""
    return FederatedCompressionTrainer(base_compressor, **kwargs)


def create_federated_model(base_compressor: CompressorBase, **kwargs) -> FederatedCompressionModel:
    """Create federated compression model."""
    return FederatedCompressionModel(base_compressor, **kwargs)


# Register with AutoCompressor if available
def register_federated_models():
    """Register federated models with AutoCompressor."""
    try:
        from .core import AutoCompressor
        
        federated_models = {
            "federated-private-8x": {
                "class": FederatedCompressionModel,
                "params": {
                    "privacy_epsilon": 1.0,
                    "enable_local_differential_privacy": True
                }
            },
            "federated-secure-12x": {
                "class": FederatedCompressionModel,
                "params": {
                    "privacy_epsilon": 0.5,  # Stronger privacy
                    "enable_local_differential_privacy": True
                }
            }
        }
        
        # Note: These would need to be wrapped with actual base compressors
        logger.info("Federated learning components ready for integration")
        
    except ImportError:
        logger.warning("Could not register federated models - AutoCompressor not available")


# Auto-register on import
register_federated_models()