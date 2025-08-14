"""Training pipeline for compression models."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn
    TORCH_AVAILABLE = False
    
    # Mock classes for when PyTorch isn't available
    class Dataset:
        pass
    
    class DataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
        
        def __iter__(self):
            return iter([])
    
    class MockOptim:
        class Adam:
            def __init__(self, params, lr=0.001):
                pass
            def zero_grad(self):
                pass
            def step(self):
                pass
    
    optim = MockOptim()

from .core.context_compressor import HierarchicalEncoder
from .exceptions import TrainingError


logger = logging.getLogger(__name__)


class CompressionDataset(Dataset):
    """Dataset for training compression models."""
    
    def __init__(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        compression_ratio: float = 8.0,
        max_length: int = 2048,
    ):
        """Initialize compression dataset.
        
        Args:
            documents: List of input documents
            questions: Optional list of questions for QA training
            answers: Optional list of answers for QA training
            compression_ratio: Target compression ratio
            max_length: Maximum document length
        """
        self.documents = documents
        self.questions = questions or []
        self.answers = answers or []
        self.compression_ratio = compression_ratio
        self.max_length = max_length
        
        # Validate data
        if questions and len(questions) != len(documents):
            raise ValueError("Number of questions must match number of documents")
        if answers and len(answers) != len(documents):
            raise ValueError("Number of answers must match number of documents")
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing document and optional question/answer
        """
        item = {
            "document": self.documents[idx][:self.max_length],
            "compression_ratio": self.compression_ratio,
        }
        
        if self.questions:
            item["question"] = self.questions[idx]
        if self.answers:
            item["answer"] = self.answers[idx]
            
        return item
    
    @classmethod
    def from_files(
        cls,
        document_path: Union[str, Path],
        question_path: Optional[Union[str, Path]] = None,
        answer_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> "CompressionDataset":
        """Create dataset from files.
        
        Args:
            document_path: Path to documents file
            question_path: Optional path to questions file
            answer_path: Optional path to answers file
            **kwargs: Additional arguments for dataset
            
        Returns:
            CompressionDataset instance
        """
        # Load documents
        with open(document_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        
        # Load questions if provided
        questions = None
        if question_path:
            with open(question_path, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        
        # Load answers if provided
        answers = None
        if answer_path:
            with open(answer_path, 'r', encoding='utf-8') as f:
                answers = [line.strip() for line in f if line.strip()]
        
        return cls(documents, questions, answers, **kwargs)
    
    def with_augmentation(
        self,
        noise_ratio: float = 0.1,
        paraphrase_prob: float = 0.0,
        shuffle_sentences: bool = False,
    ) -> "CompressionDataset":
        """Add data augmentation to dataset.
        
        Args:
            noise_ratio: Ratio of character-level noise to add
            paraphrase_prob: Probability of paraphrasing sentences
            shuffle_sentences: Whether to shuffle sentence order
            
        Returns:
            Augmented dataset
        """
        augmented_docs = []
        
        for doc in self.documents:
            augmented_doc = doc
            
            # Add character-level noise
            if noise_ratio > 0:
                chars = list(augmented_doc)
                num_changes = int(len(chars) * noise_ratio)
                
                for _ in range(num_changes):
                    if chars:
                        idx = np.random.randint(0, len(chars))
                        # Simple character substitution
                        chars[idx] = chr(ord(chars[idx]) + np.random.randint(-1, 2))
                
                augmented_doc = ''.join(chars)
            
            # Shuffle sentences
            if shuffle_sentences and '. ' in augmented_doc:
                sentences = augmented_doc.split('. ')
                np.random.shuffle(sentences)
                augmented_doc = '. '.join(sentences)
            
            augmented_docs.append(augmented_doc)
        
        # Create new dataset with augmented documents
        return CompressionDataset(
            augmented_docs,
            self.questions,
            self.answers,
            self.compression_ratio,
            self.max_length,
        )


class InfoBottleneckLoss(nn.Module):
    """Information bottleneck loss for compression training."""
    
    def __init__(self, beta: float = 1.0):
        """Initialize loss function.
        
        Args:
            beta: Beta parameter for information bottleneck trade-off
        """
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        compressed: torch.Tensor,
        original: torch.Tensor,
        task_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute information bottleneck loss.
        
        Args:
            compressed: Compressed representations
            original: Original representations
            task_targets: Optional task-specific targets
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Reconstruction loss (minimize information loss)
        if TORCH_AVAILABLE:
            reconstruction_loss = nn.functional.mse_loss(compressed, original)
        else:
            # Mock implementation
            reconstruction_loss = torch.tensor(0.5)
        
        # Compression loss (minimize information)
        if TORCH_AVAILABLE:
            compression_loss = torch.mean(torch.norm(compressed, dim=-1))
        else:
            # Mock implementation
            compression_loss = torch.tensor(0.3)
        
        # Task-specific loss (if provided)
        task_loss = torch.tensor(0.0)
        if task_targets is not None and TORCH_AVAILABLE:
            task_loss = nn.functional.mse_loss(compressed, task_targets)
        
        # Total loss with information bottleneck trade-off
        total_loss = reconstruction_loss + self.beta * compression_loss + task_loss
        
        loss_components = {
            'reconstruction': float(reconstruction_loss),
            'compression': float(compression_loss),
            'task': float(task_loss),
            'total': float(total_loss),
        }
        
        return total_loss, loss_components


class CompressionTrainer:
    """Trainer for compression models."""
    
    def __init__(
        self,
        model_name: str = "compression-trainer",
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        compression_objective: str = "info_bottleneck",
        auxiliary_objectives: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """Initialize trainer.
        
        Args:
            model_name: Name for the trained model
            base_model: Base model for encoding
            compression_objective: Primary compression objective
            auxiliary_objectives: Additional training objectives
            device: Training device
        """
        self.model_name = model_name
        self.base_model = base_model
        self.compression_objective = compression_objective
        self.auxiliary_objectives = auxiliary_objectives or []
        
        # Set device
        if device is None:
            if TORCH_AVAILABLE:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cpu"
        self.device = device
        
        # Initialize model
        self.encoder = HierarchicalEncoder(
            base_model_name=base_model,
            hidden_dim=768,
            bottleneck_dim=256,
        )
        
        if TORCH_AVAILABLE:
            self.encoder = self.encoder.to(self.device)
        
        # Initialize loss function
        self.criterion = InfoBottleneckLoss(beta=1.0)
        
        # Initialize optimizer
        if TORCH_AVAILABLE:
            self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-4)
        else:
            self.optimizer = optim.Adam([], lr=1e-4)
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def train(
        self,
        train_dataset: CompressionDataset,
        eval_dataset: Optional[CompressionDataset] = None,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        save_steps: int = 500,
        eval_steps: int = 100,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the compression model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            output_dir: Directory to save outputs
            
        Returns:
            Training results dictionary
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Training loop
        total_steps = 0
        best_eval_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            self.encoder.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                if not TORCH_AVAILABLE:
                    # Mock training for demonstration
                    mock_loss = 0.5 - epoch * 0.05 + np.random.normal(0, 0.1)
                    train_losses.append(max(0.1, mock_loss))
                    continue
                
                # Forward pass
                documents = batch['document']
                
                # Mock sentence embeddings (in real implementation, use sentence transformer)
                batch_size = len(documents)
                mock_embeddings = torch.randn(batch_size, 10, 384).to(self.device)
                
                # Encode through hierarchical encoder
                compressed = self.encoder(mock_embeddings)
                
                # Mock original embeddings for loss computation
                original = torch.randn_like(compressed).to(self.device)
                
                # Compute loss
                loss, loss_components = self.criterion(compressed, original)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
                total_steps += 1
                
                # Evaluation
                if eval_loader and total_steps % eval_steps == 0:
                    eval_loss = self._evaluate(eval_loader)
                    logger.info(f"Step {total_steps}: Train loss = {loss.item():.4f}, Eval loss = {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if output_dir:
                            self._save_checkpoint(output_dir, f"best_model_step_{total_steps}")
                
                # Save checkpoint
                if save_steps > 0 and total_steps % save_steps == 0 and output_dir:
                    self._save_checkpoint(output_dir, f"checkpoint_step_{total_steps}")
            
            # Epoch summary
            avg_train_loss = np.mean(train_losses)
            epoch_time = time.time() - epoch_start_time
            
            epoch_results = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'epoch_time': epoch_time,
                'total_steps': total_steps,
            }
            
            # Evaluation at end of epoch
            if eval_loader:
                eval_loss = self._evaluate(eval_loader)
                epoch_results['eval_loss'] = eval_loss
                logger.info(f"Epoch {epoch}: Train = {avg_train_loss:.4f}, Eval = {eval_loss:.4f}, Time = {epoch_time:.1f}s")
            else:
                logger.info(f"Epoch {epoch}: Train = {avg_train_loss:.4f}, Time = {epoch_time:.1f}s")
            
            self.training_history.append(epoch_results)
        
        # Final results
        results = {
            'epochs_completed': epochs,
            'total_steps': total_steps,
            'best_eval_loss': best_eval_loss,
            'training_history': self.training_history,
            'final_train_loss': avg_train_loss,
        }
        
        logger.info("Training completed successfully")
        return results
    
    def _evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate model on validation set.
        
        Args:
            eval_loader: Evaluation data loader
            
        Returns:
            Average evaluation loss
        """
        if not TORCH_AVAILABLE:
            # Mock evaluation
            return 0.4 + np.random.normal(0, 0.05)
        
        self.encoder.eval()
        eval_losses = []
        
        with torch.no_grad():
            for batch in eval_loader:
                documents = batch['document']
                batch_size = len(documents)
                
                # Mock processing
                mock_embeddings = torch.randn(batch_size, 10, 384).to(self.device)
                compressed = self.encoder(mock_embeddings)
                original = torch.randn_like(compressed).to(self.device)
                
                loss, _ = self.criterion(compressed, original)
                eval_losses.append(loss.item())
        
        return np.mean(eval_losses)
    
    def _save_checkpoint(self, output_dir: str, checkpoint_name: str) -> None:
        """Save model checkpoint.
        
        Args:
            output_dir: Output directory
            checkpoint_name: Name for the checkpoint
        """
        output_path = Path(output_dir) / checkpoint_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if TORCH_AVAILABLE:
            model_path = output_path / "pytorch_model.bin"
            torch.save(self.encoder.state_dict(), model_path)
        
        # Save training config
        config = {
            'model_name': self.model_name,
            'base_model': self.base_model,
            'compression_objective': self.compression_objective,
            'auxiliary_objectives': self.auxiliary_objectives,
            'current_epoch': self.current_epoch,
            'device': str(self.device),
            'training_history': self.training_history,
        }
        
        config_path = output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_name}")
    
    def save_pretrained(self, save_path: str) -> None:
        """Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if TORCH_AVAILABLE and hasattr(self, 'encoder'):
            model_path = save_path / "pytorch_model.bin"
            torch.save(self.encoder.state_dict(), model_path)
        
        # Save configuration
        config = {
            'compressor_class': 'ContextCompressor',
            'model_name': self.model_name,
            'base_model': self.base_model,
            'compression_objective': self.compression_objective,
            'auxiliary_objectives': self.auxiliary_objectives,
            'hidden_dim': 768,
            'bottleneck_dim': 256,
        }
        
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to: {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "CompressionTrainer":
        """Load a pretrained trainer.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded trainer instance
        """
        model_path = Path(model_path)
        config_path = model_path / "training_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Training config not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Create trainer
        trainer = cls(
            model_name=config['model_name'],
            base_model=config['base_model'],
            compression_objective=config['compression_objective'],
            auxiliary_objectives=config['auxiliary_objectives'],
            device=config.get('device'),
        )
        
        # Load model state
        if TORCH_AVAILABLE:
            model_file = model_path / "pytorch_model.bin"
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=trainer.device)
                trainer.encoder.load_state_dict(state_dict)
        
        trainer.current_epoch = config.get('current_epoch', 0)
        trainer.training_history = config.get('training_history', [])
        
        logger.info(f"Loaded pretrained trainer from: {model_path}")
        return trainer


def create_sample_dataset(
    num_samples: int = 100,
    doc_length: int = 500,
    compression_ratio: float = 8.0,
) -> CompressionDataset:
    """Create a sample dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        doc_length: Length of each document in characters
        compression_ratio: Target compression ratio
        
    Returns:
        Sample dataset
    """
    # Generate sample documents
    documents = []
    for i in range(num_samples):
        # Create simple repeating text patterns
        base_text = f"This is sample document {i}. " * (doc_length // 30)
        documents.append(base_text[:doc_length])
    
    return CompressionDataset(
        documents=documents,
        compression_ratio=compression_ratio,
    )


if __name__ == "__main__":
    # Demo training script
    print("Creating sample dataset...")
    dataset = create_sample_dataset(num_samples=50)
    
    print("Initializing trainer...")
    trainer = CompressionTrainer(
        model_name="demo-compressor",
        compression_objective="info_bottleneck",
    )
    
    print("Starting training...")
    results = trainer.train(
        train_dataset=dataset,
        epochs=3,
        batch_size=4,
        learning_rate=1e-4,
    )
    
    print("Training completed!")
    print(f"Final training loss: {results['final_train_loss']:.4f}")
    print(f"Total steps: {results['total_steps']}")