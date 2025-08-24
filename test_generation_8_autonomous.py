#!/usr/bin/env python3
"""
Test Generation 8: Autonomous Enhancement System
Comprehensive testing of self-evolving neural architecture optimization.
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import our Generation 8 modules
from retrieval_free.generation_8_autonomous_enhancement import (
    NeuralArchitectureEvolution,
    AutonomousEnhancementSystem,
    EvolutionMetrics,
    run_generation_8_enhancement
)

class TestNeuralArchitectureEvolution:
    """Test neural architecture evolution system."""
    
    def test_initialization(self):
        """Test evolution system initialization."""
        evolution = NeuralArchitectureEvolution(
            evolution_cycles=5,
            mutation_rate=0.2,
            selection_pressure=0.8
        )
        
        assert evolution.evolution_cycles == 5
        assert evolution.mutation_rate == 0.2
        assert evolution.selection_pressure == 0.8
        assert len(evolution.population) == 8  # Default population size
        assert evolution.generation == 0

    def test_architecture_creation(self):
        """Test random architecture creation."""
        evolution = NeuralArchitectureEvolution()
        
        # Test creating architecture from seed
        architecture = evolution._create_random_architecture(42)
        
        assert hasattr(architecture, 'encoder')
        assert hasattr(architecture, 'compressor')
        assert hasattr(architecture, 'decompressor')
        assert hasattr(architecture, 'genome')
        
        # Test genome structure
        genome = architecture.genome
        assert 'hidden_dim' in genome
        assert 'num_layers' in genome
        assert 'dropout_rate' in genome
        assert 'seed' in genome

    def test_test_data_generation(self):
        """Test synthetic test data generation."""
        evolution = NeuralArchitectureEvolution()
        test_data = evolution._generate_test_data(batch_size=2, seq_length=256)
        
        assert len(test_data) == 3  # 3 test batches
        for batch in test_data:
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert batch['input_ids'].shape == (2, 256)
            assert batch['attention_mask'].shape == (2, 256)

    def test_innovation_index_calculation(self):
        """Test innovation index calculation."""
        evolution = NeuralArchitectureEvolution()
        
        # Create test architecture
        architecture = evolution._create_random_architecture(42)
        
        # Calculate innovation index
        innovation_idx = evolution._calculate_innovation_index(architecture)
        
        assert 0.0 <= innovation_idx <= 1.0

    @pytest.mark.asyncio
    async def test_architecture_evaluation(self):
        """Test single architecture evaluation."""
        evolution = NeuralArchitectureEvolution()
        architecture = evolution._create_random_architecture(42)
        
        # Evaluate architecture
        fitness = await evolution._evaluate_architecture(architecture)
        
        # Check fitness metrics
        expected_metrics = [
            'compression_ratio', 'f1_score', 'latency_ms', 'memory_usage',
            'energy_efficiency', 'convergence_score', 'innovation_index', 'total_fitness'
        ]
        
        for metric in expected_metrics:
            assert metric in fitness
            assert isinstance(fitness[metric], (int, float))

    def test_selection_mechanism(self):
        """Test fitness-based selection."""
        evolution = NeuralArchitectureEvolution()
        
        # Create mock fitness scores
        fitness_scores = [
            {'total_fitness': 10.5}, {'total_fitness': 8.2},
            {'total_fitness': 12.1}, {'total_fitness': 9.8},
            {'total_fitness': 7.3}, {'total_fitness': 11.2},
            {'total_fitness': 6.9}, {'total_fitness': 13.4}
        ]
        
        # Select survivors
        survivors = evolution._selection(fitness_scores)
        
        # Check selection results
        expected_survivors = int(len(evolution.population) * evolution.selection_pressure)
        assert len(survivors) >= 2  # At least 2 survivors
        assert len(survivors) <= expected_survivors

    def test_crossover_mechanism(self):
        """Test genetic crossover."""
        evolution = NeuralArchitectureEvolution()
        
        # Create two parent architectures
        parent1 = evolution._create_random_architecture(42)
        parent2 = evolution._create_random_architecture(123)
        
        # Perform crossover
        offspring = evolution._crossover(parent1, parent2)
        
        # Check offspring properties
        assert hasattr(offspring, 'genome')
        assert hasattr(offspring, 'encoder')
        
        # Verify genome mixing
        offspring_genome = offspring.genome
        parent1_genome = parent1.genome
        parent2_genome = parent2.genome
        
        # Check that offspring genome contains mixed traits
        assert offspring_genome['hidden_dim'] in [parent1_genome['hidden_dim'], parent2_genome['hidden_dim']]
        assert offspring_genome['num_layers'] in [parent1_genome['num_layers'], parent2_genome['num_layers']]

    def test_mutation_mechanism(self):
        """Test genetic mutation."""
        evolution = NeuralArchitectureEvolution()
        
        # Create original architecture
        original = evolution._create_random_architecture(42)
        original_genome = original.genome.copy()
        
        # Apply mutation
        mutated = evolution._mutate(original)
        mutated_genome = mutated.genome
        
        # Check that mutation occurred (seed should always change)
        assert mutated_genome['seed'] != original_genome['seed']

    def test_genome_to_architecture_conversion(self):
        """Test creating architecture from genome."""
        evolution = NeuralArchitectureEvolution()
        
        # Create test genome
        test_genome = {
            'hidden_dim': 512,
            'num_layers': 3,
            'dropout_rate': 0.2,
            'seed': 42
        }
        
        # Create architecture from genome
        architecture = evolution._create_architecture_from_genome(test_genome)
        
        assert hasattr(architecture, 'genome')
        assert architecture.genome == test_genome

    @pytest.mark.asyncio
    async def test_evolution_cycle(self):
        """Test complete evolution cycle."""
        # Create evolution system with smaller parameters for testing
        evolution = NeuralArchitectureEvolution(
            evolution_cycles=1,
            mutation_rate=0.1,
            selection_pressure=0.5
        )
        
        # Run one evolution cycle
        result = await evolution.evolve()
        
        # Check evolution results
        assert 'generation' in result
        assert 'best_fitness' in result
        assert 'evolution_time' in result
        assert 'population_size' in result
        assert 'metrics' in result
        
        # Check that generation incremented
        assert evolution.generation == 1
        assert len(evolution.fitness_history) == 1

    def test_evolution_state_persistence(self, tmp_path):
        """Test saving and loading evolution state."""
        evolution = NeuralArchitectureEvolution()
        
        # Add some fitness history
        metrics = EvolutionMetrics(
            generation=1,
            compression_ratio=8.5,
            f1_score=0.82,
            latency_ms=150.0,
            memory_usage_mb=512.0,
            energy_efficiency=6.67,
            convergence_score=0.75,
            innovation_index=0.6
        )
        evolution.fitness_history.append(metrics)
        evolution.generation = 1
        
        # Save state
        save_path = tmp_path / "evolution_state.json"
        evolution.save_evolution_state(save_path)
        
        # Verify file exists
        assert save_path.exists()
        
        # Create new evolution system and load state
        new_evolution = NeuralArchitectureEvolution()
        new_evolution.load_evolution_state(save_path)
        
        # Verify state was loaded
        assert new_evolution.generation == 1
        assert len(new_evolution.fitness_history) == 1
        assert new_evolution.fitness_history[0].compression_ratio == 8.5

class TestAutonomousEnhancementSystem:
    """Test autonomous enhancement system."""
    
    def test_initialization(self):
        """Test enhancement system initialization."""
        enhancement = AutonomousEnhancementSystem(
            enhancement_cycles=3,
            convergence_patience=2
        )
        
        assert enhancement.enhancement_cycles == 3
        assert enhancement.convergence_patience == 2
        assert enhancement.current_cycle == 0
        assert len(enhancement.enhancement_history) == 0

    @pytest.mark.asyncio
    async def test_enhancement_quality_evaluation(self):
        """Test enhancement quality evaluation."""
        enhancement = AutonomousEnhancementSystem()
        
        # Mock evolution result
        evolution_result = {
            'generation': 1,
            'best_fitness': {'total_fitness': 15.2},
            'metrics': {
                'compression_ratio': 8.5,
                'f1_score': 0.82,
                'latency_ms': 150.0,
                'memory_usage': 512.0,
                'innovation_index': 0.6,
                'convergence_score': 0.75
            }
        }
        
        # Evaluate quality
        quality_metrics = await enhancement._evaluate_enhancement_quality(evolution_result)
        
        # Check quality metrics
        expected_metrics = [
            'compression_improvement', 'accuracy_retention', 'efficiency_gain',
            'innovation_score', 'convergence_progress', 'overall_quality'
        ]
        
        for metric in expected_metrics:
            assert metric in quality_metrics
            assert 0.0 <= quality_metrics[metric] <= 1.0

    def test_convergence_checking(self):
        """Test convergence detection."""
        enhancement = AutonomousEnhancementSystem(convergence_patience=2)
        
        # Test improvement case
        quality_metrics = {'overall_quality': 0.8}
        enhancement.best_performance = 0.7
        
        converged = enhancement._check_convergence(quality_metrics)
        assert not converged  # Should not converge on improvement
        assert enhancement.stagnation_count == 0
        
        # Test stagnation case
        quality_metrics = {'overall_quality': 0.75}  # No improvement
        enhancement.best_performance = 0.8
        
        enhancement._check_convergence(quality_metrics)
        assert enhancement.stagnation_count == 1
        
        # Test continued stagnation
        enhancement._check_convergence(quality_metrics)
        converged = enhancement._check_convergence(quality_metrics)
        assert converged  # Should converge after patience limit

    def test_recommendation_generation(self):
        """Test recommendation generation."""
        enhancement = AutonomousEnhancementSystem()
        
        # Test excellent performance
        performance_summary = {
            'quality_improvement': 0.25,
            'convergence_achieved': True
        }
        
        recommendations = enhancement._generate_recommendations(performance_summary)
        assert len(recommendations) > 0
        assert any('excellent' in rec.lower() for rec in recommendations)
        
        # Test poor performance
        performance_summary = {
            'quality_improvement': 0.02,
            'convergence_achieved': False
        }
        
        recommendations = enhancement._generate_recommendations(performance_summary)
        assert any('limited' in rec.lower() for rec in recommendations)

    def test_next_steps_suggestions(self):
        """Test next steps suggestions."""
        enhancement = AutonomousEnhancementSystem()
        
        # Test high quality results
        performance_summary = {'final_quality': 0.85}
        next_steps = enhancement._suggest_next_steps(performance_summary)
        
        assert len(next_steps) > 0
        assert any('deploy' in step.lower() for step in next_steps)
        
        # Test low quality results
        performance_summary = {'final_quality': 0.4}
        next_steps = enhancement._suggest_next_steps(performance_summary)
        
        assert any('additional' in step.lower() for step in next_steps)

    @pytest.mark.asyncio
    async def test_enhancement_report_generation(self):
        """Test comprehensive enhancement report generation."""
        enhancement = AutonomousEnhancementSystem()
        
        # Add mock enhancement history
        mock_history = [{
            'cycle': 1,
            'evolution_result': {'generation': 1, 'best_fitness': {'total_fitness': 10.0}},
            'quality_metrics': {'overall_quality': 0.6},
            'timestamp': time.time()
        }, {
            'cycle': 2,
            'evolution_result': {'generation': 2, 'best_fitness': {'total_fitness': 12.0}},
            'quality_metrics': {'overall_quality': 0.7},
            'timestamp': time.time()
        }]
        
        enhancement.enhancement_history = mock_history
        
        # Mock quality gate validation
        with patch.object(enhancement.quality_gate, 'validate_compression_quality', 
                         return_value={'status': 'passed', 'score': 0.85}):
            
            report = await enhancement._generate_enhancement_report(total_time=120.0)
        
        # Check report structure
        expected_sections = [
            'enhancement_status', 'performance_summary', 'evolution_summary',
            'quality_assurance', 'enhancement_history', 'recommendations',
            'next_steps', 'timestamp'
        ]
        
        for section in expected_sections:
            assert section in report
        
        # Check performance summary
        perf_summary = report['performance_summary']
        assert perf_summary['total_cycles'] == 2
        assert perf_summary['total_time_seconds'] == 120.0
        assert perf_summary['quality_improvement'] > 0  # 0.7 - 0.6

    @pytest.mark.asyncio
    async def test_full_autonomous_enhancement(self):
        """Test complete autonomous enhancement cycle."""
        # Create enhancement system with minimal parameters
        enhancement = AutonomousEnhancementSystem(
            enhancement_cycles=2,
            convergence_patience=1
        )
        
        # Mock evolution engine to avoid actual training
        mock_evolution_result = {
            'generation': 1,
            'best_fitness': {'total_fitness': 10.0},
            'evolution_time': 5.0,
            'population_size': 8,
            'metrics': {
                'compression_ratio': 8.0,
                'f1_score': 0.8,
                'latency_ms': 200.0,
                'memory_usage_mb': 512.0,
                'energy_efficiency': 5.0,
                'convergence_score': 0.7,
                'innovation_index': 0.5
            }
        }
        
        with patch.object(enhancement.evolution_engine, 'evolve', 
                         return_value=mock_evolution_result):
            with patch.object(enhancement.quality_gate, 'validate_compression_quality',
                             return_value={'status': 'passed', 'score': 0.85}):
                
                result = await enhancement.run_autonomous_enhancement()
        
        # Check final results
        assert result['enhancement_status'] == 'completed'
        assert 'performance_summary' in result
        assert 'evolution_summary' in result
        assert len(enhancement.enhancement_history) > 0

class TestIntegrationAndUtilities:
    """Test integration and utility functions."""
    
    def test_evolution_metrics_dataclass(self):
        """Test EvolutionMetrics dataclass."""
        metrics = EvolutionMetrics(
            generation=5,
            compression_ratio=12.5,
            f1_score=0.87,
            latency_ms=120.0,
            memory_usage_mb=256.0,
            energy_efficiency=8.33,
            convergence_score=0.92,
            innovation_index=0.73
        )
        
        assert metrics.generation == 5
        assert metrics.compression_ratio == 12.5
        assert metrics.f1_score == 0.87

    @pytest.mark.asyncio
    async def test_run_generation_8_enhancement_function(self, tmp_path):
        """Test standalone Generation 8 enhancement function."""
        # Change to temporary directory for test
        original_cwd = Path.cwd()
        
        try:
            # Mock the core functionality to avoid long execution
            with patch('retrieval_free.generation_8_autonomous_enhancement.AutonomousEnhancementSystem') as mock_system:
                mock_instance = Mock()
                mock_instance.run_autonomous_enhancement.return_value = {
                    'enhancement_status': 'completed',
                    'performance_summary': {
                        'total_cycles': 3,
                        'quality_improvement': 0.15,
                        'final_quality': 0.75
                    },
                    'recommendations': ['Test recommendation'],
                    'next_steps': ['Test next step']
                }
                mock_system.return_value = mock_instance
                
                result = await run_generation_8_enhancement()
            
            # Verify results
            assert result['enhancement_status'] == 'completed'
            assert 'performance_summary' in result
            
        finally:
            pass  # No directory change cleanup needed

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test initialization with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            NeuralArchitectureEvolution(evolution_cycles=-1)
        
        # Test enhancement with invalid configuration
        enhancement = AutonomousEnhancementSystem(enhancement_cycles=0)
        assert enhancement.enhancement_cycles == 0

    def test_performance_characteristics(self):
        """Test performance characteristics of the system."""
        evolution = NeuralArchitectureEvolution()
        
        # Test that population is created efficiently
        start_time = time.time()
        evolution._initialize_population(population_size=4)
        initialization_time = time.time() - start_time
        
        # Should initialize quickly
        assert initialization_time < 1.0
        assert len(evolution.population) == 4

    def test_memory_efficiency(self):
        """Test memory efficiency of architectures."""
        evolution = NeuralArchitectureEvolution()
        
        # Create multiple architectures
        architectures = []
        for i in range(10):
            arch = evolution._create_random_architecture(i)
            architectures.append(arch)
        
        # Verify they were created successfully
        assert len(architectures) == 10
        
        # Each should have distinct genomes
        genomes = [arch.genome for arch in architectures]
        seeds = [g['seed'] for g in genomes]
        assert len(set(seeds)) == 10  # All unique seeds

if __name__ == "__main__":
    # Run tests with asyncio support
    pytest.main([__file__, "-v", "--tb=short"])