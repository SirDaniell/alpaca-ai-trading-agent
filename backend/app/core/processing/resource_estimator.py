"""
Resource Estimation and Memory Constraint System

Provides memory safety for parallel training:
- SystemResourceEstimator: Get current system resources (RAM, CPU)
- DatasetMemoryEstimator: Estimate per-dataset memory needs
- ModelMemoryEstimator: Estimate per-model memory needs
- ParallelTrainingConstraintCalculator: Calculate safe parallelism level
- MemoryMonitor: Track memory during training
"""

import os
import psutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """Current system resource snapshot"""
    total_memory_gb: float
    available_memory_gb: float
    cpu_count: int
    swap_available_gb: float
    memory_percent_used: float
    
    def safe_parallel_memory_gb(self, max_percent: float = 0.85) -> float:
        """Maximum memory available for parallel training"""
        return self.available_memory_gb * (max_percent / 100.0)


@dataclass
class JobMemoryEstimate:
    """Memory requirements for a single training job"""
    job_id: str
    dataset_size_mb: float
    model_size_mb: float
    preprocessing_overhead_mb: float
    total_required_mb: float
    
    def to_gb(self) -> float:
        return self.total_required_mb / 1024.0


@dataclass
class ParallelTrainingConstraints:
    """Calculated constraints for parallel training"""
    max_concurrent_jobs: int
    max_workers: int
    recommended_workers: int
    total_dataset_memory_needed_gb: float
    estimated_runtime_seconds: int
    is_feasible: bool
    warning_messages: List[str]
    suggestions: List[str]


class SystemResourceEstimator:
    """Get current system resource availability"""
    
    @staticmethod
    def get_system_resources() -> SystemResources:
        """Snapshot current system resources"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return SystemResources(
                total_memory_gb=memory.total / (1024 ** 3),
                available_memory_gb=memory.available / (1024 ** 3),
                cpu_count=os.cpu_count() or 4,
                swap_available_gb=swap.free / (1024 ** 3),
                memory_percent_used=memory.percent
            )
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            # Fallback for testing
            return SystemResources(
                total_memory_gb=16.0,
                available_memory_gb=12.0,
                cpu_count=4,
                swap_available_gb=4.0,
                memory_percent_used=25.0
            )
    
    @staticmethod
    def check_memory_pressure() -> Tuple[bool, str]:
        """
        Check if system memory pressure is too high
        
        Returns: (is_critical, message)
        """
        resources = SystemResourceEstimator.get_system_resources()
        
        if resources.memory_percent_used > 90:
            return True, f"CRITICAL: System memory {resources.memory_percent_used:.0f}% used, cannot start new jobs"
        elif resources.memory_percent_used > 80:
            return True, f"HIGH: System memory {resources.memory_percent_used:.0f}% used, reducing parallelism"
        else:
            return False, "OK: Memory pressure nominal"


class DatasetMemoryEstimator:
    """Estimate memory needed for datasets"""
    
    @staticmethod
    def estimate_dataset_size(
        n_samples: int,
        n_features: int,
        dtype_bytes: int = 4  # float32
    ) -> float:
        """
        Estimate single array size in MB
        
        Formula: (n_samples × n_features × dtype_bytes) / (1024 × 1024)
        """
        if n_samples <= 0 or n_features <= 0:
            return 0.0
        return (n_samples * n_features * dtype_bytes) / (1024 * 1024)
    
    @staticmethod
    def estimate_job_memory(
        dataset_id: str,
        n_samples: int,
        n_features: int,
        n_targets: int = 1,
        train_split: float = 0.8
    ) -> JobMemoryEstimate:
        """
        Estimate total memory for training job
        
        Includes:
        - X_train (training features)
        - y_train (training labels)
        - X_val (validation features)
        - y_val (validation labels)
        - Preprocessing buffers (2x multiplier)
        """
        X_train_mb = DatasetMemoryEstimator.estimate_dataset_size(
            n_samples=int(n_samples * train_split),
            n_features=n_features
        )
        
        y_train_mb = DatasetMemoryEstimator.estimate_dataset_size(
            n_samples=int(n_samples * train_split),
            n_features=n_targets
        )
        
        # Validation set
        X_val_mb = DatasetMemoryEstimator.estimate_dataset_size(
            n_samples=int(n_samples * (1 - train_split)),
            n_features=n_features
        )
        
        y_val_mb = DatasetMemoryEstimator.estimate_dataset_size(
            n_samples=int(n_samples * (1 - train_split)),
            n_features=n_targets
        )
        
        dataset_total_mb = X_train_mb + y_train_mb + X_val_mb + y_val_mb
        
        # Add 2x for preprocessing, augmentation, intermediate tensors
        preprocessing_overhead_mb = dataset_total_mb * 2.0
        
        return JobMemoryEstimate(
            job_id=dataset_id,
            dataset_size_mb=dataset_total_mb,
            model_size_mb=0,  # Set separately if needed
            preprocessing_overhead_mb=preprocessing_overhead_mb,
            total_required_mb=dataset_total_mb + preprocessing_overhead_mb
        )


class ModelMemoryEstimator:
    """Estimate memory needed for models"""
    
    @staticmethod
    def estimate_model_size_from_layers(layers_config: Optional[Dict]) -> float:
        """
        Estimate model memory from layer configuration
        
        Returns size in MB including weights + gradients + activations
        """
        if not layers_config:
            # Default conservative estimate for small models
            return 100.0  # MB
        
        # Simplified estimation
        estimated_params = 0
        
        try:
            for layer_name, layer_config in layers_config.items():
                if isinstance(layer_config, dict):
                    if 'units' in layer_config and 'input_shape' in layer_config:
                        # Dense layer: params = input × output + bias
                        estimated_params += (
                            layer_config['input_shape'] * layer_config['units'] + 
                            layer_config['units']
                        )
        except Exception as e:
            logger.warning(f"Error estimating model size from layers: {e}")
            return 100.0
        
        # Convert to MB: params × 4 bytes for float32, × 2 for gradients
        model_weights_mb = max(100.0, (estimated_params * 4 * 2) / (1024 * 1024))
        
        # Add activation memory (typically bulk of inference memory)
        activation_memory_mb = model_weights_mb * 0.5  # Conservative
        
        return model_weights_mb + activation_memory_mb


class ParallelTrainingConstraintCalculator:
    """Calculate safe parallelism constraints"""
    
    @staticmethod
    def calculate_constraints(
        jobs: List[Dict[str, Any]],
        enable_memory_safety: bool = True,
        max_memory_percent: float = 85.0
    ) -> ParallelTrainingConstraints:
        """
        Determine optimal parallel training configuration
        
        Returns constraints object with max_workers and reasoning
        """
        resources = SystemResourceEstimator.get_system_resources()
        warnings = []
        suggestions = []
        
        # Check current memory pressure
        is_critical, pressure_msg = SystemResourceEstimator.check_memory_pressure()
        if is_critical:
            warnings.append(pressure_msg)
        
        # Estimate memory needs for all jobs
        total_dataset_memory_mb = 0.0
        job_estimates = []
        
        for job in jobs:
            # Estimate from job configuration
            est = DatasetMemoryEstimator.estimate_job_memory(
                dataset_id=job.get('dataset_id', f'job_{len(job_estimates)}'),
                n_samples=job.get('estimated_samples', 10000),
                n_features=job.get('estimated_features', 100),
                n_targets=job.get('estimated_targets', 1)
            )
            job_estimates.append(est)
            total_dataset_memory_mb += est.total_required_mb
        
        safe_memory_budget_gb = (
            resources.available_memory_gb * (max_memory_percent / 100.0)
        )
        total_dataset_memory_gb = total_dataset_memory_mb / 1024.0
        
        # Calculate max concurrent jobs based on memory
        if total_dataset_memory_mb > 0 and job_estimates:
            avg_job_memory_mb = total_dataset_memory_mb / len(job_estimates)
            max_concurrent_jobs = max(
                1,
                int((safe_memory_budget_gb * 1024.0) / avg_job_memory_mb)
            )
        else:
            max_concurrent_jobs = len(jobs)
        
        # Calculate max workers
        max_workers = min(
            resources.cpu_count - 2,  # Reserve 2 cores for OS/system
            max_concurrent_jobs,      # Memory constraint
            len(jobs)                 # Can't exceed number of jobs
        )
        
        max_workers = max(1, max_workers)  # At least 1
        
        # Recommended workers: be conservative (50% of max)
        recommended_workers = max(1, max_workers // 2) if enable_memory_safety else max_workers
        
        # Feasibility check
        is_feasible = (
            not is_critical and
            max_workers >= 1 and
            total_dataset_memory_gb < safe_memory_budget_gb
        )
        
        # Generate suggestions
        if max_workers < len(jobs):
            suggestions.append(
                f"Only {max_workers} parallel workers possible for {len(jobs)} jobs. "
                f"Remaining jobs will queue."
            )
        
        if total_dataset_memory_gb > safe_memory_budget_gb * 0.9:
            suggestions.append(
                f"Total dataset memory ({total_dataset_memory_gb:.1f}GB) is "
                f"{(total_dataset_memory_gb/safe_memory_budget_gb*100):.0f}% of budget. "
                f"Consider training fewer models in parallel."
            )
        
        if resources.memory_percent_used > 70:
            suggestions.append(
                f"System memory already {resources.memory_percent_used:.0f}% used. "
                f"Close other applications to improve performance."
            )
        
        # Rough time estimate: jobs / workers * 100s per job
        estimated_runtime = max(100, int((len(jobs) / max(max_workers, 1)) * 100))
        
        return ParallelTrainingConstraints(
            max_concurrent_jobs=max_concurrent_jobs,
            max_workers=max_workers,
            recommended_workers=recommended_workers,
            total_dataset_memory_needed_gb=total_dataset_memory_gb,
            estimated_runtime_seconds=estimated_runtime,
            is_feasible=is_feasible,
            warning_messages=warnings,
            suggestions=suggestions
        )


class MemoryMonitor:
    """Monitor memory usage during training"""
    
    def __init__(self, max_memory_percent: float = 90.0):
        self.max_memory_percent = max_memory_percent
        self.baseline_memory = psutil.virtual_memory().percent
    
    def check_safe_to_continue(self) -> Tuple[bool, float]:
        """
        Check if memory usage is safe to continue training
        
        Returns: (is_safe, current_memory_percent)
        """
        try:
            current_memory = psutil.virtual_memory()
            memory_percent = current_memory.percent
            safe = memory_percent < self.max_memory_percent
            return safe, memory_percent
        except Exception as e:
            logger.warning(f"Failed to check memory: {e}")
            return True, 50.0  # Assume safe if check fails
    
    def get_pressure_level(self) -> str:
        """Get human-readable memory pressure level"""
        _, percent = self.check_safe_to_continue()
        
        if percent < 60:
            return "LOW"
        elif percent < 75:
            return "MEDIUM"
        elif percent < 85:
            return "HIGH"
        else:
            return "CRITICAL"
