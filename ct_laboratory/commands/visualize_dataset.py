"""
Visualize Dataset Command
Visualizes a specified dataset from the GMI package.
"""

import click
from typing import Optional, Dict, Any
import os
from pathlib import Path

# Try to import optional dependencies
try:
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some dependencies are missing: {e}")
    print("   Please install required packages: pip install torch matplotlib numpy")
    DEPENDENCIES_AVAILABLE = False

# Import dataset classes
from gmi.datasets.mnist import MNIST
from gmi.datasets.medmnist import MedMNIST
from gmi.datasets.core import GMI_Dataset, GeneralPurposeDataset

# Support both MNIST and MedMNIST variants
KNOWN_DATASETS = {
    'MNIST': {
        'class': MNIST,
        'description': 'MNIST handwritten digits dataset',
        'params': {'train': True, 'download': True}
    }
}

# MedMNIST variants - these will be handled dynamically
MEDMNIST_VARIANTS = [
    'PathMNIST', 'ChestMNIST', 'DermaMNIST', 'OCTMNIST', 'PneumoniaMNIST',
    'RetinaMNIST', 'BreastMNIST', 'BloodMNIST', 'TissueMNIST', 'OrganAMNIST',
    'OrganCMNIST', 'OrganSMNIST', 'OrganMNIST3D', 'NoduleMNIST3D',
    'AdrenalMNIST3D', 'FractureMNIST3D', 'VesselMNIST3D', 'SynapseMNIST3D'
]

# Valid MedMNIST sizes and splits
MEDMNIST_SIZES = [28, 64, 128, 224]
MEDMNIST_SPLITS = ['train', 'val', 'test']

def parse_medmnist_name(dataset_name: str) -> tuple:
    """
    Parse MedMNIST dataset name to extract dataset, size, and split.
    
    Supports formats:
    - Simple: 'chestmnist' -> (ChestMNIST, 28, 'train')
    - Full: 'ChestMNIST_64_val' -> (ChestMNIST, 64, 'val')
    
    Args:
        dataset_name (str): Dataset name to parse
        
    Returns:
        tuple: (dataset_name, size, split)
        
    Raises:
        ValueError: If name format is invalid or values are not in allowed lists
    """
    # Normalize to lowercase for comparison
    name_lower = dataset_name.lower().strip()
    
    # Check if it's a simple name (just the dataset)
    for variant in MEDMNIST_VARIANTS:
        if name_lower == variant.lower():
            return variant, 28, 'train'  # Defaults
    
    # Check if it's in NAME_SIZE_SPLIT format
    parts = dataset_name.split('_')
    if len(parts) == 3:
        name_part, size_part, split_part = parts
        
        # Validate dataset name
        name_found = None
        for variant in MEDMNIST_VARIANTS:
            if name_part.lower() == variant.lower():
                name_found = variant
                break
        
        if not name_found:
            raise ValueError(f"Unknown MedMNIST dataset: {name_part}. Available: {MEDMNIST_VARIANTS}")
        
        # Validate size
        try:
            size = int(size_part)
            if size not in MEDMNIST_SIZES:
                raise ValueError(f"Invalid size {size}. Allowed sizes: {MEDMNIST_SIZES}")
        except ValueError:
            raise ValueError(f"Invalid size format: {size_part}. Must be one of {MEDMNIST_SIZES}")
        
        # Validate split
        split_lower = split_part.lower()
        if split_lower not in [s.lower() for s in MEDMNIST_SPLITS]:
            raise ValueError(f"Invalid split {split_part}. Allowed splits: {MEDMNIST_SPLITS}")
        
        # Find the correct case for split
        split = next(s for s in MEDMNIST_SPLITS if s.lower() == split_lower)
        
        return name_found, size, split
    
    # If we get here, the format is not recognized
    raise ValueError(f"Invalid MedMNIST name format: {dataset_name}. "
                    f"Use 'datasetname' or 'DatasetName_size_split' format")

# Populate KNOWN_DATASETS with all MedMNIST variants
for variant in MEDMNIST_VARIANTS:
    # Add simple name (case-insensitive)
    KNOWN_DATASETS[variant.lower()] = {
        'class': MedMNIST,
        'description': f'{variant} dataset',
        'params': {'dataset_name': variant, 'split': 'train', 'size': 28, 'download': True}
    }
    
    # Add all size/split combinations
    for size in MEDMNIST_SIZES:
        for split in MEDMNIST_SPLITS:
            key = f"{variant.lower()}_{size}_{split}"
            KNOWN_DATASETS[key] = {
                'class': MedMNIST,
                'description': f'{variant} dataset ({size}x{size}, {split})',
                'params': {'dataset_name': variant, 'split': split, 'size': size, 'download': True}
            }

def get_dataset_path(dataset_name: str) -> Path:
    """
    Get the dataset path based on the main.py location.
    Assumes main.py is in gmi_base, so dataset path is gmi_base/gmi_data/datasets/{dataset_name}/
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        Path to the dataset directory
    """
    # Get the directory containing main.py (should be gmi_base)
    main_py_path = Path(__file__).parent.parent.parent / 'main.py'
    gmi_base_path = main_py_path.parent
    
    # Construct dataset path
    dataset_path = gmi_base_path / 'gmi_data' / 'datasets' / dataset_name
    
    return dataset_path

def load_dataset(dataset_name: str) -> GMI_Dataset:
    """
    Load a dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load
        
    Returns:
        GMI_Dataset object
    """
    # Normalize dataset name for case-insensitive matching
    normalized_name = dataset_name.strip()
    
    # Check if it's a simple name (just the dataset) and convert to full format
    if '_' not in normalized_name:
        # Try to find a matching MedMNIST variant and convert to default format
        for variant in MEDMNIST_VARIANTS:
            if normalized_name.lower() == variant.lower():
                normalized_name = f"{variant.lower()}_28_train"
                print(f"🔄 Converting '{dataset_name}' to '{normalized_name}' (default size=28, split=train)")
                break
    
    # Check if it's a known dataset (case-insensitive)
    if normalized_name.lower() in [k.lower() for k in KNOWN_DATASETS.keys()]:
        # Find the correct case from KNOWN_DATASETS
        correct_key = next(k for k in KNOWN_DATASETS.keys() if k.lower() == normalized_name.lower())
        try:
            dataset_info = KNOWN_DATASETS[correct_key]
            dataset_class = dataset_info['class']
            params = dataset_info['params']
            
            print(f"🔄 Loading {correct_key} dataset...")
            dataset = dataset_class(**params)
            print(f"✅ Successfully loaded {correct_key} dataset")
            
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{correct_key}': {str(e)}")
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(KNOWN_DATASETS.keys())}")


def visualize_dataset(dataset_name: str):
    """
    Visualize a dataset with the given name.
    
    Args:
        dataset_name (str): Name of the dataset to visualize
    """
    if not DEPENDENCIES_AVAILABLE:
        raise RuntimeError("Required dependencies (torch, matplotlib, numpy) are not available")
    
    print(f"🎯 Starting visualization for dataset: {dataset_name}")
    print("=" * 60)
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        print(f"📊 Loaded {dataset_name} as GMI dataset")
        
        # Show dataset statistics
        print("\n" + "=" * 60)
        show_dataset_statistics(dataset, dataset_name)
        
        # Create output directory
        output_dir = Path('./gmi_data/outputs/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{dataset_name}_visualization.png'
        
        # Visualize using the dataset's visualize method
        print("\n" + "=" * 60)
        print(f"📊 Visualizing {dataset_name} using dataset.visualize() method...")
        
        # Get the dataset class to determine if it supports num_samples_per_class
        dataset_class = type(dataset)
        
        # Check if this is a MedMNIST dataset (or any dataset that supports num_samples_per_class)
        # We can check by looking at the class name or by trying to call with the parameter
        try:
            # Try to call visualize with num_samples_per_class parameter
            dataset.visualize(save_path=str(output_path), num_samples_per_class=10)
        except TypeError:
            # If that fails, try without the parameter
            dataset.visualize(save_path=str(output_path))
        
        print("\n" + "=" * 60)
        print(f"✅ Visualization complete for {dataset_name}")
        print(f"💾 Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise

def show_dataset_statistics(dataset: GMI_Dataset, dataset_name: str):
    """
    Display dataset statistics.
    
    Args:
        dataset: GMI_Dataset object
        dataset_name (str): Name of the dataset
    """
    print(f"📈 Computing statistics for {dataset_name}...")
    
    # Basic statistics
    dataset_size = len(dataset)
    print(f"📊 Dataset size: {dataset_size:,} samples")
    
    # Get a few samples to analyze
    sample_data = []
    sample_labels = []
    
    num_samples_to_analyze = min(1000, dataset_size)
    for i in range(num_samples_to_analyze):
        try:
            data = dataset[i]
            if isinstance(data, tuple):
                sample, label = data
                sample_labels.append(label)
            else:
                sample = data
                sample_labels.append(None)
            sample_data.append(sample)
        except Exception:
            continue
    
    if not sample_data:
        print("⚠️  Could not analyze dataset statistics")
        return
    
    # Convert to tensor
    if not isinstance(sample_data[0], torch.Tensor):
        sample_data = [torch.tensor(sample) for sample in sample_data]
    
    samples_tensor = torch.stack(sample_data)
    
    # Image statistics
    print(f"🖼️  Image shape: {samples_tensor.shape[1:]}")
    print(f"📏 Data type: {samples_tensor.dtype}")
    
    if samples_tensor.dim() >= 3:
        print(f"📊 Value range: [{samples_tensor.min():.3f}, {samples_tensor.max():.3f}]")
        print(f"📊 Mean: {samples_tensor.mean():.3f}")
        print(f"📊 Std: {samples_tensor.std():.3f}")
    
    # Label statistics (if available)
    if any(label is not None for label in sample_labels):
        valid_labels = [label for label in sample_labels if label is not None]
        if valid_labels:
            if isinstance(valid_labels[0], torch.Tensor):
                labels_tensor = torch.stack(valid_labels)
            else:
                labels_tensor = torch.tensor(valid_labels)
            
            unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
            print(f"🏷️  Number of unique labels: {len(unique_labels)}")
            print(f"🏷️  Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")

def main():
    """
    Main entry point for visualize_dataset command.
    This function is kept for backward compatibility and direct module execution.
    """
    import sys
    
    # Simple argument parsing for direct execution
    if len(sys.argv) < 2:
        print("Error: Dataset name required")
        print("Usage: python -m gmi.commands.visualize_dataset <dataset_name>")
        print(f"Available datasets: {list(KNOWN_DATASETS.keys()) + MEDMNIST_VARIANTS}")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    visualize_dataset(dataset_name)

if __name__ == "__main__":
    main() 