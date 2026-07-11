"""
General-purpose configuration utilities for GMI.
Provides tools to instantiate Python objects from YAML/dict configurations.
"""

import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, Union, Type, TypeVar, List

T = TypeVar('T')

def load_object_from_dict(config: Dict[str, Any], expected_type: Type[T] | None = None) -> T:
    """
    Load a Python object from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'class' and 'params' keys
        expected_type: Optional type hint for validation
        
    Returns:
        Instantiated object
        
    Raises:
        ValueError: If configuration is invalid or instantiation fails
    """
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
    
    if 'class' not in config:
        raise ValueError("Configuration missing 'class' field")
    
    class_path = config['class']
    params = config.get('params', {})
    
    # Load the class dynamically
    try:
        class_obj = _load_class(class_path)
    except Exception as e:
        raise ValueError(f"Failed to load class '{class_path}': {e}")
    
    # Validate type if expected_type is provided
    if expected_type is not None and not issubclass(class_obj, expected_type):
        raise ValueError(f"Class '{class_path}' is not a subclass of {expected_type}")
    
    # Instantiate the class with parameters
    try:
        instance = class_obj(**params)
    except Exception as e:
        raise ValueError(f"Failed to instantiate '{class_path}' with params {params}: {e}")
    
    return instance

def load_object_from_yaml(config_path: Union[str, Path], expected_type: Type[T] | None = None) -> T:
    """
    Load a Python object from a YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        expected_type: Optional type hint for validation
        
    Returns:
        Instantiated object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return load_object_from_dict(config, expected_type)

def _load_class(class_path: str):
    """
    Load a class from a string path like 'gmi.network.SimpleCNN'.
    
    Args:
        class_path: Dot-separated path to the class
        
    Returns:
        Class object
    """
    try:
        # Split the path into module and class name
        module_path, class_name = class_path.rsplit('.', 1)
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class from the module
        if not hasattr(module, class_name):
            raise AttributeError(f"Module '{module_path}' has no attribute '{class_name}'")
        
        return getattr(module, class_name)
        
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")
    except ValueError as e:
        raise ValueError(f"Invalid class path '{class_path}': {e}")

def load_components_from_dict(config: Dict[str, Any], component_types: Dict[str, type] | None = None) -> Dict[str, Any]:
    """
    Load multiple components from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with component configurations
        component_types: Optional dict mapping component names to expected types
        
    Returns:
        Dictionary of instantiated components
    """
    components = {}
    
    for component_name, component_config in config.items():
        expected_type = component_types.get(component_name) if component_types else None
        components[component_name] = load_object_from_dict(component_config, expected_type)
    
    return components

def load_components_from_yaml(config_path: Union[str, Path], component_types: Dict[str, type] | None = None) -> Dict[str, Any]:
    """
    Load multiple components from a YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        component_types: Optional dict mapping component names to expected types
        
    Returns:
        Dictionary of instantiated components
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return load_components_from_dict(config, component_types)

def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        configs: List of configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ValueError: If there are duplicate keys across configs
    """
    merged = {}
    seen_keys = set()
    
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError(f"All configs must be dictionaries, got {type(config)}")
        
        for key, value in config.items():
            if key in seen_keys:
                raise ValueError(f"Duplicate key '{key}' found in config files. This creates ambiguity.")
            seen_keys.add(key)
            merged[key] = value
    
    return merged

def load_and_merge_configs(config_paths: List[Union[str, Path]]) -> Dict[str, Any]:
    """
    Load and merge multiple YAML configuration files.
    
    Args:
        config_paths: List of paths to YAML configuration files
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If any config file is not found
        ValueError: If there are duplicate keys across configs
    """
    configs = []
    
    for config_path in config_paths:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        configs.append(config)
    
    return merge_configs(configs)

def load_components_from_multiple_configs(config_paths: List[Union[str, Path]], 
                                        component_types: Dict[str, type] | None = None) -> Dict[str, Any]:
    """
    Load components from multiple YAML configuration files.
    
    Args:
        config_paths: List of paths to YAML configuration files
        component_types: Optional dict mapping component names to expected types
        
    Returns:
        Dictionary of instantiated components
    """
    merged_config = load_and_merge_configs(config_paths)
    return load_components_from_dict(merged_config, component_types) 