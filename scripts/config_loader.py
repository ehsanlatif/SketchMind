"""
Configuration loader for SketchMind Multi-Agent System.

This module handles loading and validating configuration from YAML files,
environment variables, and provides structured access to all settings.
"""

import os
import yaml
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration object for SketchMind pipeline."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        self.task_id = config_dict.get('task_id', 'default_task')
        
        # Model configuration
        model_config = config_dict.get('model', {})
        self.model_type = model_config.get('type', 'gpt')
        
        # GPT configuration (OpenAI) - only used when model_type = gpt
        gpt_config = config_dict.get('gpt', {})
        self.gpt_model_name = gpt_config.get('name', 'gpt-4o')
        self.gpt_max_tokens = gpt_config.get('max_tokens', 1500)
        
        # Llama4 configuration (OpenRouter) - only used when model_type = llama4
        llama4_config = config_dict.get('llama4', {})
        self.llama4_model_name = llama4_config.get('model_name', 'meta-llama/llama-4-maverick:free')
        self.llama4_api_base = llama4_config.get('api_base', 'https://openrouter.ai/api/v1')
        self.llama4_max_tokens = llama4_config.get('max_tokens', 1500)
        
        # Question and rubric
        question_config = config_dict.get('question', {})
        self.question_text = self._load_text_or_file(question_config)
        
        rubric_config = config_dict.get('rubric', {})
        self.rubric_text = self._load_text_or_file(rubric_config)
        
        # Paths
        paths_config = config_dict.get('paths', {})
        self.data_dir = paths_config.get('data_dir', 'data')
        self.golden_standard_images = paths_config.get('golden_standard_images', [])
        
        # Output directories with task_id interpolation
        self.output_dir = paths_config.get('output_dir', 'outputs')
        self.logs_dir = self._interpolate_path(paths_config.get('logs_dir', 'outputs/{task_id}/logs'))
        self.cache_dir = self._interpolate_path(paths_config.get('cache_dir', 'outputs/{task_id}/cache'))
        self.results_dir = self._interpolate_path(paths_config.get('results_dir', 'outputs/{task_id}/results'))
        
        # Agent configuration
        agents_config = config_dict.get('agents', {})
        self.max_retry_attempts = agents_config.get('max_retry_attempts', 3)
        self.score_threshold = agents_config.get('score_threshold', 0.75)
        
        # Logging configuration
        logging_config = config_dict.get('logging', {})
        self.log_level = logging_config.get('level', 'INFO')
        self.log_format = logging_config.get('format', '%(asctime)s [%(levelname)s] %(message)s')
        
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    def _load_text_or_file(self, config: Dict[str, Any]) -> str:
        """Load text from config directly or from a file path."""
        if 'text' in config:
            return config['text']
        elif 'file' in config:
            file_path = config['file']
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"Text file not found: {file_path}")
        else:
            return ""
    
    def _interpolate_path(self, path: str) -> str:
        """Replace {task_id} placeholder in paths."""
        return path.replace('{task_id}', self.task_id)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        Returns empty list if configuration is valid.
        """
        errors = []
        
        # Validate model type
        if self.model_type not in ['gpt', 'llama4']:
            errors.append(f"Invalid model type: {self.model_type}. Must be 'gpt' or 'llama4'")
        
        # Validate API key for GPT models
        if self.model_type == 'gpt' and not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable is required for GPT models")
        
        # Validate API key for Llama4 models
        if self.model_type == 'llama4' and not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY environment variable is required for Llama4 models")
        
        # Validate question and rubric
        if not self.question_text:
            errors.append("Question text is required")
        
        if not self.rubric_text:
            errors.append("Rubric text is required")
        
        # Validate golden standard images
        if not self.golden_standard_images:
            errors.append("At least one golden standard image is required")
        else:
            for img_path in self.golden_standard_images:
                if not os.path.exists(img_path):
                    errors.append(f"Golden standard image not found: {img_path}")
        
        return errors
    
    def create_output_directories(self):
        """Create necessary output directories if they don't exist."""
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_log_file_path(self, prefix: str = "agent_logs") -> str:
        """Generate a timestamped log file path."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = self.model_type.upper()
        return os.path.join(self.logs_dir, f"{prefix}_{model_suffix}_{timestamp}.log")
    
    def get_cache_file_path(self, filename: str) -> str:
        """Get path for cache file."""
        return os.path.join(self.cache_dir, filename)
    
    def get_results_file_path(self, filename: str) -> str:
        """Get path for results file."""
        return os.path.join(self.results_dir, filename)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with validated settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_msg)
    
    # Create output directories
    config.create_output_directories()
    
    return config


def load_config_dict(config_path: str) -> Dict[str, Any]:
    """
    Load raw configuration dictionary from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


