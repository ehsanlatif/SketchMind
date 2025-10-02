
"""
SketchMind: Multi-Agent Sketch Reasoning Graph Evaluation System
Unified entry point for running the pipeline with either GPT or Llama4 models.

Usage:
    python run.py --config config/task_config.yaml --model-type gpt --student-image path/to/image.jpg
    python run.py --config config/example_task.yaml --model-type llama4 --student-image data/Task42_R1_1/student_images/1_40185.jpg
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.config_loader import load_config
from PIL import Image


def setup_logging(config):
    """Configure logging based on config settings."""
    log_file = config.get_log_file_path()
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), Image.open(image_path).size


def run_gpt_pipeline(config, student_image_path, logger):
    """Run GPT-based pipeline."""
    from scripts.GPT_SRG_MAS import run_pipeline, encode_image_to_base64 as encode_gpt
    
    logger.info("=" * 80)
    logger.info("Starting GPT-based Multi-Agent Sketch Evaluation Pipeline")
    logger.info("=" * 80)
    
    # Load student image
    logger.info(f"Loading student image: {student_image_path}")
    encoded_student_image, image_size = encode_gpt(student_image_path)
    
    # Load golden standard images
    logger.info("Loading golden standard images...")
    golden_standard_images = []
    for img_path in config.golden_standard_images:
        if not os.path.exists(img_path):
            logger.error(f"Golden standard image not found: {img_path}")
            sys.exit(1)
        golden_standard_images.append(encode_gpt(img_path))
    
    logger.info(f"Loaded {len(golden_standard_images)} golden standard images")
    
    # Run pipeline
    feedback, result = run_pipeline(
        question_id=config.task_id,
        question_text=config.question_text,
        rubric_text=config.rubric_text,
        encoded_student_image=encoded_student_image,
        image_size=image_size,
        golden_standard_images=golden_standard_images,
        config=config
    )
    
    return feedback, result


def run_llama4_pipeline(config, student_image_path, logger):
    """Run Llama4-based pipeline."""
    from scripts.Llama4_SRG_MAS import run_pipeline, encode_image_to_base64 as encode_llama4
    
    logger.info("=" * 80)
    logger.info("Starting Llama4-based Multi-Agent Sketch Evaluation Pipeline")
    logger.info("=" * 80)
    
    # Load student image
    logger.info(f"Loading student image: {student_image_path}")
    encoded_student_image, image_size = encode_llama4(student_image_path)
    
    # Load golden standard images
    logger.info("Loading golden standard images...")
    golden_standard_images = []
    for img_path in config.golden_standard_images:
        if not os.path.exists(img_path):
            logger.error(f"Golden standard image not found: {img_path}")
            sys.exit(1)
        golden_standard_images.append(encode_llama4(img_path))
    
    logger.info(f"Loaded {len(golden_standard_images)} golden standard images")
    
    # Run pipeline
    feedback, result = run_pipeline(
        question_id=config.task_id,
        question_text=config.question_text,
        rubric_text=config.rubric_text,
        encoded_student_image=encoded_student_image,
        image_size=image_size,
        golden_standard_images=golden_standard_images,
        config=config
    )
    
    return feedback, result


def save_feedback_to_file(feedback, config, student_image_name):
    """Save feedback text to a results file."""
    feedback_filename = f"feedback_{student_image_name}.txt"
    feedback_path = config.get_results_file_path(feedback_filename)
    
    with open(feedback_path, 'w', encoding='utf-8') as f:
        f.write(feedback)
    
    return feedback_path


def main():
    parser = argparse.ArgumentParser(
        description="SketchMind: Multi-Agent Sketch Reasoning Graph Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GPT-4o on a student sketch
  python run.py --config config/example_task.yaml --model-type gpt --student-image data/Task42_R1_1/student_images/sketch.jpg
  
  # Run with Llama4
  python run.py --config config/example_task.yaml --model-type llama4 --student-image data/Task42_R1_1/student_images/sketch.jpg
  
  # Override model type from config file
  python run.py --config config/example_task.yaml --model-type gpt --student-image sketch.jpg
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['gpt', 'llama4'],
        help='Model type to use (gpt or llama4). Overrides config file if specified.'
    )
    
    parser.add_argument(
        '--student-image',
        type=str,
        required=True,
        help='Path to student sketch image to evaluate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--show-image',
        action='store_true',
        help='Display the output image with visual hints (if applicable)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.student_image):
        print(f"Error: Student image not found: {args.student_image}")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override model type if specified
    if args.model_type:
        config.model_type = args.model_type
        print(f"Model type overridden to: {args.model_type}")
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
        config.create_output_directories()
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info(f"Task ID: {config.task_id}")
    logger.info(f"Model Type: {config.model_type}")
    logger.info(f"Student Image: {args.student_image}")
    
    # Run appropriate pipeline
    try:
        if config.model_type == 'gpt':
            feedback, result = run_gpt_pipeline(config, args.student_image, logger)
        elif config.model_type == 'llama4':
            feedback, result = run_llama4_pipeline(config, args.student_image, logger)
        else:
            logger.error(f"Unknown model type: {config.model_type}")
            sys.exit(1)
        
        # Display results
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("FEEDBACK:")
        print("=" * 80)
        print(feedback)
        print("=" * 80)
        
        # Save feedback to file
        student_image_name = Path(args.student_image).stem
        feedback_path = save_feedback_to_file(feedback, config, student_image_name)
        print(f"\nFeedback saved to: {feedback_path}")
        
        # Handle visual output
        if isinstance(result, Image.Image):
            logger.info("Visual hint generated as PIL Image")
            if args.show_image:
                result.show()
        elif isinstance(result, str):
            print(f"Visual hint saved to: {result}")
            logger.info(f"Visual hint saved to: {result}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


