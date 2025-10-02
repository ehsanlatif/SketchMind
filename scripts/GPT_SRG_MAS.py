import base64
import json
from PIL import Image
from io import BytesIO
import os
import logging
from datetime import datetime
from GPT_SRG_Agents import Agent1, Agent2, Agent3, Agent4, SRGBuilder

# Logger will be configured by the calling script
logger = logging.getLogger(__name__)

def decode_image(base64_str):
    return Image.open(BytesIO(base64.b64decode(base64_str)))

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), Image.open(image_path).size

def strip_json_wrapping(text):
    text = text.strip()
    text = text[text.find("{"):]  # Skip to first open brace
    text = text[:text.rfind("}") + 1] # Skip to last close brace

    return text.strip()

def get_valid_srg(agent, question_id=None, is_agent1=False, cache_dir="srg_cache"):
    srg_file = os.path.join(cache_dir, f"srg_{question_id}.json") if question_id else None
    revmap_file = os.path.join(cache_dir, f"reverse_map_{question_id}.json") if question_id and is_agent1 else None

    if srg_file and os.path.exists(srg_file):
        logger.info(f"Loading cached SRG from {srg_file}")
        with open(srg_file, "r") as f:
            srg_json = json.load(f)
        srg = SRGBuilder("", "")
        srg.nodes = [(node['label'], node['bloom_level']) for node in srg_json['nodes']]
        srg.edges = [(edge['from'], edge['to']) for edge in srg_json['edges']]
        if is_agent1 and revmap_file and os.path.exists(revmap_file):
            logger.info(f"Loading cached reverse mapping from {revmap_file}")
            with open(revmap_file, "r") as f:
                agent.reverse_mapping = json.load(f)
        return srg

    attempt = 0
    max_attempts = 3
    feedback = None
    while attempt < max_attempts:
        logger.info(f"Calling agent (attempt {attempt + 1}) with feedback: {feedback}")
        result_raw = agent.run(feedback) if feedback else agent.run()
        logger.info(f"Agent response received")
        logger.info(f"Agent's response raw: {result_raw}")
        result_json = json.loads(strip_json_wrapping(result_raw))

        srg_data = result_json.get("srg", result_json)
        reverse_mapping = result_json.get("reverse_mapping", {}) if is_agent1 else {}

        srg = SRGBuilder("", "")
        srg.nodes = [(node['label'], node['bloom_level']) for node in srg_data['nodes']]
        srg.edges = [(edge['from'], edge['to']) for edge in srg_data['edges']]
        if not is_agent1:
            return srg

        if srg.validate_graph():
            logger.info("Valid SRG created")
            if srg_file:
                os.makedirs(cache_dir, exist_ok=True)
                with open(srg_file, "w") as f:
                    json.dump(srg_data, f, indent=2)
            if is_agent1 and revmap_file:
                with open(revmap_file, "w") as f:
                    json.dump(reverse_mapping, f, indent=2)
                agent.reverse_mapping = reverse_mapping
            return srg

        logger.warning("Invalid SRG, retrying with feedback")
        feedback = "Please correct the SRG, ensure the graph is connected and Bloom levels follow proper order."
        attempt += 1
    logger.error("Failed to generate a valid SRG after multiple attempts")
    return srg


def run_pipeline(question_id, question_text, rubric_text, encoded_student_image, image_size, golden_standard_images, config=None):
    """
    Run the GPT-based multi-agent sketch evaluation pipeline.
    
    Args:
        question_id: Unique identifier for the task/question
        question_text: The question/task description
        rubric_text: Evaluation rubric text
        encoded_student_image: Base64-encoded student sketch image
        image_size: Tuple of (width, height) of the image
        golden_standard_images: List of base64-encoded reference images with sizes
        config: Optional Config object for advanced settings
    
    Returns:
        Tuple of (feedback_text, output_path_or_image)
    """
    # Extract cache directory from config or use default
    cache_dir = config.cache_dir if config else "outputs/cache"
    results_dir = config.results_dir if config else "outputs/results"
    
    logger.info("Running GPT pipeline...")
    logger.info("Calling Agent1 to create SRG")
    agent1 = Agent1(question_text, rubric_text, golden_standard_images)
    ref_srg = get_valid_srg(agent1, question_id, is_agent1=True, cache_dir=cache_dir)
    logger.info("Agent1 SRG: %s", ref_srg)
    logger.info("Agent1 reverse mapping: %s", agent1.reverse_mapping)
    
    logger.info("Calling Agent2 to analyze student image")
    agent2 = Agent2(encoded_student_image, ref_srg.build_graph())
    obs_srg = get_valid_srg(agent2, cache_dir=cache_dir)
    logger.info("Agent2 SRG: %s", obs_srg)
    

    logger.info("Calling Agent3 to evaluate SRGs")
    agent3 = Agent3()
    analysis = agent3.run(ref_srg.build_graph(), obs_srg.build_graph())
    logger.info("Agent3 analysis: %s", analysis)

    logger.info("Calling Agent4 to provide feedback and visual hints")
    agent4 = Agent4(agent1_output=ref_srg.build_graph(), reverse_mapping=agent1.reverse_mapping)
    feedback = agent4.run(analysis)
    logger.info("Feedback: %s", feedback)

    # Select top priority concept to visualize
    if not analysis['terminate'] and analysis['priority_fix']:
        top_concept = analysis['priority_fix'][0]
        concept_name = top_concept[0]
        visual_hint = agent4.generate_visual_hint(top_concept)
        logger.info(f"Calling GPT-4o to generate visual hint overlay {visual_hint}")
        updated_image_response =  agent4.generate_visual_overlay(encoded_student_image, concept_name, visual_hint, image_size)
        logger.info(f"Overlay image with GPT-4o hint generated")
               
        local_vars = {}
        try:
            from textwrap import dedent
            response_code = updated_image_response.strip()
            response_code = response_code[response_code.find("```python") + 10:] if "```python" in response_code else response_code
            response_code = response_code[:response_code.rfind("```")] if "```" in response_code else response_code

            exec(dedent(response_code), globals(), local_vars)

            overlay_hint = local_vars.get("overlay_hint")
            if not overlay_hint:
                raise ValueError("overlay_hint function not found in GPT response.")

            image = decode_image(encoded_student_image)
            modified_image = overlay_hint(image)
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, "visual_hint_dynamic.png")
            modified_image.save(output_path)
            logger.info(f"Overlay applied and saved at {output_path}")
            return feedback, output_path
        except Exception as e:
            logger.error(f"Failed to apply overlay: {e}")
            return None

    # fallback to plain image if no priority fix
    sketch_image = decode_image(encoded_student_image)
    logger.info("Pipeline complete - no visual hint needed")
    return feedback, sketch_image

# ------------------ Main Entry ------------------
if __name__ == "__main__":
    """
    Standalone execution mode with hardcoded values for backward compatibility.
    For modular usage, use run.py with a config file instead.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config_loader import load_config
    
    # Setup logging for standalone mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    print("=" * 60)
    print("Running GPT_SRG_MAS in standalone mode")
    print("For better modularity, use: python run.py --config config/task_config.yaml")
    print("=" * 60)
    
    # Try to load from config if available, otherwise use hardcoded values
    config_path = "config/example_task.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        question_id = config.task_id
        question_text = config.question_text
        rubric_text = config.rubric_text
        
        # Setup logging with config
        log_file = config.get_log_file_path()
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(config.log_format))
        logging.getLogger().addHandler(file_handler)
        
        # Load golden standard images
        golden_standard_images = [encode_image_to_base64(img_path) for img_path in config.golden_standard_images]
        
        # Example student image - update this path
        student_image_path = "data/Task42_R1_1/student_images/1_40185.jpg"
        if not os.path.exists(student_image_path):
            logger.error(f"Student image not found: {student_image_path}")
            logger.info("Please place student images in data/Task42_R1_1/student_images/")
            sys.exit(1)
    else:
        # Fallback to hardcoded values
        logger.warning(f"Config file not found: {config_path}")
        logger.warning("Using hardcoded values - this is not recommended for production use")
        
        config = None
        base_path = "../dataset/Task42_R1_1_drawings/"
        student_image_path = base_path + "02/1_40185.jpg"
        golden_standard_images_path = ["2_878.jpg", "2_39456.jpg", "2_28743.jpg"]
        question_id = "Task42_R1_1"
        question_text = "Shawn placed a red-coated chocolate candy into three dishes of water: one cold, one at room temperature, and one hot, to investigate how temperature influences dye diffusion. Students were instructed explicitly to illustrate and label clearly both the movement of water particles and dye particles at each temperature, including clear visual or textual indicators of particle types and motion."
        rubric_text = (
            "Rubric Element (A): Student must depict water particles and show their motion at each temperature.\n"
            "Rubric Element (B): Student must distinguish between water and dye particles visually or with labels.\n"
            "Rubric Element (C): Motion must be illustrated with directional arrows or labels showing speed.\n"
            "Proficient: Student develops a model that identifies both water and dye particles and their motion while describing that water molecules move faster at higher temperatures (and vice versa). Satisfied All Rubric Elements.\n"
            "Developing: Student develops a model that partially identifies both water and dye particles and their motion while describing that water molecules move faster at higher temperatures (and vice versa). Satisfied most but not all rubric elements.\n"
            "Beginning: Student does not develop a model that identifies both water and dye particles and their motion while describing that water molecules move faster at higher temperatures (and vice versa). Satisfied one or less rubric elements.\n"
        )
        golden_standard_images = [encode_image_to_base64(base_path + "03/" + image_path) for image_path in golden_standard_images_path]

    encoded_image, image_size = encode_image_to_base64(student_image_path)
    feedback, image_with_hints = run_pipeline(question_id, question_text, rubric_text, encoded_image, image_size, golden_standard_images, config)

    print("\n" + "=" * 60)
    print("FEEDBACK:")
    print("=" * 60)
    print(feedback)
    print("=" * 60)
    
    if isinstance(image_with_hints, Image.Image):
        image_with_hints.show()
    else:
        print(f"\nVisual overlay saved at: {image_with_hints}")