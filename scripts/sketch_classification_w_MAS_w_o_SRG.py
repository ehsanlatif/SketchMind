import base64
from openai import OpenAI
import logging
import json
from datetime import datetime

client = OpenAI(
    # This is the default and can be omitted
    api_key="your api key",
)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"MAS_logs/agent_logs_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Encode image as base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# GPT-4o call utility
# ------------------ Core GPT Call ------------------
def gpt_call(history, user_text=None, image_base64=None):
    if user_text or image_base64:
        content = []
        if image_base64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
        if user_text:
            content.append({"type": "text", "text": user_text})
        history.append({"role": "user", "content": content})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
        max_tokens=1500
    )
    output = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": output})
    return output

# ------------------- Agents -------------------

# ------------------ Agent 1 ------------------
class Agent1:
    def __init__(self, question, rubric):
        self.history = [{
            "role": "system",
            "content": "You extract the required visual components from the rubric and question for a science sketch evaluation. Be precise."
        }, {
            "role": "user",
            "content": f"Question: {question}\nRubric: {rubric}"
        }]
    
    def run(self, feedback=None):
        if feedback:
            self.history.append({"role": "user", "content": f"Feedback to improve extraction:\n{feedback}"})
        return gpt_call(self.history)

# ------------------ Agent 2 ------------------
class Agent2:
    def __init__(self, image_base64):
        self.image_base64 = image_base64
        self.history = [{
            "role": "system",
            "content": "You are a sketch analysis model. Identify clearly visible components (labels, particles, arrows) from the sketch. Do not evaluate. Be accurate and clear."
        }]
    
    def run(self, feedback=None):
        if feedback:
            self.history.append({"role": "user", "content": f"Focus more accurately using this feedback:\n{feedback}"})
        return gpt_call(self.history, image_base64=self.image_base64)

# ------------------ Agent 3 ------------------
class Agent3:
    def __init__(self):
        self.history = [{
            "role": "system",
            "content": """You are a grading assistant. Compare required vs observed components and classify the proficiency level (Beginning, Developing, Proficient).
              After evaluating each rubric element separately, return your final response in the following JSON format:
                {
                "rubric_element_A": {
                    "status": "Present" or "Absent",
                    "justification": "Your justification for rubric element A."
                },
                "rubric_element_B": {
                    "status": "Present" or "Absent",
                    "justification": "Your justification for rubric element B."
                },
                "rubric_element_C": {
                    "status": "Present" or "Absent",
                    "justification": "Your justification for rubric element C."
                },
                "classification_label": "Proficient" or "Developing" or "Beginning",
                "justification": "Brief, explicit summary that explains the assigned proficiency level based on the number of Present elements."
                }

                Be extremely precise and consistent with the format. Return only valid JSON.
                """
        }]
    
    def run(self, required, observed, feedback=None):
        input_text = f"Required:\n{required}\n\nObserved:\n{observed}"
        if feedback:
            input_text += f"\n\nUse this guidance to improve your evaluation:\n{feedback}"
        response =  gpt_call(self.history, user_text=input_text)
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            return json.loads(response[json_start:json_end])
        except Exception:
            logger.warning("Agent 3 response not valid JSON.")
            return {"classification_label": None, "justification":"", "rubric_element_A": {"status": "", "justification": ""}, "rubric_element_B": {"status": "", "justification": ""}, "rubric_element_C": {"status": "", "justification": ""}}

# ------------------ Agent 4: Validator ------------------
def agent_4_validate_with_context(predicted_label, ground_truth, agent1_io, agent2_io, agent3_io):
    # system_prompt = (
    #     "You are a grading reviewer. Evaluate whether the classication by other agent is correct or not. Further, analyze rubric-wise status and justification to understand the overall classification procedure.\n"
    #     "If incorrect, provide feedback for each agent:\n"
    #     "- Agent1: Extraction of rubric requirements\n"
    #     "- Agent2: Visual analysis of the sketch\n"
    #     "- Agent3: Evaluation logic\n"
    #     "Return JSON:\n{\n  'match': true/false,\n  'feedback': {\n     'agent1': '', 'agent2': '', 'agent3': ''\n  }\n}"
    # )
    system_prompt = """You are a grading supervisor reviewing a multi-agent system that evaluates student sketch proficiency. Your task is evaluate the reason behind the incorrect prediction and help improve each agent by reviewing their individual role, input, and output.

        You are given:
        - The final predicted label and the expected ground truth.
        - The role of each agent.
        - Their most recent input/output.

        Instructions:
        1. Determine if the prediction is correct (`match: true/false`).
        2. Provide specific, constructive feedback for each agent, focusing only on their responsibility.
        3. Your response **must** be in JSON format like:

        {
        "match": true/false,
        "reasoning": "Explanation of correctness or issues",
        "feedback": {
            "agent1": "Feedback on rubric extraction",
            "agent2": "Feedback on image analysis",
            "agent3": "Feedback on evaluation reasoning"
        }
        }
        """
    user_prompt = f"""
        Final Prediction: {predicted_label}
        Ground Truth: {ground_truth}

        Agent 1 Role: Extract visual components required from rubric and question.
        Input: {agent1_io['input']}
        Output: {agent1_io['output']}

        Agent 2 Role: Analyze sketch image and extract visible components.
        Output: {agent2_io['output']}

        Agent 3 Role: Evaluate student proficiency based on Agent 1 and Agent 2 outputs.
        Input: Required = {agent3_io['input_required']} | Observed = {agent3_io['input_observed']}
        Output: {agent3_io['output']}
        """

    response = gpt_call([{"role": "system", "content": system_prompt}], user_text=user_prompt)
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception:
        logger.warning("Agent 4 response not valid JSON.")
        return {"match": False, "reasoning":"", "feedback": {"agent1": "", "agent2": "", "agent3": ""}}



# ------------------ Orchestrator ------------------
def run_reasoning_loop(image_path, question, rubric, ground_truth, max_loops=5):
    image_base64 = encode_image_to_base64(image_path)
    agent1 = Agent1(question, rubric)
    agent2 = Agent2(image_base64)
    agent3 = Agent3()

    feedback = {"agent1": None, "agent2": None, "agent3": None}

    for iteration in range(max_loops):
        logger.info(f"\n========== Iteration {iteration+1} ==========")

        # Agent 1
        required = agent1.run(feedback["agent1"])
        logger.info(f"[Agent 1] Required Components:\n{required}")

        # Agent 2
        observed = agent2.run(feedback["agent2"])
        logger.info(f"[Agent 2] Observed Components:\n{observed}")

        # Agent 3
        prediction = agent3.run(required, observed, feedback["agent3"])
        logger.info(f"[Agent 3] Prediction Output:\n{prediction}")

        # Extract label
        predicted_label = prediction.get("classification_label")
        if not predicted_label:
            logger.error("<X> Agent 3 did not return a valid label.")
            break
        elif predicted_label == ground_truth:
            logger.info(f"<O> System converged successfully! Final label: {predicted_label}")
            break

        # Agent 4: Validate
        agent1_io = {"input": question + "\n" + rubric, "output": required}
        agent2_io = {"output": observed}
        agent3_io = {
            "input_required": required,
            "input_observed": observed,
            "output": prediction
        }

        result = agent_4_validate_with_context(
            predicted_label,
            ground_truth,
            agent1_io,
            agent2_io,
            agent3_io
        )
        logger.info(f"[Agent 4] Validation: {result}")

        # if result["match"]:
        #     logger.info(f"<O> System converged successfully! Final label: {predicted_label}")
        #     break
        # else:
        logger.info("<-> Applying custom feedback to all agents.")
        feedback = result["feedback"]

# ------------------- Run Example -------------------
label_map = [ "Beginning", "Developing", "Proficient" ]

if __name__ == "__main__":
    image_path = "GGLee_ModelGPT/Task42_R1_1_drawings/02/1_40239.jpg"
    question = "Shawn placed a red-coated chocolate candy into three dishes of water: one cold, one at room temperature, and one hot, to investigate how temperature influences dye diffusion. Students were instructed explicitly to illustrate and label clearly both the movement of water particles and dye particles at each temperature, including clear visual or textual indicators of particle types and motion."
    rubric = (
        "Rubric Element (A): Student must depict water particles and show their motion at each temperature.\n"
        "Rubric Element (B): Student must distinguish between water and dye particles visually or with labels.\n"
        "Rubric Element (C): Motion must be illustrated with directional arrows or labels showing speed."
    )

    ground_truth = label_map[1] 
    run_reasoning_loop(image_path, question, rubric, ground_truth)
