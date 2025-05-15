import os
import base64
from openai import OpenAI
import csv
import json
import pandas as pd

# Set your OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key="you api key"
)
# Define folder-to-label mapping
label_map = {
    "01": "Beginning",
    "02": "Developing",
    "03": "Proficient"
}


# Define updated system prompt requesting JSON output
system_prompt = """You are an expert evaluator for science education. You will evaluate student responses based on explicitly defined rubrics. Use rubric-based language to be extremely clear, explicit, structured, and precise in your reasoning.

Problem Context: Shawn placed a red-coated chocolate candy into three dishes of water: one cold, one at room temperature, and one hot, to investigate how temperature influences dye diffusion. Students were instructed explicitly to illustrate and label clearly both the movement of water particles and dye particles at each temperature, including clear visual or textual indicators of particle types and motion.

Carefully observe the student's drawing provided and explicitly evaluate the following criteria in detail. Provide structured and explicit reasoning for each rubric element separately:

Rubric Element (A): - Explicitly check whether the student's drawing clearly depicts water particles and explicitly illustrates their motion at the different temperatures (slow in cold water, moderate at room temperature, fastest in hot water). - Clearly state your evaluation result (Present/Absent), and explicitly justify your answer by closely describing what is visually present or missing in the drawing.

Rubric Element (B): - Explicitly evaluate if the student's drawing clearly identifies two distinct particle types—water particles and dye particles—with explicit visual differentiation or explicit textual labels or keys. - Clearly state your evaluation result (Present/Absent), and explicitly justify your reasoning by describing exactly what's visible or missing in the drawing.

Rubric Element (C): - Explicitly determine if the drawing clearly illustrates particle motion (faster/slower) using explicit annotations, such as labels, directional arrows, or textual indicators. - Clearly state your evaluation result (Present/Absent), explicitly justifying your conclusion by detailing exactly how particle motion is depicted or what is explicitly missing.

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

Be extremely precise and consistent with the format. Return only valid JSON."""

# Output CSV file
output_file = "image_classification_results.csv"
old_file = "image_classification_results_01.csv"
old_df = pd.read_csv(old_file)

# Prepare CSV
with open(output_file, mode="w", newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "Image Path",
        "True Label",
        "Classification Label",
        "Rubric A Status", "Rubric A Justification",
        "Rubric B Status", "Rubric B Justification",
        "Rubric C Status", "Rubric C Justification",
        "Overall Justification"
    ])

    # Loop through each folder and image
    for folder in ["01"]:
        label = label_map[folder]
        folder_path = os.path.join("GGLee_ModelGPT/Task42_R1_1_drawings", folder)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                print(image_path)
                if not old_df["Image Path"].str.contains(image_path).any():
                    # Open and encode image
                    with open(image_path, "rb") as image_file:
                        image_bytes = image_file.read()
                        base64_image = base64.b64encode(image_bytes).decode("utf-8")

                    # Call GPT-4 Vision API
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                ]
                            }
                        ],
                        max_tokens=1200
                    )

                    # Extract GPT response content
                    gpt_content = response.choices[0].message.content
                    print(gpt_content)

                    # Try to extract JSON from GPT response
                    try:
                        # Locate JSON block
                        json_start = gpt_content.find("{")
                        json_end = gpt_content.rfind("}") + 1
                        json_data = json.loads(gpt_content[json_start:json_end])

                        writer.writerow([
                            image_path,
                            label,
                            json_data.get("classification_label", "Unknown"),

                            json_data.get("rubric_element_A", {}).get("status", ""),
                            json_data.get("rubric_element_A", {}).get("justification", ""),

                            json_data.get("rubric_element_B", {}).get("status", ""),
                            json_data.get("rubric_element_B", {}).get("justification", ""),

                            json_data.get("rubric_element_C", {}).get("status", ""),
                            json_data.get("rubric_element_C", {}).get("justification", ""),

                            json_data.get("justification", "")
                        ])

                    except Exception as e:
                        print(f"Failed to parse JSON for {filename}: {e}")
                        writer.writerow([
                            image_path, label, "ParseError", "", "", "", "", "", "", f"Raw response: {gpt_content}"
                        ])

print("Classification complete. Results saved to:", output_file)
