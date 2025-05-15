Based on your provided scripts and updated context, here's the revised and complete `README.md` file. It now reflects the LLaMA-4 based and GPT-4o based agent pipelines implemented in the following scripts:
# SketchMind: A Multi-Agent Cognitive Framework for Scientific Sketch Assessment

SketchMind introduces a cognitively grounded, multi-agent framework for assessing scientific sketches using semantic structures known as **Sketch Reasoning Graphs (SRGs)**. Each SRG is annotated with **Bloom’s Taxonomy** levels and constructed via AI agents that collaboratively parse rubrics, analyze student sketches, evaluate conceptual understanding, and generate pedagogical feedback.

Two full multi-agent pipelines are provided:

- `SRG_MAS.py` + `SRG_Agents.py` (GPT-4o based)
- `Llama4_SRG_MAS.py` + `Llama4_SRG_Agents.py` (LLaMA-4 Maverick based)

---

## Directory Structure
.
├── graphs/            # Gold-standard SRG annotations from experts
├── scripts/           # All core scripts including agent implementations
├── srg\_cache/         # Cached SRGs and intermediate feedback outputs

---

## Scripts and Agent Descriptions

### `scripts/SRG_Agents.py` and `scripts/SRG_MAS.py` (GPT-4o)

- **Agent 1: Rubric Parser**
  - Inputs: Rubric, question text, golden-standard sketches
  - Outputs: Gold-standard SRG with Bloom levels and reverse mapping

- **Agent 2: Sketch Parser**
  - Inputs: Student sketch image, reference SRG
  - Outputs: SRG constructed from visible sketch content

- **Agent 3: SRG Evaluator**
  - Compares reference and student SRG via graph edit distance and Bloom-level mismatch
  - Outputs: Cognitive score, proficiency label, missing/inaccurate concepts

- **Agent 4: Feedback Generator**
  - Generates visual hints using LLM + PIL-based image overlays
  - Produces: Next-step sketch revision plan with Bloom-guided visual modifications

### `scripts/Llama4_SRG_Agents.py` and `scripts/Llama4_SRG_MAS.py` (LLaMA-4)

- Fully parallel implementation using the `meta-llama/llama-4-maverick` model
- Same agent structure and JSON output schema
- Adapted prompts and structure for LLaMA-4 API usage
- Image reasoning supported via multimodal interface

---

## Running the Pipeline

### Requirements

```bash
pip install -r requirements.txt
````

Dependencies include: `transformers`, `Pillow`, `networkx`, `tqdm`, `openai`, `torch`.

### Example (GPT-4o pipeline):

```bash
python scripts/SRG_MAS.py
```

### Example (LLaMA-4 pipeline):

```bash
python scripts/Llama4_SRG_MAS.py
```

Both scripts will:

1. Parse rubrics and generate gold-standard SRGs
2. Analyze student sketches to infer cognitive structure
3. Evaluate alignment with expert models
4. Generate scaffolded visual feedback and co-creation suggestions

Intermediate SRGs and feedback will be stored in `srg_cache/`.

---

## Dataset

This project uses a curated dataset of 3,500+ student-drawn scientific sketches across 6 NGSS-aligned assessment items. Each item is annotated with:

* A rubric
* 3 expert-generated gold SRGs (Beginning, Developing, Proficient)
* Corresponding student images and proficiency labels

**Note:** The full dataset is pending release. This repository includes only sample images and expert SRG examples.

---

## License

This repository is provided for academic, non-commercial use only. All rights reserved.

---

## Contact

For further questions or collaboration inquiries, please contact the authors after the anonymity period ends or open an issue in this repository.