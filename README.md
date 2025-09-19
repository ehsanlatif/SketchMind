# SketchMind 🧠✒️

<p align="center">
<a href="https://www.arxiv.org/abs/2507.22904"><img src="https://img.shields.io/badge/Paper-NeurIPS%202025-red" alt="Paper"></a>
<a href="https://github.com/ehsanlatif/SketchMind"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="Code"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey" alt="License"></a>
<img src="https://i.imgur.com/waxVImv.png" alt="SketchMind">
</p>

<h3 align="center">SketchMind: A Multi-Agent Cognitive Framework for Assessing Student-Drawn Scientific Sketches [NeurIPS 2025 🔥]</h3>

<h4 align="center"><a href="https://scholar.google.com/citations?hl=en&user=UsrsXx0AAAAJ">Ehsan Latif</a>, <a href="https://scholar.google.com/citations?user=T4oy5R0AAAAJ">Zirak Khan</a> and <a href="https://scholar.google.com/citations?hl=en&user=pLk6BrEAAAAJ">Xiaoming Zhai</a></h4>

<h4 align="center"><strong>University of Georgia</strong></h4>

---

#### **Performance Benchmarks**

[![Average Accuracy Improvement](https://img.shields.io/badge/Average%20Accuracy%20Improvement-15.9%25-brightgreen)](https://github.com/ehsanlatif/SketchMind)

[![Highest Accuracy With SRG](https://img.shields.io/badge/Highest%20Accuracy%20With%20SRG%20-90.2%25-blue)](https://github.com/ehsanlatif/SketchMind)

[![Highest Feedback Quality](https://img.shields.io/badge/Highest%20Feedback%20Quality%20-4.1-purple)](https://github.com/ehsanlatif/SketchMind)

[![Best Results](https://img.shields.io/badge/Best%20Results-GPT--4.1%20-red)](https://github.com/ehsanlatif/SketchMind)

---

## 📢 Latest Updates

- **Sep-18-25**: SketchMind accepted at **NeurIPS 2025**! 🔥🥳
- **Jul-18-25**: Released complete multi-agent framework with both GPT-4 and LLaMA-4 implementations 🔥
- **May-15-25**: Published comprehensive evaluation on 3,500+ scientific sketches across 6 NGSS-aligned assessment items  🔥

---

## SketchMind Overview 💡

SketchMind introduces a **cognitively grounded, multi-agent framework** for assessing student drawn scientific sketches using semantic structures known as **Sketch Reasoning Graphs (SRGs)**. Each SRG is annotated with **Bloom's Taxonomy** levels and constructed via mulitmodal agents that collaboratively parse rubrics, analyze student sketches, evaluate conceptual understanding, and generate pedagogical feedback.

---

## Contributions 🏆

- **Novel Multi-Agent Framework**: First cognitively-grounded multi-agent system for scientific sketch assessment
- **Sketch Reasoning Graphs (SRGs)**: New semantic representation combining visual elements with Bloom's taxonomy levels
- **Comprehensive Evaluation**: Extensive validation on 3,500+ student sketches across 6 NGSS-aligned assessment items
- **Dual Model Implementation**: Complete pipelines for both proprietary (GPT-4) and open-source (LLaMA-4) models
- **Interactive Visualization**: Web-based tools for exploring and understanding SRG structures

---

## Multi-Agent Architecture ⚙️

### Agent 1: Rubric Parser 📄

- **Inputs**: Rubric, question text, gold-standard sketches
- **Outputs**: Gold-Standard Reference SRG with Bloom's taxonomy levels and reverse mapping
- **Function**: Establishes cognitive benchmarks from expert-designed rubrics and 3 gold-standard reference scientific sketches for each assessment tasks

### Agent 2: Sketch Parser 👁

- **Inputs**: Student sketch image, reference SRG
- **Outputs**: Student SRG constructed from visible sketch content
- **Function**: Extracts semantic elements and relationships from hand-drawn sketches

### Agent 3: SRG Evaluator ⚖️

- **Inputs**: Reference SRG, student SRG
- **Outputs**: Cognitive alignment score, proficiency classification, concept gaps
- **Function**: Compares graphs using edit distance and Bloom-level analysis

### Agent 4: Feedback Generator 💬

- **Inputs**: Evaluation results, original sketch, reference standards
- **Outputs**: Next-step sketch revision plan with Bloom-guided visual hints
- **Function**: Generates pedagogical feedback with visual overlays

---

## Setup 🔧

We recommend setting up a conda environment for the project:

```bash
# Create and activate environment
conda create --name sketchmind python=3.9+
conda activate sketchmind

# Clone repository
git clone https://github.com/ehsanlatif/SketchMind.git
cd SketchMind

# Install dependencies
pip install -r scripts/requirements.txt

```

For a complete list of dependencies, see [`scripts/requirements.txt`](scripts/requirements.txt).

---

## Quick Start Examples 🚀

### GPT-4o Pipeline

```bash
# Set your OpenAI API key in scripts/GPT_SRG_Agents.py
client = OpenAI(api_key="you api key",)

# Run the complete Multi-Agent-System pipeline
python scripts/GPT_SRG_MAS.py
```

### LLaMA-4 Pipeline

```bash
# Run with open-source models
python scripts/Llama4_SRG_MAS.py
```

---

## Directory Structure 📁

```
.
├── dataset/           # Gold-standard sketches from experts & student sketches
├── scripts/           # All core scripts including agent implementations
├── srg_cache/         # Cached Reference SRGs and intermediate feedback outputs
```

---

## Dataset 📂

SketchMind is evaluated on a comprehensive dataset of **3,500+ student-drawn scientific sketches** across **6 NGSS-aligned assessment items**:

### Annotation Schema

Each assessment item includes:

- ✅ Detailed textual rubric
- ✅ 3 gold standard scientific sketches (Beginning, Developing, Proficient)
- ✅ Student scientific sketch images
- ✅ Expert assigned proficiency labels

Note: The full dataset release is pending approval.

---

## Acknowledgements 🙏

- **OpenAI** for GPT API access and multimodal capabilities
- **Meta AI** for open-sourcing multimodal models like LLaMA-4
- **Open Router** for making LLaMa models available via GPT like API calls for easy reproducibility

Thanks to Dr. Xiaoming Zhai for his unwavering support throughout the project. Special thanks to our educators at AI4STEM Education Center at University of Georgia who provided domain expertise for rubric development.

## Citation 📜

```bibtex
@misc{latif2025sketchmindmultiagentcognitiveframework,
      title={SketchMind: A Multi-Agent Cognitive Framework for Assessing Student-Drawn Scientific Sketches}, 
      author={Ehsan Latif and Zirak Khan and Xiaoming Zhai},
      year={2025},
      eprint={2507.22904},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2507.22904}, 
}
```

## Contact ✉️

For questions, collaborations, or support:

- 📧 **Email**: Zirak.khan@uga.edu || Ehsan.Latif@uga.edu
- 🐛 **Issues**: [GitHub Issues](https://github.com/ehsanlatif/SketchMind/issues)

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---

Looking forward to your feedback, contributions, and stars! 🌟

<p align="center">
    <img src="https://img.shields.io/github/stars/ehsanlatif/SketchMind?style=social" alt="GitHub stars">
    <img src="https://img.shields.io/github/forks/ehsanlatif/SketchMind?style=social" alt="GitHub forks">
    <img src="https://img.shields.io/github/watchers/ehsanlatif/SketchMind?style=social" alt="GitHub watchers">
</p>
