from typing import List, Tuple, Dict
import networkx as nx
from difflib import SequenceMatcher
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenRouter client for Llama4 models
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


SRGBuilder_Class_String ="""
class SRGBuilder:
    BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    def __init__(self, question: str, rubric: str):
        self.question = question
        self.rubric = rubric
        self.nodes: List[Tuple[str, str]] = []  # (concept, Bloom level)
        self.edges: List[Tuple[str, str]] = []  # (from, to)

    def add_node(self, concept: str, bloom_level: str):
        assert bloom_level in self.BLOOM_ORDER, f"Invalid Bloom level: {bloom_level}"
        self.nodes.append((concept, bloom_level))

    def add_edge(self, source: str, target: str):
        self.edges.append((source, target))

    def build_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        return {"nodes": self.nodes, "edges": self.edges}

    def validate_graph(self) -> bool:
        node_names = {n[0] for n in self.nodes}
        valid_edges = all(u in node_names and v in node_names for u, v in self.edges)

        # Check for connectivity
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        connected = nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False

        # Ensure Bloom level ordering is respected
        bloom_levels = [self.BLOOM_ORDER.index(level) for _, level in self.nodes]
        ordered = bloom_levels == sorted(bloom_levels)

        return valid_edges and connected and ordered
"""
class SRGBuilder:
    BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    def __init__(self, question: str, rubric: str):
        self.question = question
        self.rubric = rubric
        self.nodes: List[Tuple[str, str]] = []  # (concept, Bloom level)
        self.edges: List[Tuple[str, str]] = []  # (from, to)

    def add_node(self, concept: str, bloom_level: str):
        assert bloom_level in self.BLOOM_ORDER, f"Invalid Bloom level: {bloom_level}"
        self.nodes.append((concept, bloom_level))

    def add_edge(self, source: str, target: str):
        self.edges.append((source, target))

    def build_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        return {"nodes": self.nodes, "edges": self.edges}

    def validate_graph(self) -> bool:
        node_names = {n[0] for n in self.nodes}
        valid_edges = all(u in node_names and v in node_names for u, v in self.edges)

        # Check for connectivity
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        connected = nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False

        # Ensure Bloom level ordering is respected
        bloom_levels = [self.BLOOM_ORDER.index(level) for _, level in self.nodes]
        ordered = bloom_levels == sorted(bloom_levels)

        return valid_edges and connected and ordered

    def graph_edit_distance(self, other: Dict[str, List[Tuple[str, str]]]) -> int:
        # Approximate edit distance: unmatched nodes + unmatched edges
        nodes_self = set(self.nodes)
        edges_self = set(self.edges)
        nodes_other = set(other['nodes'])
        edges_other = set(other['edges'])
        node_diff = len(nodes_self.symmetric_difference(nodes_other))
        edge_diff = len(edges_self.symmetric_difference(edges_other))
        return node_diff + edge_diff

    def ontology_alignment_score(self, other: Dict[str, List[Tuple[str, str]]]) -> float:
        def concept_similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        scores = []
        for a_concept, _ in self.nodes:
            max_sim = max([concept_similarity(a_concept, b_concept) for b_concept, _ in other["nodes"]], default=0.0)
            scores.append(max_sim)
        return sum(scores) / len(scores) if scores else 0.0

    def compute_similarity(self, other: Dict[str, List[Tuple[str, str]]], gamma_1=0.6, gamma_2=0.4, Z=10.0) -> float:
        ged = self.graph_edit_distance(other)
        oa = self.ontology_alignment_score(other)
        max_ged = len(set(self.nodes)) + len(set(self.edges)) + len(set(other['nodes'])) + len(set(other['edges']))
        score = 1 - (gamma_1 * (ged/max_ged) + gamma_2 * (1 - oa))
        print(f"Graph Edit Distance: {ged}, Ontology Alignment: {oa}, Score: {score}")
        return max(0.0, min(1.0, score))  # clamp to [0,1]

SCORE_THRESHOLD = 0.75

# ------------------ OpenRouter API Call for Llama4 ------------------
def gpt_call(history, user_text=None, image_base64=None, model='meta-llama/llama-4-maverick:free'):
    """
    Call OpenRouter API with Llama4 model.
    
    Args:
        history: Chat history in OpenAI format
        user_text: Optional text to append
        image_base64: Optional base64-encoded image
        model: OpenRouter model identifier
    
    Returns:
        Model response string
    """
    content = []
    if image_base64:
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })
    if user_text:
        content.append({"type": "text", "text": user_text})
    
    if content:
        history.append({"role": "user", "content": content})
    
    response = client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=5000,
        temperature=0.0,
        top_p=1.0  # Set top_p to 1 for deterministic output
    )
    
    output = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": output})
    return output
# ------------------ Agent 1: Rubric Parser ------------------
class Agent1:
    def __init__(self, question, rubric, golden_standard_images):
        self.history = [{
            "role": "system",
            "content": (
            "You are Agent 1: Rubric Parser in a cognitive sketch evaluation system\n"
            "Your task is to construct a Bloom-aligned Sketch Reasoning Graph (SRG) based on the provided question, rubric, and example sketches.\n"
            f"Use the SRGBuilder class {SRGBuilder_Class_String} to:\n"
            "Follow these steps:\n"
            "1. Extract concise semantic concepts relevant to the question and rubric. Label each concept with its corresponding Bloom's taxonomy level: Remember, Understand, Apply, Analyze, Evaluate, or Create.\n"
            "2. Define directed edges that capture causal or logical relationships between these concepts.\n"
            "3. Incorporate insights from example sketches to identify additional visual concepts and reinforce expectations.\n"
            "4. Ensure the graph is connected, meaning there is a path between every pair of nodes, and that the Bloom levels follow a bottom-up progression.\n"
            "5. Provide a reverse mapping from each concept to a visual drawing hint, utilizing Bloom's taxonomy to guide the representation.\n"
            "Your output must strictly adhere to the following JSON format:\n"
            "{\n"
                "  'srg': {'nodes': [{'label':...,'bloom_level':...}], 'edges': [{'from':...,'to':...}]},\n"
                "  'reverse_mapping': {concept_name: visual_hint}\n"
                "}\n"
            "Use double quotes for all keys and string values. Do not include any explanations or additional text outside the JSON structure."
            )
        }, {
            "role": "user",
            "content": f"Question: {question}\nRubric: {rubric}\n Please analyze the above information and generate the Sketch Reasoning Graph (SRG) and reverse mapping as per the system instructions and Please analyze the following golden standard sketches for guidance\n"
        }]

        # Attach golden-standard images
        for img_b64 in golden_standard_images:
            self.history.append({
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{img_b64[0]}" } },
                    { "type": "text", "text": "This is a golden standard sketch." }
                ]
            })

    def run(self, feedback=None):
        if feedback:
            self.history.append({ "role": "user", "content": feedback })
        return gpt_call(self.history)


# ------------------ Agent 2: Sketch SRG Extractor ------------------
class Agent2:
    def __init__(self, image_base64, ref_srg):
        self.image_base64 = image_base64
        self.ref_srg = ref_srg

        node_labels = [n[0] for n in self.ref_srg['nodes']]
        edge_pairs = [(e[0], e[1]) for e in self.ref_srg['edges']]

        ref_srg_text = (
            "Reference SRG Node Labels:\n" +
            "\n".join(f"- {label}" for label in node_labels) +
            "\n\nReference SRG Edges:\n" +
            "\n".join(f"- {source} → {target}" for source, target in edge_pairs)
        )

        self.history = [{
            "role": "system",
            "content": (
                "You are Agent 2: Sketch Parser in a cognitive sketch evaluation system.\n"
                "You receive a student's sketch and analyze its contents to construct a Sketch Reasoning Graph (SRG).\n"
                "Your SRG output must use the same node labels and edge labels as in the reference SRG if possible.\n"
                "Only include a node or edge if it is visibly present or clearly inferable.\n"
                "Instructions:\n"
                "1. Use the reference SRG node and edge names as a template.\n"
                "2. Detect which concepts and relationships from the reference are actually present in the sketch.\n"
                "3. Label each node with the Bloom level shown by the sketch.\n"
                "4. Return only valid components and their Bloom levels.\n"
                "Return JSON in the format (double quotes for keys and values):\n"
                "{\n"
                "  'srg': {\n"
                "    'nodes': [{'label': ..., 'bloom_level': ...}],\n"
                "    'edges': [{'from': ..., 'to': ...}]\n"
                "  }\n"
                "}\n"
                "Be strict and do not assume missing content. Match names exactly where applicable."
                "Use double quotes for all keys and string values. Do not include any explanations or additional text outside the JSON structure."
            )
        }, {
            "role": "user",
            "content": f"Reference SRG:\n{ref_srg_text}\nPlease analyze the student's sketch provided below and based on the reference SRG, generate the Sketch Reasoning Graph (SRG) as per the system instructions."
        }]

    def run(self):
        return gpt_call(self.history, image_base64=self.image_base64)

# (Above content preserved)

# ------------------ Agent 3: SRG Evaluator ------------------
class Agent3:
    def __init__(self):
        pass  # No Llama4 prompt needed since this agent runs deterministic logic only.

    def run(self, reference_srg, student_srg):
        ref = SRGBuilder("", "")
        stu = SRGBuilder("", "")
        ref.nodes, ref.edges = reference_srg['nodes'], reference_srg['edges']
        stu.nodes, stu.edges = student_srg['nodes'], student_srg['edges']

        score = stu.compute_similarity(reference_srg)

        missing_nodes = [n for n in reference_srg['nodes'] if n not in student_srg['nodes']]
        missing_edges = [e for e in reference_srg['edges'] if e not in student_srg['edges']]
        irrelevant_nodes = [n for n in student_srg['nodes'] if n not in reference_srg['nodes']]
        irrelevant_edges = [e for e in student_srg['edges'] if e not in reference_srg['edges']]

        bloom_discrepancies = []
        ref_node_dict = dict(reference_srg['nodes'])
        for concept, level in student_srg['nodes']:
            if concept in ref_node_dict and level != ref_node_dict[concept]:
                bloom_discrepancies.append({"concept": concept, "expected": ref_node_dict[concept], "observed": level})

        if score >= SCORE_THRESHOLD:
            label = "Proficient"
        elif score >= 0.5:
            label = "Developing"
        else:
            label = "Beginning"

        justification = f"The student SRG received a similarity score of {score:.2f}, indicating a {label.lower()} level of conceptual understanding."
        terminate = score >= SCORE_THRESHOLD

        # Suggest a priority list of missing concepts based on Bloom level order
        priority_fix = sorted(missing_nodes, key=lambda x: SRGBuilder.BLOOM_ORDER.index(x[1]))

        # Check for logic flow issues: any expected source nodes without outgoing edges
        expected_sources = {src for src, _ in reference_srg['edges']}
        actual_sources = {src for src, _ in student_srg['edges']}
        conceptual_gaps = list(expected_sources - actual_sources)

        encouragement = "The student demonstrated sound structure despite missing some components." if score >= 0.5 and not irrelevant_edges else "The sketch needs refinement to follow clear reasoning."

        return {
            "similarity_score": round(score, 3),
            "classification": label,
            "missing_nodes": missing_nodes,
            "missing_edges": missing_edges,
            "irrelevant_nodes": irrelevant_nodes,
            "irrelevant_edges": irrelevant_edges,
            "bloom_discrepancies": bloom_discrepancies,
            "priority_fix": priority_fix,
            "conceptual_gaps": conceptual_gaps,
            "encouragement": encouragement,
            "justification": justification,
            "terminate": terminate
        }

# ------------------ Agent 4: Feedback Generator and Co-Creation Facilitator ------------------
class Agent4:
    def __init__(self, agent1_output, reverse_mapping, golden_standard_images=None, threshold  = SCORE_THRESHOLD):
        self.similarity_threshold = threshold
        self.reference_srg = agent1_output
        self.reverse_mapping = reverse_mapping or {}
        self.golden_standard_images = golden_standard_images or []

    def generate_visual_hint(self, node_or_edge):
        # if isinstance(node_or_edge, tuple):  # edge
        #     source, target = node_or_edge
        #     return f"Draw a connection from '{source}' to '{target}' using an arrow to show their relationship."
        return self.reverse_mapping.get(node_or_edge[0])#, f"Illustrate '{node_or_edge[0]}' in your diagram.")

    def generate_visual_overlay(self, image_base64, missing_concepts, image_size, feedback_text):
        # Step 1: Enhanced system prompt emphasizing WHERE and WHAT
        system_prompt = """You are an expert visual education assistant that creates helpful sketch annotations.

        Your task: Generate Python PIL code to overlay visual hints showing WHERE and WHAT concepts are missing.

        IMPORTANT CONTEXT:
        - Students receive TEXTUAL feedback explaining WHAT is missing and HOW to draw it
        - Your overlay must show WHERE on the sketch to add concepts and WHAT each concept is
        - Create clear labels for MULTIPLE missing concepts on one image

        VISUAL DESIGN REQUIREMENTS:
        1. POSITIONING & ARROWS:
        - Identify WHERE each missing concept should be added on the sketch
        - Use arrows/indicators pointing to appropriate locations
        - Place labels in empty areas, not covering existing work

        2. CONCEPT LABELS:
        - Label WHAT concept should be added at each location
        - Use clear text boxes with concept names
        - Number or color-code different concepts if needed

        3. CLARITY & READABILITY:
        - Use high contrast colors for different concepts
        - Semi-transparent overlays (alpha=0.75-0.85)
        - Font size: minimum 18px for labels
        - Distinct colors for each concept (red, blue, green, orange, etc.)

        4. MULTIPLE CONCEPTS:
        - Show ALL missing concepts on the same image
        - Use different colors/numbers to distinguish them
        - Ensure labels don't overlap or clutter

        5. CODE QUALITY:
        - Use ImageDraw, ImageFont from PIL
        - Handle font loading gracefully (use default font as fallback)
        - Function name MUST be: overlay_hint(image: Image.Image) -> Image.Image

        GOAL: Student should see WHERE to add each concept and WHAT it is. The textual feedback explains HOW."""

        # Step 2: Build list of concepts with their hints
        concepts_list = "\n".join([
            f"{i+1}. {concept[0]} - {self.reverse_mapping.get(concept[0], 'Add this concept')}"
            for i, concept in enumerate(missing_concepts)
        ])

        # Step 3: Multi-part user prompt with feedback context
        analysis_prompt = f"""STUDENT FEEDBACK (for context):
        {feedback_text}

        TASK CONTEXT:
        - Image size: {image_size[0]}px × {image_size[1]}px
        - Missing Concepts to visualize:
        {concepts_list}

        REFERENCE IMAGES:
        Below are golden standard sketches showing examples.

        YOUR TASK:
        1. Study the student sketch (provided below):
        - Identify WHERE each missing concept should be added
        - Find appropriate locations for multiple concept labels
        - Avoid covering existing elements

        2. Study the golden standard images:
        - See where concepts appear in good examples
        - Understand spatial relationships

        3. Generate Python PIL code that creates ONE overlay with:
        - **Labels** for each missing concept showing WHAT it is
        - **Arrows/indicators** pointing to WHERE each should be added
        - **Distinct colors** to differentiate concepts
        - **Numbers or color codes** (e.g., "1. Water Particles →", "2. Dye Particles →")
        
        Example overlay for multiple concepts:
        - Red arrow + text: "1. Add Water Particles here →"
        - Blue arrow + text: "2. Add Dye Particles here →"  
        - Green arrow + text: "3. Add Motion Arrows here →"
        - Each pointing to appropriate locations on sketch

        Return ONLY the Python function code, no explanations."""

                    # Step 3: Build message history with golden standards first
        history = [{"role": "system", "content": system_prompt}]
                    
                    # Add golden standard images
        for img_b64 in self.golden_standard_images:
            history.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64[0]}"}},
                    {"type": "text", "text": "Golden standard reference."}
                ]
            })
        
        # Add student image with analysis prompt
        response = gpt_call(
            history=history,
            user_text=analysis_prompt,
            image_base64=image_base64
        )
        
        return response

    def run(self, agent3_output):
        summary = f"\n**Your Proficiency Level:** {agent3_output['classification']}\n"
        summary += f"\n**What You Did Well:**\n{agent3_output['encouragement']}\n"

        if agent3_output['terminate']:
            summary += "\n Your sketch shows full conceptual alignment. Great job! You may stop here."
            return summary

        summary += "\n**What Needs Attention:**\n"
        if agent3_output['missing_nodes']:
            summary += "- Missing Concepts: " + ", ".join(n[0] for n in agent3_output['missing_nodes']) + "\n"
        if agent3_output['missing_edges']:
            summary += "- Missing Relationships: connections not shown in sketch.\n"
        if agent3_output['irrelevant_nodes']:
            summary += "- Extra Concepts: " + ", ".join(n[0] for n in agent3_output['irrelevant_nodes']) + "\n"
        if agent3_output['bloom_discrepancies']:
            summary += "- Depth Conflicts: Some concepts are not represented at the correct Bloom level.\n"

        summary += "\n**Co-Creation Guidance (Next Sketch Revisions):**\n"
        for item in agent3_output['priority_fix']:
            hint = self.generate_visual_hint(item)
            summary += f"- {item[0]} ({item[1]}): {hint}\n"

        # for edge in agent3_output['missing_edges']:
        #     summary += f"- Relationship: {self.generate_visual_hint(edge)}\n"

        if agent3_output['conceptual_gaps']:
            summary += "\n**Reasoning Gaps Detected In:**\n"
            summary += "  → " + ", ".join(agent3_output['conceptual_gaps']) + "\n"

        return summary
