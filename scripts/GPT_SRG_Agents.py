from openai import OpenAI
from typing import List, Tuple, Dict
import networkx as nx
from difflib import SequenceMatcher
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
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

# ------------------ GPT-4o API Call ------------------
def gpt_call(history, user_text=None, image_base64=None,model='gpt-4o'):
    content = []
    if image_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    if user_text:
        content.append({"type": "text", "text": user_text})
    if content:
        history.append({"role": "user", "content": content})
    response = client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=1500
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
                "You are Agent 1: Rubric Parser in a cognitive sketch evaluation system.\n"
                "Your job is to construct a Bloom-aligned Sketch Reasoning Graph (SRG) based on the rubric, question, and example sketches.\n"
                f"Use the SRGBuilder class {SRGBuilder_Class_String} to:\n"
                "1. Extract semantic concepts relavent to the given question and rubric.\n"
                "2. Label them with Bloom levels (Remember, Understand, Apply, Analyze, Evaluate, Create).\n"
                "3. Define directed edges that capture causal or logical relationships.\n"
                "4. Use example sketches to identify additional visual concepts and reinforce what is expected.\n"
                "5. Validate the graph for connectivity (There should be a path from each node to other node) and Bloom level ordering.\n"
                "6. Provide a reverse mapping from concepts to drawing hints using Bloom's taxonomy.\n"
                "Return JSON in this format:\n"
                "{\n"
                "  'srg': {'nodes': [{'label':...,'bloom_level':...}], 'edges': [{'from':...,'to':...}]},\n"
                "  'reverse_mapping': {concept_name: visual_hint}\n"
                "}\n"
                "Be consistent with node naming across examples. Be thorough but not redundant."
            )
        }, {
            "role": "user",
            "content": f"Question: {question}\nRubric: {rubric}\nPlease analyze the following golden standard sketches for guidance."
        }]

        # Attach golden-standard images
        for img_b64 in golden_standard_images:
            self.history.append({
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{img_b64[0]}" } },
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
                "Return JSON in the format:\n"
                "{\n"
                "  'srg': {\n"
                "    'nodes': [{'label': ..., 'bloom_level': ...}],\n"
                "    'edges': [{'from': ..., 'to': ...}]\n"
                "  }\n"
                "}\n"
                "Be strict and do not assume missing content. Match names exactly where applicable."
            )
        }, {
            "role": "user",
            "content": ref_srg_text
        }]

    def run(self):
        return gpt_call(self.history, image_base64=self.image_base64)

# (Above content preserved)

# ------------------ Agent 3: SRG Evaluator ------------------
class Agent3:
    def __init__(self):
        pass  # No GPT prompt needed since this agent runs deterministic logic only.

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
    def __init__(self, agent1_output, reverse_mapping, threshold  = SCORE_THRESHOLD):
        self.similarity_threshold = threshold
        self.reference_srg = agent1_output
        self.reverse_mapping = reverse_mapping or {}

    def generate_visual_hint(self, node_or_edge):
        # if isinstance(node_or_edge, tuple):  # edge
        #     source, target = node_or_edge
        #     return f"Draw a connection from '{source}' to '{target}' using an arrow to show their relationship."
        return self.reverse_mapping.get(node_or_edge[0])#, f"Illustrate '{node_or_edge[0]}' in your diagram.")

    def generate_visual_overlay(self,image_base64, concept, visual_hint, image_size):

        prompt = (
            f"Generate a Python function using PIL to visually overlay a hint onto a student sketch. \n"
            f"The original image is {image_size[0]}px wide and {image_size[1]}px tall.\n"
            f"Concept: {concept}\n"
            f"Hint: {visual_hint}\n"
            f"Return only the function named `overlay_hint(image: Image.Image) -> Image.Image`."
        )

        response = gpt_call(
            history=[{"role": "system", "content": "You are an assistant that returns Python code to overlay visual sketch hints on images. Before generating the code, understand the given image on which this overly will apply, carefully to position objects appropriatly. Do not just randomly place overlay on the image in the code."}],
            user_text=prompt,
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
