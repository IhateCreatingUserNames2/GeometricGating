from typing import Any, List
from mcp.server.fastmcp import FastMCP
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.linalg import qr


# --- SUA CLASSE ORIGINAL (Com pequenas adaptações para o Server) ---
class GeometricGate:
    def __init__(self, target_dim=15, epsilon=0.90, model_name='all-mpnet-base-v2'):
        self.target_dim = target_dim
        self.epsilon = epsilon
        print(f"📥 Loading model: {model_name}...")  # Isso vai pro log do MCP, não pro chat
        self.model = SentenceTransformer(model_name)
        self.memory_vectors = []
        self.memory_texts = []
        self.projection_matrix = None

    def _get_projection(self, input_dim):
        if self.projection_matrix is None:
            np.random.seed(42)
            random_matrix = np.random.randn(input_dim, self.target_dim)
            q, _ = qr(random_matrix, mode='economic')
            self.projection_matrix = q
        return self.projection_matrix

    def project(self, embedding):
        P = self._get_projection(len(embedding))
        v_proj = np.dot(embedding, P)
        return v_proj / np.linalg.norm(v_proj)

    def check(self, text, auto_add=True):
        embedding = self.model.encode(text)
        v_proj = self.project(embedding)

        if not self.memory_vectors:
            if auto_add:
                self.memory_vectors.append(v_proj)
                self.memory_texts.append(text)
            return True, float('inf'), None

        memory_matrix = np.array(self.memory_vectors)
        distances = np.linalg.norm(memory_matrix - v_proj, axis=1)
        min_dist = np.min(distances)
        idx_min = np.argmin(distances)

        is_distinct = min_dist >= self.epsilon

        if is_distinct and auto_add:
            self.memory_vectors.append(v_proj)
            self.memory_texts.append(text)

        similar_text = self.memory_texts[idx_min] if not is_distinct else None
        return is_distinct, float(min_dist), similar_text

    def reset(self):
        self.memory_vectors = []
        self.memory_texts = []


# --- INICIALIZAÇÃO DO MCP ---
mcp = FastMCP("geometric-gating")

# Estado Global do Servidor
# Iniciamos no modo "Balanced" (Chat) por padrão
gate = GeometricGate(target_dim=10, epsilon=0.85, model_name='all-MiniLM-L6-v2')


# --- DEFINIÇÃO DAS TOOLS PARA O CLAUDE ---

@mcp.tool()
def configure_mode(mode: str) -> str:
    """
    Configures the Geometric Gate mode.
    Options:
    - 'balanced' (Default): For chat context optimization (Dim=10, Eps=0.85)
    - 'strict': For RAG/Deduplication (Dim=15, Eps=0.90, requires MPNet)
    - 'creative': For Brainstorming (Dim=7, Eps=1.10)
    """
    global gate
    # Presets calibrados no seu paper
    presets = {
        "balanced": {"target_dim": 10, "epsilon": 0.85, "model_name": "all-MiniLM-L6-v2"},
        "strict": {"target_dim": 15, "epsilon": 0.90, "model_name": "all-mpnet-base-v2"},
        "creative": {"target_dim": 7, "epsilon": 1.10, "model_name": "all-MiniLM-L6-v2"},
    }

    if mode not in presets:
        return f"Error: Mode '{mode}' not found. Use 'balanced', 'strict', or 'creative'."

    config = presets[mode]
    # Reinicializa o gate com nova configuração
    gate = GeometricGate(**config)
    gate.reset()  # Limpa memória ao trocar de modo para evitar incompatibilidade vetorial

    return f"Geometric Gate reconfigured to '{mode.upper()}' mode. Memory cleared."


@mcp.tool()
def check_novelty(text: str) -> str:
    """
    Checks if a text is geometrically distinct from current memory WITHOUT adding it.
    Useful for filtering lists before deciding what to keep.
    """
    is_distinct, dist, similar = gate.check(text, auto_add=False)

    if is_distinct:
        return f"NOVEL (Dist: {dist:.2f})"
    else:
        return f"REDUNDANT (Dist: {dist:.2f}). Too similar to: '{similar}'"


@mcp.tool()
def add_to_memory(text: str) -> str:
    """
    Checks semantic novelty and ADDS to memory if distinct.
    Returns the decision. Use this to build up a context of unique ideas.
    """
    is_distinct, dist, similar = gate.check(text, auto_add=True)

    if is_distinct:
        return f"ACCEPTED. Added to geometric memory (Dist: {dist:.2f})"
    else:
        return f"REJECTED. Redundant (Dist: {dist:.2f} vs '{similar}')"


@mcp.tool()
def filter_list(items: List[str]) -> str:
    """
    Takes a list of strings and returns ONLY the geometrically distinct ones.
    Processes them sequentially.
    """
    kept = []
    rejected_count = 0

    # Snapshot da memória atual para não poluir permanentemente se for só um teste
    # Mas para este uso, vamos assumir que queremos popular a memória
    for item in items:
        is_distinct, _, _ = gate.check(item, auto_add=True)
        if is_distinct:
            kept.append(item)
        else:
            rejected_count += 1

    return f"Filtered {len(items)} items. Kept {len(kept)} distinct items. Rejected {rejected_count}."


@mcp.tool()
def reset_memory() -> str:
    """Clears the geometric memory."""
    gate.reset()
    return "Memory cleared."


if __name__ == "__main__":
    mcp.run(transport="stdio")