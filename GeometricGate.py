import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.linalg import qr
import os
from openai import OpenAI

# --- CONFIGURAÇÃO DO USUÁRIO ---
LLM_PROVIDER = "openrouter"
API_KEY = "sk-or-v1-"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "x-ai/grok-code-fast-1"  # Modelo rápido solicitado

# Inicialização do Cliente
try:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    HAS_AI_API = True
    print(f"✅ Conectado ao {LLM_PROVIDER} usando {MODEL_NAME}")
except Exception as e:
    HAS_AI_API = False
    print(f"⚠️ Erro ao configurar API: {e}")


# --- MOTOR GEOMÉTRICO (GEOMETRIC GATING) ---
class GeometricGate:
    def __init__(self, target_dim=10, epsilon=0.85, model_name='all-MiniLM-L6-v2'):
        """
        O Motor do Paper.
        target_dim=7: Limite de Miller / Eficiência Geométrica.
        epsilon=1.10: Kissing Number (1.0) + Margem de Ruído (0.1).
        """
        self.target_dim = target_dim
        self.epsilon = epsilon

        # Carrega embeddings locais (CPU/GPU)
        print(f"📥 Carregando modelo de embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)

        self.memory_vectors = []
        self.memory_texts = []
        self.projection_matrix = None

    def _get_projection(self, input_dim):
        """Matriz ortogonal fixa R^N -> R^7 via Decomposição QR."""
        if self.projection_matrix is None:
            np.random.seed(42)  # Determinístico
            random_matrix = np.random.randn(input_dim, self.target_dim)
            q, _ = qr(random_matrix, mode='economic')
            self.projection_matrix = q
        return self.projection_matrix

    @classmethod
    def create(cls, mode="balanced"):
        """
        Fábrica de Gates com presets calibrados empiricamente (Validação Fev/2026).
        """
        presets = {
            # Modo Padrão: Alta precisão (90% Acc), usa MPNet
            "balanced": {
                "target_dim": 15,
                "epsilon": 0.90,
                "model_name": "all-mpnet-base-v2"
            },

            # Modo Leve: Para velocidade (CPU), usa MiniLM
            "fast": {
                "target_dim": 10,
                "epsilon": 0.85,
                "model_name": "all-MiniLM-L6-v2"
            },

            # Modo Criativo: Para Brainstorming, força divergência
            "creative": {
                "target_dim": 7,
                "epsilon": 1.10,
                "model_name": "all-MiniLM-L6-v2"
            },
        }

        # Padrão para balanced se não encontrar
        config = presets.get(mode, presets["balanced"])
        print(f"⚙️  GeometricGate iniciado em modo '{mode.upper()}': {config}")
        return cls(**config)

    def project(self, embedding):
        """Projeção e Normalização."""
        P = self._get_projection(len(embedding))
        v_proj = np.dot(embedding, P)
        return v_proj / np.linalg.norm(v_proj)

    def check(self, text, auto_add=True):
        """Retorna: (is_distinct, min_distance, similar_text)"""
        embedding = self.model.encode(text)
        v_proj = self.project(embedding)

        if not self.memory_vectors:
            if auto_add:
                self.memory_vectors.append(v_proj)
                self.memory_texts.append(text)
            return True, float('inf'), None

        # Distância Euclidiana em 7D
        memory_matrix = np.array(self.memory_vectors)
        distances = np.linalg.norm(memory_matrix - v_proj, axis=1)

        min_dist = np.min(distances)
        idx_min = np.argmin(distances)

        # O Juiz Geométrico
        is_distinct = min_dist >= self.epsilon

        if is_distinct and auto_add:
            self.memory_vectors.append(v_proj)
            self.memory_texts.append(text)

        return is_distinct, min_dist, self.memory_texts[idx_min]


# --- FERRAMENTAS ---

def demo_context_optimizer():
    print("\n" + "=" * 60)
    print("🛠️  TOOL 1: CONTEXT OPTIMIZER (Anti-Redundancy)")
    print("=" * 60)

    gate = GeometricGate()

    user_inputs = [
        "Meu script python não roda.",
        "Estou tendo erro de execução no Python.",  # Redundante
        "Qual a capital da França?",  # Novo
        "Paris é a capital de onde?",  # Redundante
        "Receita de bolo de cenoura."  # Novo
    ]

    clean_context = []
    print(f"\n{'INPUT':<45} | {'STATUS':<10} | {'DIST':<6} | {'REASON'}")
    print("-" * 105)

    for text in user_inputs:
        is_distinct, dist, similar = gate.check(text)
        status = "✅ ACCEPT" if is_distinct else "❌ REJECT"
        reason = "New concept" if is_distinct else f"Close to: '{similar[:15]}...'"

        if is_distinct: clean_context.append(text)
        print(f"{text[:45]:<45} | {status:<10} | {dist:.2f}   | {reason}")


def demo_diverse_rag():
    print("\n" + "=" * 60)
    print("🛠️  TOOL 2: DIVERSE RAG (Semantic Diversity Filter)")
    print("=" * 60)

    chunks = [
        "Grok é uma IA desenvolvida pela xAI.",
        "A xAI criou o modelo Grok.",  # Redundante
        "Grok tem acesso a dados em tempo real.",  # Novo
        "Dados em tempo real são usados pelo Grok.",  # Redundante
        "O modelo possui janelas de contexto longas."  # Novo
    ]

    gate = GeometricGate(target_dim=12, epsilon=0.9)

    print("Filtrando chunks...")
    for chunk in chunks:
        is_distinct, dist, _ = gate.check(chunk)
        if is_distinct:
            print(f"✅ Mantido: {chunk}")
        else:
            print(f"❌ Removido (Dist: {dist:.2f}): {chunk}")


def demo_forced_brainstorming():
    print("\n" + "=" * 60)
    print(f"🛠️  TOOL 3: FORCED BRAINSTORMING com {MODEL_NAME}")
    print("=" * 60)

    # Threshold 1.15 força ideias muito diferentes
    gate = GeometricGate(target_dim=10, epsilon=0.95)

    topic = "Usos futuristas para IA na Agricultura"
    accepted_ideas = []
    attempts = 0
    max_ideas = 3

    print(f"🤖 Gerando ideias sobre: '{topic}'...")

    while len(accepted_ideas) < max_ideas and attempts < 10:
        attempts += 1

        if HAS_AI_API:
            try:
                # Prompt simples, deixando o Gate fazer o trabalho pesado de filtragem
                prompt = f"Gere UMA ideia de startup curta (1 frase) sobre {topic}. Tentativa {attempts}."

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,  # Temperatura alta para ajudar
                    extra_headers={
                        "HTTP-Referer": "https://geometric-gating.local",
                        "X-Title": "Geometric Demo"
                    }
                )
                idea = response.choices[0].message.content.strip().replace('"', '')
            except Exception as e:
                print(f"Erro API: {e}")
                break
        else:
            print("API não disponível.")
            break

        # O Filtro Geométrico decide se a ideia é boa/nova
        is_distinct, dist, similar = gate.check(idea)

        if is_distinct:
            print(f"\n💡 IDEIA {len(accepted_ideas) + 1} (Dist: {dist:.2f}):\n   {idea}")
            accepted_ideas.append(idea)
        else:
            print(f"\n🗑️  REJEITADA (Dist: {dist:.2f} vs '{similar[:15]}...'):\n   {idea}")


def demo_forced_brainstorming_v2():
    print("\n" + "=" * 60)
    print(f"🛠️  TOOL 3 V2: GEOMETRIC FEEDBACK LOOP")
    print("=" * 60)

    # Modo Criativo (7D, 1.10) - O mais difícil de passar
    gate = GeometricGate(target_dim=7, epsilon=1.10)

    topic = "Usos futuristas para IA na Agricultura"
    accepted_ideas = []
    rejected_topics = []  # Memória de curto prazo do que foi rejeitado
    attempts = 0

    print(f"🤖 Brainstorming Ativo: '{topic}'...")

    while len(accepted_ideas) < 3 and attempts < 10:
        attempts += 1

        # Constrói restrições negativas baseadas no que o Gate rejeitou
        constraints = ""
        if rejected_topics:
            constraints = f"EVITE assuntos já mencionados: {', '.join(rejected_topics[-2:])}."

        prompt = f"Gere UMA ideia de startup CURTA (max 15 palavras) sobre {topic}. {constraints} Seja radicalmente diferente."

        # ... (chamada da API igual antes) ...
        # Se não tiver API, use o mock
        try:
            if HAS_AI_API:
                response = client.chat.completions.create(
                    model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
                )
                idea = response.choices[0].message.content.strip()
            else:
                idea = "Mock idea about drones"  # Mock simples
        except:
            break

        # Check Geométrico
        is_distinct, dist, similar = gate.check(idea)

        if is_distinct:
            print(f"\n✅ ACEITA (Dist: {dist:.2f}): {idea}")
            accepted_ideas.append(idea)
        else:
            print(f"❌ REJEITADA (Dist: {dist:.2f}): {idea}")
            # O "PULO DO GATO": Adiciona a ideia rejeitada na lista negra
            # Extrai uma palavra chave simples ou usa a frase toda
            rejected_topics.append(f"'{idea[:20]}...'")
            print(f"   ↪️ Adicionando restrição ao prompt...")

    print(f"\nSessão encerrada com {len(accepted_ideas)} ideias diversas.")


# --- EXECUÇÃO ---
if __name__ == "__main__":

    demo_diverse_rag()
    demo_context_optimizer()
    demo_forced_brainstorming()
