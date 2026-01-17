# Geometric Gating MCP Server

> **"Seven is not a magic number. It's a geometric constraint."**

A Model Context Protocol (MCP) server that implements **Geometric Gating**—a mathematical framework for filtering semantic redundancy using low-dimensional sphere packing principles.

Unlike traditional semantic search (which ranks similarity), **Geometric Gating** acts as a hard filter based on information density. It projects embeddings into a "cognitive workspace" (e.g., 7D or 15D) and rejects any new input that falls within the "noise margin" of existing concepts.

---

## 🧠 The Theory

This project is based on the paper *"Dimensional Separation in Cognitive Representation"* (2026), which proposes that cognitive capacity limits (like Miller's $7 \pm 2$) are emergent properties of high-dimensional geometry.

### How it Works
1.  **Embed**: Input text is converted to a vector (384d or 768d).
2.  **Project**: The vector is projected onto a low-dimensional hypersphere (e.g., 7D).
3.  **Gate**: A new vector is accepted **only if** its Euclidean distance to all stored vectors exceeds a strict threshold ($\epsilon$).

### Modes
We empirically calibrated three operating modes for different AI tasks:

| Mode | Dimensions | Threshold ($\epsilon$) | Best For | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Balanced** | 10D | 0.85 | Chat Context | Allows topic shifts, blocks repetitions. |
| **Strict** | 15D | 0.90 | RAG / Deduplication | Distinguishes fine details (e.g., France vs Brazil). |
| **Creative** | 7D | 1.10 | Brainstorming | Forces radical divergence; rejects "more of the same". |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`
- Claude Desktop App (for MCP integration)

### 1. Clone and Install
```bash
git clone https://github.com/your-username/geometric-gating-mcp.git
cd geometric-gating-mcp

# Using uv (fastest)
uv venv
source .venv/bin/activate
uv add "mcp[cli]" sentence-transformers scipy numpy
```

### 2. Configure Claude Desktop
Edit your config file (`claude_desktop_config.json`):

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Mac/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`

Add the server configuration:
```json
{
  "mcpServers": {
    "geometric-gate": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/geometric-gating-mcp",
        "run",
        "geometric_server.py"
      ]
    }
  }
}
```
*(Remember to replace `/ABSOLUTE/PATH/TO/...` with your actual path. On Windows, use double backslashes `\\`)*

---

## 🛠️ Tools Available

Once connected, Claude will have access to these tools:

### `configure_mode(mode: str)`
Switches the internal geometry. Warning: Resets memory.
- Example: `configure_mode("strict")` for cleaning a dataset.

### `add_to_memory(text: str)`
Attempts to add a concept to the geometric workspace.
- **Returns:** "ACCEPTED" if distinct, or "REJECTED (Too similar to X)" if redundant.
- Use this to build a list of unique ideas.

### `check_novelty(text: str)`
Checks if text is new *without* adding it.
- Useful for filtering a stream of tokens before generation.

### `filter_list(items: List[str])`
Batch processes a list and returns only the geometrically distinct items.
- Example: "Filter this list of 50 customer feedback comments to the core unique complaints."

### `reset_memory()`
Wipes the slate clean.

---

## 🧪 Empirical Validation

We validated this implementation against a ground-truth dataset of semantic pairs.

**Results (Strict Mode - 15D/0.90):**
- **Accuracy:** 90%
- **False Positives:** 0% (Never confused exact duplicates)
- **Key Win:** Successfully distinguished *"Capital of France"* from *"Capital of Brazil"* (Dist: 1.02) while correctly identifying *"Python script error"* and *"Error running Python"* as duplicates (Dist: 0.50).

---

## 💡 Example Usage Prompts

**For RAG (Retrieval Augmented Generation):**
> "I have retrieved 20 chunks from my vector DB. They are very repetitive. Use `filter_list` in 'strict' mode to give me only the pieces of information that are strictly unique."

**For Creative Writing:**
> "Switch to 'creative' mode. I want to brainstorm names for a new soda brand. Generate a name, try to `add_to_memory`. If rejected, generate a radically different one. Do this until we have 5 accepted names."

**For Meeting Summaries:**
> "Read this transcript. For every paragraph, check if it adds new information using `check_novelty`. If it's novel, add it to the summary. If it's redundant, skip it."

---

## 📜 License

MIT License. Feel free to use this in your own AI pipelines.
```
