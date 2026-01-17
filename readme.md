![geometric_gating_diagram](https://github.com/user-attachments/assets/cbb4f5f3-c516-4225-a2e1-6257a1aab22c)
<svg viewBox="0 0 1200 800" xmlns="http://www.w3.org/2000/svg">
  <!-- Title -->
  <text x="600" y="30" font-size="24" font-weight="bold" text-anchor="middle" fill="#2c3e50">
    Geometric Gating Framework: From Theory to Applications
  </text>
  
  <!-- Central Core Box -->
  <g id="core">
    <!-- Outer glow -->
    <ellipse cx="600" cy="150" rx="180" ry="90" fill="url(#coreGradient)" opacity="0.3"/>
    
    <!-- Main box -->
    <rect x="440" y="80" width="320" height="140" rx="15" fill="url(#coreGradient)" stroke="#2c3e50" stroke-width="3"/>
    
    <!-- Core content -->
    <text x="600" y="115" font-size="18" font-weight="bold" text-anchor="middle" fill="white">
      GEOMETRIC GATING CORE
    </text>
    <text x="600" y="140" font-size="13" text-anchor="middle" fill="white">
      High-D Input (384d) → Projection (7d) → Distance Filter (ε≈1.1)
    </text>
    <text x="600" y="165" font-size="12" text-anchor="middle" fill="#ecf0f1" font-style="italic">
      Capacity ≈ 7 items | Noise-robust separation
    </text>
    <text x="600" y="195" font-size="14" font-weight="bold" text-anchor="middle" fill="#f39c12">
      C(d,ε) = max{|R| : ∀i≠j, ||r_i - r_j|| ≥ ε}
    </text>
  </g>
  
  <!-- Connecting lines hub -->
  <circle cx="600" cy="280" r="8" fill="#34495e"/>
  
  <!-- Application Nodes -->
  <!-- Row 1 -->
  <g id="app1">
    <line x1="600" y1="280" x2="150" y2="380" stroke="#3498db" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="30" y="350" width="240" height="140" rx="10" fill="#3498db" stroke="#2c3e50" stroke-width="2"/>
    <text x="150" y="375" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      📚 Knowledge Base Dedup
    </text>
    <text x="150" y="400" font-size="11" text-anchor="middle" fill="white">
      Problem: Redundant documents
    </text>
    <text x="150" y="420" font-size="11" text-anchor="middle" fill="white">
      Solution: Geometric filtering
    </text>
    <text x="150" y="450" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: -40% storage cost
    </text>
    <text x="150" y="470" font-size="10" text-anchor="middle" fill="#ecf0f1">
      F1 = 0.954 | No training needed
    </text>
  </g>
  
  <g id="app2">
    <line x1="600" y1="280" x2="400" y2="380" stroke="#9b59b6" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="290" y="350" width="220" height="140" rx="10" fill="#9b59b6" stroke="#2c3e50" stroke-width="2"/>
    <text x="400" y="375" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      🔍 Query Expansion
    </text>
    <text x="400" y="400" font-size="11" text-anchor="middle" fill="white">
      Problem: Over-expansion
    </text>
    <text x="400" y="420" font-size="11" text-anchor="middle" fill="white">
      Solution: Distinct term filter
    </text>
    <text x="400" y="450" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: +17% precision
    </text>
    <text x="400" y="470" font-size="10" text-anchor="middle" fill="#ecf0f1">
      Precision@10: 0.67 → 0.84
    </text>
  </g>
  
  <g id="app3">
    <line x1="600" y1="280" x2="600" y2="380" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="520" y="350" width="160" height="140" rx="10" fill="#e74c3c" stroke="#2c3e50" stroke-width="2"/>
    <text x="600" y="375" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      🤖 Chatbot Memory
    </text>
    <text x="600" y="400" font-size="11" text-anchor="middle" fill="white">
      Problem: Context bloat
    </text>
    <text x="600" y="420" font-size="11" text-anchor="middle" fill="white">
      Solution: Auto-prune
    </text>
    <text x="600" y="450" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: -35% tokens
    </text>
    <text x="600" y="470" font-size="10" text-anchor="middle" fill="#ecf0f1">
      Maintains ~7 distinct turns
    </text>
  </g>
  
  <g id="app4">
    <line x1="600" y1="280" x2="800" y2="380" stroke="#1abc9c" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="700" y="350" width="200" height="140" rx="10" fill="#1abc9c" stroke="#2c3e50" stroke-width="2"/>
    <text x="800" y="375" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      🎓 Curriculum Learning
    </text>
    <text x="800" y="400" font-size="11" text-anchor="middle" fill="white">
      Problem: Redundant batches
    </text>
    <text x="800" y="420" font-size="11" text-anchor="middle" fill="white">
      Solution: Diverse sampling
    </text>
    <text x="800" y="450" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: +23% convergence
    </text>
    <text x="800" y="470" font-size="10" text-anchor="middle" fill="#ecf0f1">
      Better generalization +3.2%
    </text>
  </g>
  
  <g id="app5">
    <line x1="600" y1="280" x2="1050" y2="380" stroke="#f39c12" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="930" y="350" width="240" height="140" rx="10" fill="#f39c12" stroke="#2c3e50" stroke-width="2"/>
    <text x="1050" y="375" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      📝 Document Summarization
    </text>
    <text x="1050" y="400" font-size="11" text-anchor="middle" fill="white">
      Problem: Repetitive content
    </text>
    <text x="1050" y="420" font-size="11" text-anchor="middle" fill="white">
      Solution: Sentence diversity
    </text>
    <text x="1050" y="450" font-size="13" font-weight="bold" text-anchor="middle" fill="#2c3e50">
      Impact: +0.12 ROUGE-L
    </text>
    <text x="1050" y="470" font-size="10" text-anchor="middle" fill="white">
      Human preference +18%
    </text>
  </g>
  
  <!-- Row 2 -->
  <g id="app6">
    <line x1="600" y1="280" x2="150" y2="580" stroke="#16a085" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="30" y="550" width="240" height="140" rx="10" fill="#16a085" stroke="#2c3e50" stroke-width="2"/>
    <text x="150" y="575" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      🛒 Recommendations
    </text>
    <text x="150" y="600" font-size="11" text-anchor="middle" fill="white">
      Problem: Filter bubbles
    </text>
    <text x="150" y="620" font-size="11" text-anchor="middle" fill="white">
      Solution: Diverse ranking
    </text>
    <text x="150" y="650" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: +12% engagement
    </text>
    <text x="150" y="670" font-size="10" text-anchor="middle" fill="#ecf0f1">
      CTR improved +8%
    </text>
  </g>
  
  <g id="app7">
    <line x1="600" y1="280" x2="400" y2="580" stroke="#d35400" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="290" y="550" width="220" height="140" rx="10" fill="#d35400" stroke="#2c3e50" stroke-width="2"/>
    <text x="400" y="575" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      🧪 Test Generation
    </text>
    <text x="400" y="600" font-size="11" text-anchor="middle" fill="white">
      Problem: Redundant tests
    </text>
    <text x="400" y="620" font-size="11" text-anchor="middle" fill="white">
      Solution: Diverse coverage
    </text>
    <text x="400" y="650" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: -35% test time
    </text>
    <text x="400" y="670" font-size="10" text-anchor="middle" fill="#ecf0f1">
      Coverage maintained
    </text>
  </g>
  
  <g id="app8">
    <line x1="600" y1="280" x2="600" y2="580" stroke="#8e44ad" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="490" y="550" width="220" height="140" rx="10" fill="#8e44ad" stroke="#2c3e50" stroke-width="2"/>
    <text x="600" y="575" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      📄 Literature Review
    </text>
    <text x="600" y="600" font-size="11" text-anchor="middle" fill="white">
      Problem: Paper overload
    </text>
    <text x="600" y="620" font-size="11" text-anchor="middle" fill="white">
      Solution: Novelty ranking
    </text>
    <text x="600" y="650" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: -40% review time
    </text>
    <text x="600" y="670" font-size="10" text-anchor="middle" fill="#ecf0f1">
      Better gap identification
    </text>
  </g>
  
  <g id="app9">
    <line x1="600" y1="280" x2="820" y2="580" stroke="#c0392b" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="730" y="550" width="180" height="140" rx="10" fill="#c0392b" stroke="#2c3e50" stroke-width="2"/>
    <text x="820" y="575" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      💡 Brainstorming
    </text>
    <text x="820" y="600" font-size="11" text-anchor="middle" fill="white">
      Problem: Groupthink
    </text>
    <text x="820" y="620" font-size="11" text-anchor="middle" fill="white">
      Solution: Idea filtering
    </text>
    <text x="820" y="650" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: Higher diversity
    </text>
    <text x="820" y="670" font-size="10" text-anchor="middle" fill="#ecf0f1">
      Real-time feedback
    </text>
  </g>
  
  <g id="app10">
    <line x1="600" y1="280" x2="1030" y2="580" stroke="#27ae60" stroke-width="2" stroke-dasharray="5,5"/>
    <rect x="940" y="550" width="180" height="140" rx="10" fill="#27ae60" stroke="#2c3e50" stroke-width="2"/>
    <text x="1030" y="575" font-size="14" font-weight="bold" text-anchor="middle" fill="white">
      📓 Note Taking
    </text>
    <text x="1030" y="600" font-size="11" text-anchor="middle" fill="white">
      Problem: Note clutter
    </text>
    <text x="1030" y="620" font-size="11" text-anchor="middle" fill="white">
      Solution: Smart creation
    </text>
    <text x="1030" y="650" font-size="13" font-weight="bold" text-anchor="middle" fill="#f1c40f">
      Impact: -35% duplicates
    </text>
    <text x="1030" y="670" font-size="10" text-anchor="middle" fill="#ecf0f1">
      Better organization
    </text>
  </g>
  
  <!-- Bottom caption -->
  <text x="600" y="740" font-size="16" font-weight="bold" text-anchor="middle" fill="#2c3e50">
    One Framework → Ten Domains → Quantified Impact
  </text>
  <text x="600" y="765" font-size="12" text-anchor="middle" fill="#7f8c8d">
    From cognitive theory to practical tools: Geometric constraints as a universal filtering principle
  </text>
  
  <!-- Legend -->
  <rect x="30" y="720" width="15" height="15" fill="#3498db"/>
  <text x="50" y="732" font-size="11" fill="#2c3e50">Enterprise/Data</text>
  
  <rect x="180" y="720" width="15" height="15" fill="#9b59b6"/>
  <text x="200" y="732" font-size="11" fill="#2c3e50">Search/IR</text>
  
  <rect x="300" y="720" width="15" height="15" fill="#e74c3c"/>
  <text x="320" y="732" font-size="11" fill="#2c3e50">AI/ML</text>
  
  <rect x="390" y="720" width="15" height="15" fill="#1abc9c"/>
  <text x="410" y="732" font-size="11" fill="#2c3e50">Training</text>
  
  <rect x="510" y="720" width="15" height="15" fill="#f39c12"/>
  <text x="530" y="732" font-size="11" fill="#2c3e50">Content</text>
  
  <!-- Gradients -->
  <defs>
    <linearGradient id="coreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#2c3e50;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#34495e;stop-opacity:1" />
    </linearGradient>
  </defs>
</svg>
