import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from templates.template_manager import TemplateManager
from core.knowledge_store import KnowledgeStore

tm = TemplateManager()
topic = "Deep Reinforcement Learning for Continuous Control and Decision Making"

selected = tm.select_best_template_for_topic(topic)
suitability = tm.assess_template_suitability(topic)
print("=== TEMPLATE ROUTER TEST ===")
print(f"Selected Template: {selected}")
print(f"Is Supported: {suitability['is_supported']}")
print(f"Confidence: {suitability['confidence']}/10")
print(f"Reasoning: {suitability['reasoning']}")

print("\n=== KNOWLEDGE STORE TEST ===")
ks = KnowledgeStore()
ks.load_all_data()
cov = ks.assess_topic_coverage(topic)
print(f"Coverage Status: {cov['status']} ({cov['coverage_percent']}%)")
print(f"Matched Keywords: {cov['matched_keywords']}")
if cov.get("best_paper"):
    print(f"Best Paper Found: {cov['best_paper'].get('title')} ({cov['best_paper'].get('year')})")
    print(f"Authors: {[a.get('name') for a in cov['best_paper'].get('authors', [])]}")
    print(f"Citations: {cov['best_paper'].get('citationcount')}")
