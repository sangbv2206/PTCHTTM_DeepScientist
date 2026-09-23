import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.knowledge_store import KnowledgeStore
store = KnowledgeStore()

topics = [
    "Kolmogorov Arnold Networks KAN for Symbolic Regression",
    "Neuromorphic Event-based Optical Flow Sensors",
    "Quantum Game Theory for Autonomous Drone Swarms",
    "DNA Storage with Fountain Codes and Error Correcting Nanopore Reads",
    "Perovskite Silicon Tandem Photovoltaics Halide Segregation"
]

for t in topics:
    res = store.assess_topic_coverage(t)
    print(f"{t} --> {res['status']} ({res['coverage_percent']}%) | Missing: {res.get('missing_keywords')}")
