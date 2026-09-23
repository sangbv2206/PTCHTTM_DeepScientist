import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge_store import KnowledgeStore
from templates.template_manager import TemplateManager

def test_all():
    print("=== Testing KnowledgeStore ===")
    ks = KnowledgeStore()
    topic = "Kolmogorov-Arnold Networks KAN for Scientific Machine Learning"
    cov = ks.assess_topic_coverage(topic)
    print(f"Topic: {topic}")
    print(f"Status: {cov.get('status')}")
    print(f"Coverage percent: {cov.get('coverage_percent')}%")
    print(f"Found keywords: {cov.get('found_keywords')}")
    print(f"Missing keywords: {cov.get('missing_keywords')}")

    papers = ks.search_target_papers_with_scores(topic, top_k=5)
    print(f"\nTop 5 papers with scores for topic:")
    for s, p in papers:
        title = p.get('title', 'N/A')
        print(f"  Score: {s:6.1f} | Title: {title[:70]}")

    print("\n=== Testing TemplateManager Scoring Router ===")
    tm = TemplateManager()
    topics = [
        "Kolmogorov-Arnold Networks KAN for Scientific Machine Learning",
        "Time Series Energy Consumption Forecasting",
        "Target Trial Emulation for SGLT2 inhibitors and cardiovascular risk",
        "Deep Q-Network Reinforcement Learning for Robot Cartpole",
        "Convolutional Image Classification on CIFAR with MobileNetV3",
        "Graph Neural Network for Node Classification on Citation Graph",
        "2D Diffusion Model for Generative Denoising",
        "Epidemic spread SEIR model simulation with ODEs",
        "Grokking and delayed generalization in algorithmic transformers"
    ]
    for t in topics:
        selected = tm.select_best_template_for_topic(t)
        print(f"  Topic: {t[:45]:<45} -> Template: {selected}")

if __name__ == "__main__":
    test_all()
