import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.knowledge_store import KnowledgeStore

def test_enrichment():
    topic = "Mamba State Space Models for Sequence Modeling"
    print(f"=== TEST KIEM TRA ENRICHMENT VOI TOPIC LA: '{topic}' ===")
    
    store = KnowledgeStore()
    cov = store.assess_topic_coverage(topic)
    print(f"1. Do bao phu offline: {cov['status']} ({cov['coverage_percent']}%)")

    print("\n2. Goi enrich_with_external_papers()...")
    res = store.enrich_with_external_papers(topic, top_k=5)
    
    assert res is not None and "target_paper" in res, "Failed to get target paper!"
    target = res["target_paper"]
    refs = res["references"]
    entities = res["entities"]

    print(f"\n3. KET QUA KEO DU LIEU THAT:")
    print(f"   -> Seed Paper: {target.get('title')}")
    print(f"   -> Authors: {[a.get('name') for a in target.get('authors', [])[:3]]}")
    print(f"   -> Venue: {target.get('venue')}")
    print(f"   -> Year: {target.get('year')}")
    print(f"   -> Source: {target.get('source')}")
    print(f"   -> So bai tham chieu di kem: {len(refs)}")
    for i, ref in enumerate(refs, 1):
        print(f"      [{i}] {ref.get('title')} ({ref.get('year')})")

    print(f"\n4. Kiem tra sinh BibTeX entry:")
    bib = store.generate_bibtex(target, cite_key="seed_paper_2024")
    print(bib)

    print(f"\n5. Kiem tra so thuc the khoa hoc khai pha: {len(entities)}")
    print(f"   Top entities: {entities[:8]}")
    
    print("\n=== TEST THANH CONG HOAN HAO ===")

if __name__ == "__main__":
    test_enrichment()
