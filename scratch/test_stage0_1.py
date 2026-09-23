import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
from pipeline.orchestrator import DeepScientistPipeline

p = DeepScientistPipeline(api_key=os.getenv("GEMINI_API_KEY"))
topic = "The Effects of Air Pollution on Respiratory Health"
print("\n=== TESTING STAGE 0 ===")
cov = p.knowledge_store.assess_topic_coverage(topic)
suit = p.template_manager.assess_template_suitability(topic, llm=p.llm)
print(f"Coverage: {cov['status']} ({cov['coverage_percent']}%)")
print(f"Template Selected: {suit['selected_template']} (Confidence: {suit['confidence']}/10)")
print(f"Reasoning: {suit['reasoning']}")

if cov["status"] == "INSUFFICIENT":
    print("\n=== TESTING OPEN-WORLD SYNTHESIS (STAGE 1) ===")
    ow = p.knowledge_store.synthesize_open_world_seed_paper(topic, llm=p.llm)
    print(f"Seed paper title: {ow['target_paper']['title']}")
    print(f"Venue: {ow['target_paper']['venue']} ({ow['target_paper']['year']})")
    print(f"Entities: {ow['entities'][:5]}")
