import sys, os
sys.path.insert(0, os.path.abspath("."))
from agents.research_agents import ProblemValidator, MethodValidator

class MockLLM:
    def generate(self, p):
        return '{"scores": {"topic_fidelity": 5.0, "clarity": 5.0, "relevance": 5.0, "originality": 5.0, "feasibility": 5.0, "significance": 5.0, "soundness": 5.0, "novelty": 5.0, "effectiveness": 5.0}, "average_score": 5.0, "feedbacks": "Great"}'

pv = ProblemValidator(MockLLM())
res_pv = pv.run({'user_topic': 'KAN', 'problem': 'Test problem', 'problem_rationale': 'Test rationale'})
print('ProblemValidator run succeeded:', res_pv['average_score'])

mv = MethodValidator(MockLLM())
res_mv = mv.run({'user_topic': 'KAN', 'method': 'Test method', 'mathematical_formulation': 'Test math', 'method_rationale': 'Test rationale'})
print('MethodValidator run succeeded:', res_mv['average_score'])
print("ALL VALIDATOR RUN TESTS PASSED!")
