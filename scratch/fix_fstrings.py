ra_path = r"d:\PTIT\Y3-1\PTCHTTM\DeepScientist\agents\research_agents.py"
with open(ra_path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix 1: In ProblemValidator
bad_pv = '''6. Significance (1-5): Will solving it advance the field?

Respond strictly in JSON format:
```json
{
  "scores": {
    "topic_fidelity": 4.8,
    "clarity": 4.5,
    "relevance": 4.5,
    "originality": 4.0,
    "feasibility": 4.5,
    "significance": 4.2
  },
  "average_score": 4.41,
  "feedbacks": "Constructive critique highlighting specific improvements needed"
}
```
"""'''

good_pv = '''6. Significance (1-5): Will solving it advance the field?
""" + """
Respond strictly in JSON format:
```json
{
  "scores": {
    "topic_fidelity": 4.8,
    "clarity": 4.5,
    "relevance": 4.5,
    "originality": 4.0,
    "feasibility": 4.5,
    "significance": 4.2
  },
  "average_score": 4.41,
  "feedbacks": "Constructive critique highlighting specific improvements needed"
}
```
"""'''

# Fix 2: In MethodValidator
bad_mv = '''6. Effectiveness (1-5): Is it well-suited to solve the underlying problem?

Respond strictly in JSON format:
```json
{
  "scores": {
    "topic_fidelity": 4.8,
    "clarity": 4.5,
    "soundness": 4.2,
    "novelty": 4.0,
    "feasibility": 4.8,
    "effectiveness": 4.3
  },
  "average_score": 4.43,
  "feedbacks": "Constructive critique to improve mathematical rigor"
}
```
"""'''

good_mv = '''6. Effectiveness (1-5): Is it well-suited to solve the underlying problem?
""" + """
Respond strictly in JSON format:
```json
{
  "scores": {
    "topic_fidelity": 4.8,
    "clarity": 4.5,
    "soundness": 4.2,
    "novelty": 4.0,
    "feasibility": 4.8,
    "effectiveness": 4.3
  },
  "average_score": 4.43,
  "feedbacks": "Constructive critique to improve mathematical rigor"
}
```
"""'''

text_norm = text.replace("\r\n", "\n")
bad_pv_norm = bad_pv.replace("\r\n", "\n")
bad_mv_norm = bad_mv.replace("\r\n", "\n")

if bad_pv_norm in text_norm:
    text_norm = text_norm.replace(bad_pv_norm, good_pv)
    print("Fixed ProblemValidator f-string")
else:
    print("Could not find bad_pv")

if bad_mv_norm in text_norm:
    text_norm = text_norm.replace(bad_mv_norm, good_mv)
    print("Fixed MethodValidator f-string")
else:
    print("Could not find bad_mv")

with open(ra_path, "w", encoding="utf-8") as f:
    f.write(text_norm)
