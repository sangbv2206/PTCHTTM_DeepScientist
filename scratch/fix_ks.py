import os

with open('core/knowledge_store.py', 'rb') as f:
    raw = f.read()

text = raw.decode('utf-8', errors='replace')

target_start = '        # Hợp nhất: Ưu tiên các khái niệm sát domain bài báo trước, sau đó tới KG'
target_end = '    def _compute_paper_scores('

idx_start = text.find(target_start)
idx_end = text.find(target_end)
print(f"idx_start: {idx_start}, idx_end: {idx_end}")

new_block = '''        # Hợp nhất: Ưu tiên các khái niệm sát domain bài báo trước, sau đó tới KG
        final_entities = []
        for ent in domain_concepts + discovered:
            if ent not in final_entities and len(ent) > 2:
                final_entities.append(ent)
        return final_entities[:top_k]

    def assess_topic_coverage(self, topic: str) -> Dict[str, Any]:
        """
        Kiểm tra độ bao phủ của corpus đối với chủ đề nghiên cứu.

        Thuật toán đo coverage mới (v2):
        - Đếm số lượng bài trong TOÀN corpus có chứa từng từ khóa cốt lõi (via doc_freq).
        - Một từ khóa được coi là 'có trong corpus' nếu có >= MIN_PAPERS_PER_KEYWORD bài chứa nó.
        - coverage_ratio = (số từ khóa được corpus hỗ trợ) / (tổng số từ khóa cốt lõi).

        Lý do đổi phương pháp:
        - Phương pháp cũ so sánh query với bài top-1 -> false-MODERATE khi top-1 khớp
          các từ chung ('scientific', 'learning') nhưng không liên quan đến topic thực sự.
        - Phương pháp mới dùng tần suất toàn corpus -> phản ánh đúng dữ liệu thực có.
        """
        if not self.is_loaded:
            self.load_all_data()

        # Ngưỡng tối thiểu: cần ít nhất MIN_PAPERS bài trong corpus chứa từ khóa
        # mới coi là corpus 'có hỗ trợ' từ khóa đó
        MIN_PAPERS_PER_KEYWORD = 3

        query_words = [w for w in re.findall(r"\\b[a-z]{2,}\\b", topic.lower()) if w not in STOPWORDS]
        if not query_words:
            query_words = [w for w in re.findall(r"\\b[a-z]{2,}\\b", topic.lower())]

        unique_q_words = list(dict.fromkeys(query_words))

        # Đo coverage qua doc_freq của toàn corpus — không phụ thuộc vào top-1 paper
        matched_q = [
            w for w in unique_q_words
            if self.doc_freq.get(w, 0) >= MIN_PAPERS_PER_KEYWORD
        ]
        missing_q = [w for w in unique_q_words if w not in matched_q]
        coverage_ratio = len(matched_q) / max(len(unique_q_words), 1)

        # Lấy top papers để cung cấp context (không dùng để tính coverage nữa)
        matches = self.search_target_papers(topic, top_k=5)
        top_paper = matches[0] if matches else None

        if coverage_ratio >= 0.40:
            status = "STRONG"
            warning = None
            solutions = ["Cơ sở dữ liệu hỗ trợ tốt, có đầy đủ tài liệu nền tảng để trích dẫn và đối sánh."]
        elif coverage_ratio >= 0.20:
            status = "MODERATE"
            warning = (
                f"Chủ đề '{topic}' có dữ liệu liên quan một phần "
                f"({int(coverage_ratio*100)}% từ khóa cốt lõi có trong corpus). "
                f"Một số khía cạnh có thể thiếu tài liệu trực tiếp."
            )
            solutions = [
                "Hệ thống sẽ kết hợp bài báo gần nhất với khả năng suy luận mở của Gemini để bổ sung khoảng trống kiến thức.",
                "Có thể bổ sung thêm các bài báo chuyên sâu vào data/papers.jsonl để tăng độ chính xác."
            ]
        else:
            status = "INSUFFICIENT"
            missing_str = ', '.join(missing_q[:8]) + (" ..." if len(missing_q) > 8 else "")
            warning = (
                f"Chủ đề '{topic}' CHƯA ĐỦ THÔNG TIN trong cơ sở dữ liệu nội bộ "
                f"(chỉ {int(coverage_ratio*100)}% từ khóa cốt lõi được corpus hỗ trợ, "
                f"thiếu: {missing_str})."
            )
            solutions = [
                "Hướng 1: Chuyển sang chế độ Open-World LLM Synthesis (LLM sinh Literature Review từ tri thức toàn cầu thay vì bám vào corpus offline).",
                "Hướng 2: Bổ sung các bài báo chuyên ngành về chủ đề này vào data/papers.jsonl.",
                "Hướng 3: Thu hẹp hoặc điều chỉnh đề tài về các miền bài toán có sẵn nhiều tài liệu trong kho."
            ]

        return {
            "status": status,
            "coverage_score": coverage_ratio,
            "coverage_percent": round(coverage_ratio * 100, 1),
            "matched_keywords": matched_q,
            "missing_keywords": missing_q,
            "best_paper": top_paper,
            "candidate_papers": matches,
            "num_candidates": len(matches),
            "warning": warning,
            "solutions": solutions
        }


'''

if idx_start != -1 and idx_end != -1:
    new_text = text[:idx_start] + new_block + text[idx_end:]
    with open('core/knowledge_store.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("SUCCESS: knowledge_store.py updated cleanly.")
else:
    print("FAILED: Could not find markers.")
