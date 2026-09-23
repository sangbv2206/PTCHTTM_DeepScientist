import sys
import os

with open('core/knowledge_store.py', 'rb') as f:
    text = f.read().decode('utf-8')

idx1 = text.find('    def assess_topic_coverage(')
idx2 = text.find('    def _compute_paper_scores(')

new_code = '''    def assess_topic_coverage(self, topic: str) -> Dict[str, Any]:
        """
        Kiểm tra độ bao phủ của corpus đối với chủ đề nghiên cứu.

        Thuật toán đo coverage kép (v2):
        1. Tần suất từ khóa toàn corpus (doc_freq): Cần >= MIN_PAPERS_PER_KEYWORD (15 bài)
           để được coi là từ khóa có nền tảng học thuật trong kho.
        2. Relevance score của bài báo top-1 (top_score từ BM25+IDF+N-gram):
           - Nếu top_score < 35.0 -> Dù có từ khóa ngẫu nhiên, không có bài báo nào thực sự khớp đề tài.
           - Nếu top_score >= 50.0 và coverage >= 50% -> STRONG (đầy đủ tài liệu).
           - Nếu top_score >= 35.0 và coverage >= 25% -> MODERATE.
           - Dưới các ngưỡng trên -> INSUFFICIENT (kích hoạt External Retrieval).
        """
        if not self.is_loaded:
            self.load_all_data()

        # Ngưỡng tối thiểu: trong 38.000 bài báo, từ khóa cần xuất hiện >= 15 bài
        MIN_PAPERS_PER_KEYWORD = 15

        query_words = [w for w in re.findall(r"\\b[a-z]{2,}\\b", topic.lower()) if w not in STOPWORDS]
        if not query_words:
            query_words = [w for w in re.findall(r"\\b[a-z]{2,}\\b", topic.lower())]

        unique_q_words = list(dict.fromkeys(query_words))

        matched_q = [
            w for w in unique_q_words
            if self.doc_freq.get(w, 0) >= MIN_PAPERS_PER_KEYWORD
        ]
        missing_q = [w for w in unique_q_words if w not in matched_q]
        coverage_ratio = len(matched_q) / max(len(unique_q_words), 1)

        # Lấy top papers kèm điểm relevance
        scored_matches = self.search_target_papers_with_scores(topic, top_k=5)
        top_score = scored_matches[0][0] if scored_matches else 0.0
        matches = [p for _, p in scored_matches]
        top_paper = matches[0] if matches else None

        if coverage_ratio >= 0.50 and top_score >= 50.0:
            status = "STRONG"
            warning = None
            solutions = ["Cơ sở dữ liệu hỗ trợ tốt, có đầy đủ tài liệu nền tảng để trích dẫn và đối sánh."]
        elif (coverage_ratio >= 0.25 and top_score >= 35.0) or top_score >= 45.0:
            status = "MODERATE"
            warning = (
                f"Chủ đề '{topic}' có dữ liệu liên quan một phần "
                f"({int(coverage_ratio*100)}% từ khóa có trong corpus, top score: {top_score:.1f}). "
                f"Một số khía cạnh chuyên sâu có thể cần tra cứu thêm."
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
                f"top relevance score={top_score:.1f} < 35.0, thiếu: {missing_str})."
            )
            solutions = [
                "Hướng 1: Kích hoạt Dynamic External Retrieval từ OpenAlex / arXiv để lấy bài báo thực tế mới nhất.",
                "Hướng 2: Chuyển sang chế độ Open-World LLM Synthesis (tổng hợp từ tri thức toàn cầu của mô hình).",
                "Hướng 3: Thu hẹp hoặc điều chỉnh đề tài về các miền bài toán có sẵn nhiều tài liệu trong kho."
            ]

        return {
            "status": status,
            "coverage_score": coverage_ratio,
            "coverage_percent": round(coverage_ratio * 100, 1),
            "top_relevance_score": round(top_score, 1),
            "matched_keywords": matched_q,
            "missing_keywords": missing_q,
            "best_paper": top_paper,
            "candidate_papers": matches,
            "num_candidates": len(matches),
            "warning": warning,
            "solutions": solutions
        }


'''

if idx1 != -1 and idx2 != -1:
    updated_text = text[:idx1] + new_code + text[idx2:]
    with open('core/knowledge_store.py', 'w', encoding='utf-8') as f:
        f.write(updated_text)
    print("SUCCESS: Updated assess_topic_coverage via indices!")
else:
    print(f"FAILED: idx1={idx1}, idx2={idx2}")
