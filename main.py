import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure current project directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipeline.orchestrator import DeepScientistPipeline


def main():
    parser = argparse.ArgumentParser(
        description="DeepScientist: Hệ thống nghiên cứu khoa học tự động kết hợp ResearchAgent & Sakana AI (Chi phí 0đ với Gemini API)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Graph Neural Networks for Recommendation and Representation Learning",
        help="Chủ đề nghiên cứu khoa học cần thực hiện"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Gemini API Key (Lấy miễn phí tại https://aistudio.google.com/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Tên model Gemini sử dụng (mặc định: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục data của ResearchAgent (mặc định: ../ResearchAgent/data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục xuất kết quả (mặc định: ./output)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Số vòng lặp phản biện học thuật cho vấn đề và phương pháp (mặc định: 2)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Chỉ định template thực nghiệm (ví dụ: gnn_graph_learning, recommender_system, nanoGPT, 2d_diffusion,...)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Chế độ hoàn toàn ngoại tuyến: chỉ dùng kho bài báo có sẵn trong data/, không gọi OpenAlex/arXiv API"
    )
    parser.add_argument(
        "--semantic-scholar-key",
        type=str,
        default=None,
        help="Semantic Scholar API Key tùy chọn (đăng ký miễn phí tại semanticscholar.org)"
    )

    args = parser.parse_args()

    # Load .env nếu có
    env_file = os.path.join(current_dir, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    s2_key = args.semantic_scholar_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if not api_key:
        print("\n" + "!" * 70)
        print(" [LỖI] Chưa tìm thấy Google Gemini API Key!")
        print(" Bạn có thể cung cấp API Key bằng 1 trong 3 cách sau:")
        print("   1. Truyền tham số: python main.py --api-key YOUR_KEY --topic \"...\"")
        print("   2. Tạo file .env chứa: GEMINI_API_KEY=YOUR_KEY")
        print("   3. Thiết lập biến môi trường: $env:GEMINI_API_KEY=\"YOUR_KEY\"")
        print(" (Đăng ký lấy key 0đ hoàn toàn miễn phí tại: https://aistudio.google.com/)")
        print("!" * 70 + "\n")
        sys.exit(1)

    try:
        pipeline = DeepScientistPipeline(
            api_key=api_key,
            model_name=args.model,
            data_dir=args.data_dir,
            output_base_dir=args.output_dir,
            online_search=not args.offline,
            semantic_scholar_key=s2_key,
        )
        pipeline.run(
            topic=args.topic,
            template_name=args.template,
            max_iterations=args.iterations
        )

    except Exception as e:
        print(f"\n[ERROR] Quá trình thực hiện gặp lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
