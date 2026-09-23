DeepScientist — Hệ Thống Tự Động Hóa Nghiên Cứu Khoa Học Bằng Mô Hình Ngôn Ngữ Lớn

<p align="center">
  <b>Hệ thống nghiên cứu khoa học tự chủ (Autonomous Scientific Discovery) tích hợp Multi-Agent LLM</b><br>
  <i>Tự động phát hiện khoảng trống nghiên cứu, đề xuất giải pháp toán học, lập trình thực nghiệm PyTorch, xuất bài báo LaTeX/PDF và phản biện học thuật độc lập.</i>
</p>

1. Tổng quan dự án

DeepScientist là một hệ thống AI đa tác nhân (Multi-Agent System) được thiết kế để tự động hóa một chu trình nghiên cứu khoa học từ hình thành ý tưởng, thẩm định vấn đề, phát triển phương pháp, thực nghiệm, soạn thảo bài báo đến phản biện. Hệ thống kế thừa và tích hợp tinh hoa từ hai công trình nghiên cứu nổi tiếng thế giới:

ResearchAgent (NAACL 2025): Cơ chế khai phá tri thức học thuật từ cơ sở dữ liệu thực tế (hàng chục nghìn bài báo và đồ thị thực thể liên kết), ứng dụng mô hình tranh biện đa tác nhân (Multi-Agent Debate) để phát hiện khoảng trống nghiên cứu (research gap), định hình vấn đề và tinh chỉnh phương pháp luận.

The AI Scientist (Sakana AI, 2024): Kiến trúc tự động hóa thực nghiệm: tự sinh mã nguồn mô hình, chạy huấn luyện định lượng với PyTorch, tự động vẽ biểu đồ đối sánh khoa học đa chiều (multi-figure plotting), biên soạn bài báo toàn văn chuẩn học thuật bằng LaTeX/PDF và mô phỏng hội đồng phản biện mù đôi (Double-Blind Peer Review).

Công nghệ chính

Python 3.10: môi trường chạy chính của hệ thống.

Multi-Agent LLM: điều phối các tác nhân chuyên trách cho từng giai đoạn nghiên cứu.

OpenAlex / arXiv / Semantic Scholar: nguồn tìm kiếm và bổ sung tài liệu học thuật.

PyTorch: triển khai và chạy thực nghiệm học máy.

Matplotlib: trực quan hóa kết quả thực nghiệm.

LaTeX / PDFLaTeX: biên soạn và biên dịch bài báo PDF.

Docker / Docker Compose: đóng gói môi trường và chạy hệ thống nhất quán.

Kiến trúc quy trình

Kho tri thức học thuật
        │
        ▼
1. Xác định vấn đề nghiên cứu
        │
        ▼
2. Phản biện & thẩm định đề tài
        │
        ▼
3. Phát triển phương pháp mới
        │
        ▼
4. Phản biện phương pháp & toán học
        │
        ▼
5. Lập trình thực nghiệm & trực quan hóa
        │
        ▼
6. Soạn thảo LaTeX & biên dịch PDF
        │
        ▼
7. Phản biện học thuật độc lập

2. Quy trình nghiên cứu khép kín 7 giai đoạn

Kho Tri Thức Học Thuật (Knowledge Store & OpenAlex/arXiv)

Giai đoạn 1: Xác định vấn đề nghiên cứu (Ideation Agent)
Giai đoạn 2: Phản biện & thẩm định đề tài (Validator Agent)
Giai đoạn 3: Phát triển phương pháp mới (Method Agent)
Giai đoạn 4: Phản biện phương pháp & toán học (Validator Agent)
Giai đoạn 5: Lập trình thực nghiệm, huấn luyện & trực quan hóa (Coder Agent)
Giai đoạn 6: Soạn thảo toàn văn LaTeX & tự động biên dịch (LaTeX Writer Agent)
Giai đoạn 7: Phản biện học thuật độc lập (Peer Reviewer)

### Chi tiết các giai đoạn:
- **Giai đoạn 1 (Problem Identification)**: Khai phá cơ sở tri thức cục bộ và tìm kiếm trực tuyến (OpenAlex, arXiv, Semantic Scholar) để tìm các bài báo hạt giống (seed papers), phân tích hạn chế của các phương pháp hiện tại và đề xuất vấn đề nghiên cứu mới.
- **Giai đoạn 2 (Problem Validation)**: Hội đồng phản biện đánh giá vấn đề qua 5 tiêu chí: *Tính rõ ràng (Clarity)*, *Tính khả thi (Feasibility)*, *Tính mới (Novelty)*, *Tính phù hợp (Relevance)* và *Ý nghĩa thực tiễn (Significance)* qua nhiều vòng tranh biện.
- **Giai đoạn 3 (Method Development)**: Thiết kế giải pháp kỹ thuật chi tiết, xây dựng công thức toán học chặt chẽ, luận giải trực giác lý thuyết và đặt tên thuật toán mới.
- **Giai đoạn 4 (Method Validation)**: Phản biện tính đúng đắn toán học, độ phức tạp tính toán và tính khả thi khi cài đặt thực nghiệm.
- **Giai đoạn 5 (Experimentation & Plotting)**: Tự động đưa thuật toán vào mã nguồn thử nghiệm, thực thi huấn luyện mô hình với PyTorch, ghi nhận metrics qua từng epoch và vẽ 3 biểu đồ khoa học (`figure_1.png`, `figure_2.png`, `figure_3.png`).
- **Giai đoạn 6 (LaTeX Writeup & PDF Compilation)**: Viết toàn văn bài báo khoa học bằng LaTeX theo định dạng hội nghị 2 cột chuẩn mực, tích hợp tài liệu tham khảo BibTeX và gọi trình biên dịch `pdflatex` để xuất ra file `paper.pdf` hoàn chỉnh.
- **Giai đoạn 7 (Peer Review)**: Mô phỏng hội đồng chấm điểm khoa học độc lập đọc toàn bộ bài báo và kết quả định lượng, chấm điểm trên thang 1-10 và đưa ra phán quyết (Accept/Reject) kèm các phân tích sâu sắc.

---

## 3. Cấu trúc thư mục dự án


DeepScientist/
├── agents/                     # Các tác nhân AI chuyên trách
│   ├── ideation_agents.py      # Đề xuất và phản biện vấn đề nghiên cứu
│   ├── method_agents.py        # Thiết kế và phản biện phương pháp toán học
│   ├── experiment_agents.py    # Sinh mã PyTorch, huấn luyện và vẽ biểu đồ
│   ├── latex_writer.py         # Biên soạn toàn văn LaTeX và biên dịch PDF
│   └── peer_reviewer.py        # Phản biện học thuật độc lập theo chuẩn ICLR
├── core/                       # Thành phần lõi của hệ thống
│   ├── gemini_llm.py           # LLM Wrapper hỗ trợ luân chuyển model tự động
│   ├── knowledge_store.py      # Quản lý và truy vấn đồ thị tri thức bài báo
│   ├── knowledge_engine.py     # Trích xuất tri thức, sinh trích dẫn BibTeX
│   └── external_retriever.py   # Tìm kiếm trực tuyến từ OpenAlex, arXiv, Semantic Scholar
├── data/                       # Dữ liệu tri thức học thuật
│   ├── papers.jsonl            # Tập dữ liệu hàng nghìn bài báo khoa học
│   ├── knowledge.jsonl         # Đồ thị thực thể và khái niệm khoa học liên kết
│   ├── references.jsonl        # Danh mục bài báo tham khảo bổ sung
│   └── external_papers_cache.jsonl # Bộ nhớ đệm các bài báo tải từ Internet
├── pipeline/
│   └── orchestrator.py         # Bộ điều phối toàn bộ chu trình 7 giai đoạn
├── templates/                  # Các mẫu mã thực nghiệm cơ sở
│   ├── academic_paper.tex      # Mẫu LaTeX chuẩn hội nghị
│   ├── gnn_graph_learning/     # Mẫu học sâu trên đồ thị
│   ├── recommender_system/     # Mẫu hệ gợi ý
│   ├── time_series_forecasting/# Mẫu dự báo chuỗi thời gian
│   └── ...
├── output/                     # Thư mục lưu trữ thành phẩm sau mỗi lần chạy
├── Dockerfile                  # Cấu hình đóng gói ứng dụng kèm TeX Live
├── docker-compose.yml          # Cấu hình chạy nhanh bằng Docker Compose
├── main.py                     # Điểm khởi chạy chương trình (CLI)
├── requirements.txt            # Danh sách thư viện Python cần thiết
└── README.md                   # Tài liệu hướng dẫn sử dụng tiếng Việt


---

## 4. Thành phẩm đầu ra (Output Artifacts)

Mỗi lần chạy nghiên cứu, hệ thống sẽ tự động tạo một thư mục độc lập theo mốc thời gian tại `output/run_<YYYYMMDD_HHMMSS>/`. Bên trong chứa đầy đủ tất cả các thành phẩm học thuật bao gồm mã nguồn, biểu đồ, bài báo PDF và nhận xét phản biện:


output/run_20260924_060712/
├── experiment/
│   ├── experiment.py           # Mã nguồn baseline ban đầu của mẫu thực nghiệm
│   ├── experiment_upgraded.py  # Mã nguồn Python đã tích hợp thuật toán mới đề xuất
│   ├── plot.py                 # Mã nguồn tự động vẽ 3 biểu đồ chuẩn học thuật
│   ├── prompt.json             # Đặc tả chi tiết ý tưởng và chỉ dẫn cài đặt
│   ├── figure_1.png            # Biểu đồ động học huấn luyện (Loss / Accuracy / Convergence)
│   ├── figure_2.png            # Biểu đồ đối sánh hiệu năng định lượng (Đề xuất vs Baseline)
│   ├── figure_3.png            # Biểu đồ phân tích độ nhạy hoặc thử nghiệm loại trừ (Ablation)
│   ├── run_0/                  # Dữ liệu metrics và checkpoint vòng lặp huấn luyện thứ nhất
│   └── run_1/                  # Dữ liệu metrics và checkpoint vòng lặp huấn luyện thứ hai
├── latex/
│   ├── paper.tex               # Toàn văn mã nguồn bài báo định dạng LaTeX 2 cột (6-8 trang)
│   ├── references.bib          # Danh mục tài liệu trích dẫn chuẩn BibTeX đầy đủ
│   ├── paper.pdf               # BÀI BÁO KHOA HỌC HOÀN CHỈNH ĐƯỢC BIÊN DỊCH BẰNG PDFLATEX
│   ├── figure_1.png, 2, 3      # Các hình ảnh biểu đồ được nhúng trực tiếp vào văn bản
│   └── paper.aux, .bbl, .log   # Các tệp phụ trợ sinh ra trong quá trình biên dịch TeX
├── review/
│   ├── peer_review.md          # Biên bản nhận xét phản biện chi tiết theo mẫu ICLR/NeurIPS
│   └── peer_review.json        # Điểm số định lượng (Soundness, Presentation, Novelty) & phán quyết
└── preflight_assessment.json   # Bản tổng kết trạng thái thẩm định các giai đoạn của quy trình


### Chi tiết các tệp quan trọng:
- **`paper.pdf`**: Tác phẩm cốt lõi của nghiên cứu — một bài báo khoa học tiêu chuẩn dài 6-8 trang gồm đầy đủ Abstract, Introduction, Related Work, Method (có công thức toán học LaTeX đẹp mắt), Experiments (có nhúng 3 biểu đồ PNG và bảng số liệu so sánh), Conclusion và References.
- **`experiment_upgraded.py`**: File mã nguồn Python hoàn chỉnh và độc lập, người dùng có thể tự chạy lại bất cứ lúc nào bằng `python experiment_upgraded.py` để tái lập kết quả huấn luyện.
- **`figure_1.png`, `figure_2.png`, `figure_3.png`**: Bộ 3 biểu đồ chất lượng cao (300 DPI, font khoa học) phục vụ minh họa bài báo.
- **`peer_review.md`**: Đánh giá chi tiết từ "hội đồng phản biện AI" độc lập chỉ ra các điểm mạnh (Strengths), điểm yếu (Weaknesses), câu hỏi chất vấn (Questions) và khuyến nghị cải thiện.

---

## 5. Hướng dẫn sử dụng với Docker

### Cách 1: Sử dụng Docker Compose (khuyến nghị)

#### Bước 1: Chuẩn bị API key
Tạo Google Gemini API Key tại [Google AI Studio](https://aistudio.google.com/). Việc sử dụng API có thể phụ thuộc vào hạn mức và chính sách hiện hành của Google.

Tạo tệp `.env` tại thư mục gốc của dự án:
```env
GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere...

Lưu ý: Không đưa file .env chứa API key lên Git. Hãy thêm .env vào .gitignore.

Bước 2: Xây dựng Docker image

Mở terminal tại thư mục dự án và chạy:

docker compose build

Bước 3: Khởi chạy nghiên cứu

Cách A: Chạy với chủ đề mặc định cấu hình sẵn:

docker compose up

Cách B: Chạy với chủ đề nghiên cứu tùy chọn theo ý bạn (Khuyên dùng):

docker compose run --rm deepscientist --topic "Kolmogorov-Arnold Networks KAN for Scientific Machine Learning"

Hoặc thử nghiệm các đề tài học thuật khác:

docker compose run --rm deepscientist --topic "Graph Neural Networks for Drug-Target Interaction"

Ghi chú về thư mục kết quả: Nhờ cấu hình volumes: - ./output:/app/output trong docker-compose.yml, toàn bộ bài báo paper.pdf, biểu đồ .png và mã nguồn sẽ được đồng bộ trực tiếp ra thư mục output/ ngay trên máy tính của bạn khi container chạy xong!

Cách 2: Sử dụng Docker CLI

Nếu không dùng Docker Compose, bạn có thể build và run trực tiếp bằng lệnh Docker:

1. Build image:

docker build -t deepscientist:latest .

2. Run container:

Trên PowerShell (Windows):

docker run --rm -it `
  --env-file .env `
  -v ${PWD}/output:/app/output `
  -v ${PWD}/data:/app/data `
  deepscientist:latest --topic "Kolmogorov-Arnold Networks KAN for Scientific Machine Learning"

Trên Linux / macOS (Bash / Zsh):

docker run --rm -it \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \
  deepscientist:latest --topic "Kolmogorov-Arnold Networks KAN for Scientific Machine Learning"

6. Hướng dẫn cài đặt và chạy trực tiếp trên máy tính

Nếu bạn không muốn sử dụng Docker mà muốn chạy trực tiếp bằng môi trường Python trên máy:

6.1. Yêu cầu hệ thống

Python: Phiên bản 3.10 trở lên.

Trình biên dịch TeX (Bắt buộc để biên dịch PDF):

Trên Windows: Cài đặt MiKTeX hoặc TeX Live. Đảm bảo lệnh pdflatex đã có trong biến môi trường PATH.

Trên Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended

Trên macOS: Cài đặt MacTeX.

6.2. Cài đặt thư viện Python

pip install -r requirements.txt

6.3. Cấu hình API key

Tạo file .env tại thư mục gốc:

GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere...

6.4. Các lệnh thực thi

Chạy với chủ đề tự chọn:

python main.py --topic "Kolmogorov-Arnold Networks KAN for Scientific Machine Learning"

Chạy với chủ đề tự động gợi ý từ tập tri thức:

python main.py

Chạy ở chế độ ngoại tuyến hoàn toàn (chỉ dùng dữ liệu trong data/, không gọi API ngoài):

python main.py --topic "Graph Convolutional Networks" --offline

7. Tham số dòng lệnh (CLI Options)

Tham số

Giá trị mặc định

Giải thích chi tiết

--topic

None (Tự chọn từ data)

Chủ đề nghiên cứu khoa học muốn AI thực hiện

--api-key

Đọc từ .env

Google Gemini API Key (hoặc truyền qua biến môi trường GEMINI_API_KEY)

--model

gemini-2.0-flash

Tên mô hình Gemini sử dụng (gemini-2.0-flash, gemini-1.5-flash,...)

--template

None (Tự động chọn)

Mẫu thực nghiệm cơ sở (gnn_graph_learning, recommender_system,...)

--iterations

2

Số vòng lặp tranh biện và phản biện học thuật cho vấn đề & phương pháp

--output-dir

./output

Đường dẫn thư mục xuất thành phẩm

--data-dir

./data

Đường dẫn thư mục chứa dữ liệu tri thức học thuật

--offline

False

Tùy chọn chạy offline, không tìm kiếm bài báo trực tuyến từ OpenAlex/arXiv

--semantic-scholar-key

None

Semantic Scholar API Key (tùy chọn, nâng cao hạn mức tìm kiếm)
