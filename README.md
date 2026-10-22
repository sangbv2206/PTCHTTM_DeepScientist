# DeepScientist

Hệ thống tự động hóa thực nghiệm và viết bài báo khoa học dựa trên kiến trúc Multi-Agent kết hợp mô hình ngôn ngữ lớn (LLM).

Dự án nhận đầu vào là một đề tài nghiên cứu dưới dạng văn bản tự nhiên, sau đó tự động:
1. Đề xuất giả thuyết khoa học và phương pháp cải tiến.
2. Viết mã nguồn mô hình bằng PyTorch và chạy huấn luyện đối sánh với baseline.
3. Trích xuất chỉ số thực nghiệm, đối soát kết quả bằng ClaimLedger để tránh suy diễn sai lệch.
4. Tự động vẽ biểu đồ trực quan hóa kết quả.
5. Soạn thảo toàn văn bài báo theo định dạng LaTeX chuẩn hội nghị quốc tế và biên dịch ra file PDF.

Dự án cung cấp cả giao diện Web (FastAPI + Server-Sent Events) để theo dõi tiến trình thời gian thực và giao diện dòng lệnh (CLI).

---

## Mục lục

- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Quy trình xử lý 8 giai đoạn](#quy-trình-xử-lý-8-giai-đoạn)
- [Các lĩnh vực nghiên cứu hỗ trợ](#các-lĩnh-vực-nghiên-cứu-hỗ-trợ)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)

---

## Kiến trúc hệ thống

Hệ thống gồm 3 tầng chính:

1. **Giao diện người dùng (Frontend):**
   - Viết bằng HTML, CSS và JavaScript thuần, không phụ thuộc framework nặng.
   - Kết nối với backend qua REST API và Server-Sent Events (SSE) để nhận log huấn luyện và trạng thái từng bước theo thời gian thực.
   - Tích hợp trình xem trước ảnh biểu đồ và xem trực tiếp bài báo PDF.

2. **Máy chủ dịch vụ (Backend API):**
   - Xây dựng bằng FastAPI và Uvicorn.
   - Quản lý các phiên chạy (runs) dưới dạng background task, quản lý luồng dữ liệu SSE và phục vụ tệp tĩnh/kết quả thực nghiệm.

3. **Lõi xử lý đa tác tử (Multi-Agent Pipeline):**
   - `ScientistAgent`: Phân tích bài toán, xác định hạn chế của baseline và đề xuất giả thuyết cải tiến.
   - `CoderAgent`: Trực tiếp chỉnh sửa mã nguồn mô hình trong `experiment.py`, bổ sung kiến trúc hoặc thuật toán mới.
   - `ClaimLedger`: Module đọc log thực nghiệm, tính toán độ lệch định lượng giữa baseline và mô hình mới. Cơ chế này buộc Agent viết bài phải trích dẫn đúng số liệu thực tế, không đưa ra kết luận phóng đại khi kết quả không cải thiện đáng kể.
   - `UniversalPlotter`: Tự động đọc dữ liệu huấn luyện và vẽ các biểu đồ đối sánh (`loss_curve.png`, `metric_comparison.png`).
   - `LaTeXWriter`: Soạn thảo toàn văn mã nguồn LaTeX dựa trên cấu trúc bài báo hội nghị (NeurIPS / ICLR), tự động đồng bộ citation từ kho trích dẫn thực tế và gọi `pdflatex` biên dịch PDF.
   - `PeerReviewer`: Đóng vai trò phản biện độc lập, đánh giá bài báo theo thang điểm và tiêu chí học thuật.

---

## Quy trình xử lý 8 giai đoạn

Quá trình nghiên cứu chạy tuần tự qua các giai đoạn:

```text
[Đề tài đầu vào]
       │
       ▼
Giai đoạn 0: Khởi tạo & Định tuyến (Routing)
  - Phân tích ngữ nghĩa đề tài bằng Semantic Router.
  - Chọn template PyTorch phù hợp nhất trong kho template.
       │
       ▼
Giai đoạn 1: Khảo cứu & Hình thành giả thuyết (Hypothesis)
  - Đánh giá bài toán và đề xuất phương án kiến trúc/thuật toán mới.
       │
       ▼
Giai đoạn 2: Lập trình thực nghiệm (Coding)
  - CoderAgent can thiệp mã nguồn experiment.py để thêm mô hình đề xuất.
       │
       ▼
Giai đoạn 3: Huấn luyện mô hình & Thu thập số liệu (Training)
  - Chạy mô hình Baseline và mô hình Cải tiến.
  - Tự động tương thích với GPU (CUDA) nếu có, hoặc fallback an toàn sang CPU.
       │
       ▼
Giai đoạn 4: Đối soát tuyên bố khoa học (Claim Ledger)
  - Phân tích bảng số liệu thực tế (MSE, Loss, Accuracy, thời gian train).
  - Khóa số liệu để ngăn chặn hiện tượng hallucination trong bài viết.
       │
       ▼
Giai đoạn 5: Trực quan hóa dữ liệu (Plotting)
  - Xuất các file biểu đồ phục vụ bài báo.
       │
       ▼
Giai đoạn 6: Soạn thảo bài báo LaTeX (Paper Drafting)
  - Viết toàn văn bài báo bằng tiếng Anh học thuật chuẩn LaTeX 2 cột.
       │
       ▼
Giai đoạn 7: Phản biện học thuật & Xuất bản PDF (Review & Compilation)
  - Chấm điểm bài báo và chạy pdflatex tạo file PDF cuối cùng.
```

---

## Các lĩnh vực nghiên cứu hỗ trợ

Hệ thống sử dụng các bộ template benchmark định sẵn trong thư mục `templates/`:

1. **Dự báo chuỗi thời gian (`time_series_forecasting`)**:
   - Kiến trúc: GRU, LSTM, Temporal CNN, Multi-Step Linear.
   - Bài toán mẫu: Dự báo phụ tải điện, mực nước lũ, tín hiệu cảm biến môi trường.
2. **Xử lý ngôn ngữ tự nhiên (`nanoGPT`, `nanoGPT_lite`)**:
   - Kiến trúc: Transformer Decoder-only (dạng GPT).
   - Bài toán mẫu: Tối ưu cơ chế Attention, so sánh LayerNorm vs RMSNorm, nén mô hình.
3. **Y tế & Thử nghiệm lâm sàng giả lập (`causal_trial_emulation`)**:
   - Phương pháp: Target Trial Emulation, suy luận nhân quả (Causal Inference), cân bằng biến nhiễu IPTW, Propensity Score, phân tích sống còn (Survival Analysis).
   - Bài toán mẫu: So sánh hiệu quả thuốc điều trị, đánh giá nguy cơ phơi nhiễm môi trường.
4. **Học tăng cường (`reinforcement_learning`)**:
   - Kiến trúc: Deep Q-Network (DQN, Double DQN, Dueling DQN), Policy Gradient.
   - Bài toán mẫu: Môi trường điều khiển liên tục (CartPole).
5. **Mô hình tạo sinh khuếch tán (`2d_diffusion`)**:
   - Kiến trúc: Denoising Diffusion Probabilistic Models (DDPM 2D).
   - Bài toán mẫu: Thử nghiệm lịch trình nhiễu (Noise Scheduler), tốc độ sinh mẫu.
6. **Lý thuyết học sâu (`grokking`)**:
   - Kiến trúc: Modular Arithmetic Neural Network.
   - Bài toán mẫu: Nghiên cứu hiện tượng tổng quát hóa trễ (delayed generalization) và tác động của Weight Decay.

---

## Cấu trúc thư mục

```text
DeepScientist/
├── agents/                    # Mã nguồn các tác tử AI (Coder, Writer, Reviewer, Researcher)
├── core/                      # Các thành phần nền tảng (ClaimLedger, Plotter, Gemini LLM, Contracts)
├── data/                      # Cơ sở dữ liệu trích dẫn khoa học thực tế (.jsonl)
├── pipeline/                  # Bộ điều phối quy trình (orchestrator.py, domain_registry.py)
├── templates/                 # Kho template mô hình PyTorch (nanoGPT, time_series, causal...)
├── web/                       # Mã nguồn giao diện Web (index.html, styles.css, app.js)
├── api_server.py              # Server FastAPI xử lý API và Server-Sent Events
├── run_web.py                 # Script chạy nhanh ứng dụng Web
├── main.py                    # Script chạy pipeline qua dòng lệnh (CLI)
├── requirements.txt           # Danh sách thư viện Python phụ thuộc
├── .env.example               # Mẫu biến môi trường
└── README.md                  # Tài liệu hướng dẫn kỹ thuật
```

---

## Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10/11, Linux (Ubuntu 20.04+), hoặc macOS.
- **Python:** 3.10 trở lên.
- **Phần cứng:** Tối thiểu 8GB RAM. Có thể chạy hoàn toàn trên CPU thông thường hoặc tận dụng GPU NVIDIA nếu có CUDA.
- **Trình biên dịch LaTeX (Bắt buộc nếu cần xuất file PDF):**
  - Windows: Cài đặt [MiKTeX](https://miktex.org/).
  - Linux: Cài đặt gói TeX Live (`sudo apt install texlive-full`).
  - *Lưu ý:* Nếu máy chưa cài LaTeX, hệ thống vẫn xuất đầy đủ mã nguồn `.tex` và hình ảnh trong thư mục kết quả.

---

## Hướng dẫn cài đặt

1. Clone mã nguồn về máy:
   ```bash
   git clone https://github.com/sangbv2206/PTCHTTM_DeepScientist.git
   cd PTCHTTM_DeepScientist
   ```

2. Tạo và kích hoạt môi trường ảo Python:
   ```bash
   # Trên Windows
   python -m venv venv
   venv\Scripts\activate

   # Trên Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

4. Cấu hình biến môi trường:
   - Sao chép file mẫu `.env.example` thành `.env`:
     ```bash
     cp .env.example .env
     ```
   - Mở file `.env` và điền Google Gemini API Key:
     ```env
     GEMINI_API_KEY=your_actual_gemini_api_key_here
     GEMINI_MODEL=gemini-3.5-flash-lite
     ```

---

## Hướng dẫn sử dụng

### Cách 1: Sử dụng giao diện Web

Khởi động máy chủ:
```bash
python run_web.py
```

Mở trình duyệt truy cập: `http://localhost:8000`

- Nhập tên đề tài nghiên cứu (có thể nhập bằng tiếng Việt hoặc tiếng Anh).
- Chọn số vòng lặp (`Max Iterations`) và bấm **Bắt đầu nghiên cứu**.
- Theo dõi log huấn luyện, biểu đồ và đọc bài báo PDF trực tiếp trên giao diện.

### Cách 2: Sử dụng dòng lệnh (CLI)

Chạy trực tiếp pipeline qua file `main.py`:
```bash
python main.py --topic "Dự báo phụ tải tiêu thụ điện trong lưới điện thông minh" --max_iters 1
```

Kết quả của từng lượt chạy (gồm mã nguồn thí nghiệm, biểu đồ, file LaTeX và file PDF) được lưu tự động trong thư mục `output/run_<mã_phiên>/`.
