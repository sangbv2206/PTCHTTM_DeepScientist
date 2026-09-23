# Sử dụng base image Python gọn nhẹ có sẵn Linux
FROM python:3.10-slim

# Thiết lập biến môi trường để không tạo file .pyc và không buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài đặt pdflatex và các gói TeX phục vụ biên dịch bài báo
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    cm-super \
    dvipng \
    ghostscript \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép và cài đặt các thư viện từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Sao chep toan bo ma nguon vao container
COPY . .

# Diem chay mac dinh
ENTRYPOINT ["python", "main.py"]
