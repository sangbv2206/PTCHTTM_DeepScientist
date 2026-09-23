# Su dung base image Python gon nhe co san Linux
FROM python:3.10-slim

# Thiet lap bien moi truong khong tao file pyc va khong buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cai dat pdflatex va cac goi tex phuc vu bien dich bai bao
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

# Thiet lap thu muc lam viec
WORKDIR /app

# Sao chep va cai dat requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chep toan bo ma nguon vao container
COPY . .

# Diem chay mac dinh
ENTRYPOINT ["python", "main.py"]
