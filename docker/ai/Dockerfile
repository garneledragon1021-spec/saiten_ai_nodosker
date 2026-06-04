FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# 必要なシステムライブラリをインストール（zbar 等）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libzbar0 \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係を先にコピーしてキャッシュを有効化
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# ソースコードをコピー
COPY . /app

# デフォルトの実行コマンド（引数はコンテナ起動時に上書き可能）
ENTRYPOINT ["python3", "all.py"]
CMD ["test"]
