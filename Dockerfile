FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps \
    "tribev2 @ git+https://github.com/facebookresearch/tribev2.git@72399081ed3f1040c4d996cefb2864a4c46f5b8e"

COPY config.py .
COPY handler.py .
COPY storage/ ./storage/

CMD ["python", "-u", "handler.py"]
