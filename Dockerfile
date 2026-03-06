FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libxrender1 libxext6 wget && \
    rm -rf /var/lib/apt/lists/*

# Install AutoDock Vina 1.2.7 binary
RUN wget -q -O /usr/local/bin/vina \
    https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 && \
    chmod +x /usr/local/bin/vina

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "-m", "pureprot"]
