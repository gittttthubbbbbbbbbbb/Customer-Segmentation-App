FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    xdg-utils \
    chromium && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


ENV CHROME_PATH=/usr/bin/chromium


WORKDIR /app


COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app


EXPOSE 8501


CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
