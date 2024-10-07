FROM python:3.9

# Apt install curl to perform health checks
# Install additional APT packages
RUN apt-get update && \
    apt-get install -y \
        curl && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/*

# Pip install required packages for evidently
RUN pip install evidently

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
