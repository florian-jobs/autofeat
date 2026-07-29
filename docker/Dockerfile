FROM --platform=amd64 python:3.9-slim-bookworm

# Install java for valentine
RUN apt update && apt install -y default-jre htop build-essential

ENV NEO4J_HOST="neo4j://feature-discovery-neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="pass"
ENV NEO4J_DATABASE="benchmark"

COPY requirements.txt /requirements.txt
COPY vendor /vendor

# snakecase was removed from PyPI; valentine still depends on it, so install
# the vendored replacement (vendor/snakecase) before anything that needs it.
RUN pip install /vendor/snakecase

RUN pip install -r /requirements.txt

COPY src /src
COPY pyproject.toml /pyproject.toml

# Install python deps
RUN pip install -e .
