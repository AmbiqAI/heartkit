ARG TF_VERSION="2.9.1-gpu"
FROM tensorflow/tensorflow:${TF_VERSION} as builder

WORKDIR /app

RUN \
    apt update && \
    apt install -y --no-install-recommends curl && \
    pip3 install --no-cache-dir -U pip setuptools wheel poetry && \
    echo "Finished installing base packages"

COPY pyproject.toml poetry.toml poetry.lock /app/
RUN \
    mkdir -p /app/ecgarr && \
    poetry install && \
    echo "Finished installing project packages"

COPY ecgarr /app/
RUN \
    echo "Finished installing project sources"

ENTRYPOINT ["poetry", "run", "python", "-m", "ecgarr.pretraining.trainer"]
