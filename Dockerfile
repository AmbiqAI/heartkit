ARG TF_VERSION="2.9.1"
FROM tensorflow/tensorflow:${TF_VERSION} as builder

WORKDIR /app

RUN \
    apt update && apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt update && \
    apt install -y --no-install-recommends \
    curl && \
#    python3.9 python3.9-dev python3.9-distutils  && \
#    curl https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    pip3 install --no-cache-dir -U pip setuptools wheel poetry && \
    echo "Finished installing base packages"

COPY pyproject.toml poetry.toml poetry.lock /app/
RUN \
    touch /app/ecgarr && \
    poetry install && \
    echo "Finished installing project packages"

COPY ecgarr /app/
RUN \
    echo "Finished installing project sources"

ENTRYPOINT ["poetry", "run", "python", "-m", "ecgarr.pretraining.trainer"]
