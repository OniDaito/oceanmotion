FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends locales tzdata postgresql python3-dev libpq-dev && \
    echo "en_GB.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

COPY requirements.txt ./

COPY sealhits /tmp/sealhits

RUN pip3 install --user --upgrade --disable-pip-version-check pip

RUN pip3 install --user --no-cache-dir --disable-pip-version-check --root-user-action=ignore -r requirements.txt