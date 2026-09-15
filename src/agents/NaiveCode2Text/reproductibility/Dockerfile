FROM inseefrlab/onyxia-vscode-python:py3.13.12

USER root

RUN apt-get update \
 && apt-get install -y curl unzip jq \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://releases.hashicorp.com/vault/1.16.2/vault_1.16.2_linux_amd64.zip -o vault.zip \
 && unzip vault.zip \
 && mv vault /usr/local/bin/ \
 && chmod +x /usr/local/bin/vault \
 && rm vault.zip

USER 1000