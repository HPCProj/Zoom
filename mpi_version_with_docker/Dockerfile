FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server openssh-client \
    openmpi-bin libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir numpy mpi4py pandas matplotlib numba imageio

RUN useradd -m -s /bin/bash mpiuser \
 && mkdir -p /home/mpiuser/code \
 && chown -R mpiuser:mpiuser /home/mpiuser

RUN mkdir -p /var/run/sshd

RUN sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

RUN mkdir -p /etc/ssh/ssh_config.d && \
    printf "Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile=/dev/null\n" \
    > /etc/ssh/ssh_config.d/99-no-hostkey-check.conf

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /home/mpiuser/code
EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
