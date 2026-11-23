#!/bin/sh
set -e

if [ -d /ssh ]; then
  mkdir -p /home/mpiuser/.ssh

  if [ -f /ssh/id_rsa ]; then
    cp /ssh/id_rsa /home/mpiuser/.ssh/id_rsa
    chmod 600 /home/mpiuser/.ssh/id_rsa
  fi

  if [ -f /ssh/authorized_keys ]; then
    cp /ssh/authorized_keys /home/mpiuser/.ssh/authorized_keys
    chmod 600 /home/mpiuser/.ssh/authorized_keys
  fi

  chmod 700 /home/mpiuser/.ssh
  chown -R mpiuser:mpiuser /home/mpiuser/.ssh
fi

exec "$@"