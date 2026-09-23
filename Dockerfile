# syntax=docker/dockerfile:1
# Hardened image for lossless-claw plugin + OpenClaw gateway.
# Fixes RIC-729 / RIC-709 M-3.1 + M-9.1:
#   - base image pinned to a sha256 digest (no tag drift)
#   - non-root user (lcm); container never runs as root
#   - openclaw version locked (no @latest supply-chain drift)
#   - gateway bound to loopback only (--bind 127.0.0.1), not auto

# node:22-bookworm-slim pinned by digest (arm64/amd64 manifest resolved by your registry).
# To refresh the digest after an upstream rebuild:
#   docker pull node:22-bookworm-slim && docker inspect --format '{{index .RepoDigests 0}}' node:22-bookworm-slim
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5

# Install git and build tools. Bookworm has newer cmake (3.25+).
# Clean apt lists in the same layer to keep the image lean and avoid stale index drift.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git python3 make g++ cmake linux-libc-dev \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user/group with a fixed UID/GID and a home directory.
RUN groupadd --system --gid 1001 lcm \
 && useradd  --system --uid 1001 --gid lcm --home-dir /home/lcm --create-home --shell /usr/sbin/nologin lcm

# Install openclaw globally at a locked version (no @latest). Locks the supply chain
# to a known-good release; bump deliberately and re-audit.
ARG OPENCLAW_VERSION=2026.7.2-beta.2
RUN npm install -g openclaw@${OPENCLAW_VERSION} \
 && npm cache clean --force

# Copy our patched plugin and build it.
WORKDIR /plugin
COPY --chown=lcm:lcm package.json package-lock.json ./
RUN npm ci
COPY --chown=lcm:lcm . .
RUN npm run build

# Set up the openclaw workspace under the non-root user's home.
# Create the dir tree as root, then chown to lcm so the gateway can write
# extensions/state without hitting EACCES after USER switches.
WORKDIR /home/lcm/.openclaw
RUN mkdir -p /home/lcm/.openclaw/extensions \
 && chown -R lcm:lcm /home/lcm/.openclaw

# Install our local plugin into the docker openclaw instance, as the non-root user
# so file ownership is correct from the start.
USER lcm
# Local plugin source is trusted at build time (this repo). --force bypasses the
# ClawHub review/trust prompt that blocks unattended installs of out-of-registry paths.
RUN openclaw plugins install /plugin --force

# Run openclaw gateway in the foreground.
# --dev creates a dev config; --bind loopback restricts the gateway to 127.0.0.1
# only (the "auto" preset in the original Dockerfile could bind a non-loopback
# interface). --allow-unconfigured lets the gateway start without an onboarding
# `gateway.mode=local` in config (the original --dev hit the same block); pair
# with a reverse proxy / sidecar for any external exposure, not a wider bind.
ENTRYPOINT ["openclaw", "gateway", "run", "--dev", "--bind", "loopback", "--allow-unconfigured"]
