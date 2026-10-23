#!/bin/bash
# ==============================================================================
# DeliverAI - AWS EC2 Automated Deployment Script (CO1 & CO4)
# ==============================================================================
# This script provisions an AWS EC2 instance (Ubuntu 22.04 LTS / 24.04 LTS),
# installs Docker & Docker Compose, sets up the DeliverAI service with automated
# container restart policies, and exposes the ETA prediction API & Web UI.
# ==============================================================================

set -e

echo "=== [1/5] Updating system packages ==="
sudo apt-get update -y
sudo apt-get install -y curl git apt-transport-https ca-certificates gnupg lsb-release

echo "=== [2/5] Installing Docker Engine & Compose ==="
if ! command -v docker &> /dev/null; then
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
fi

echo "=== [3/5] Setting up DeliverAI Application ==="
APP_DIR="/opt/deliverai"
sudo mkdir -p "$APP_DIR"
sudo chown -R "$USER:$USER" "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ]; then
    echo "Cloning repository..."
    git clone https://github.com/riddhimk/DeliverAI.git "$APP_DIR"
else
    echo "Pulling latest changes..."
    cd "$APP_DIR"
    git pull origin main
fi

cd "$APP_DIR"

echo "=== [4/5] Launching Containers with Restart Policy ==="
# docker compose up with --build and detached mode
sudo docker compose down --remove-orphans || true
sudo docker compose up -d --build

echo "=== [5/5] Verifying Deployment Health ==="
sleep 8
sudo docker compose ps
curl -s http://127.0.0.1:8000/health || echo "API warming up..."

echo "=============================================================================="
echo " DeliverAI successfully deployed on AWS EC2!"
echo " Web UI & REST API:    http://<ec2-public-ip>:8000"
echo " Streamlit Dashboard:  http://<ec2-public-ip>:8501"
echo " Interactive API Docs: http://<ec2-public-ip>:8000/docs"
echo "=============================================================================="

