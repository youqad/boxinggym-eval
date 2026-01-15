#!/bin/bash
# Deploy Streamlit dashboard to Hugging Face Spaces (Docker SDK)
set -euo pipefail

HF_SPACE_NAME="${1:-boxing-gym-dashboard}"
HF_USERNAME="${HF_USERNAME:-youkad}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="/tmp/boxing-gym-hf-space"

echo "=== Preparing HF Space directory ==="
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy Streamlit app files
cp "$PROJECT_DIR/scripts/streamlit_app/app.py" "$DEPLOY_DIR/"
cp -r "$PROJECT_DIR/scripts/streamlit_app/pages" "$DEPLOY_DIR/"
cp -r "$PROJECT_DIR/scripts/streamlit_app/components" "$DEPLOY_DIR/"
cp -r "$PROJECT_DIR/scripts/streamlit_app/utils" "$DEPLOY_DIR/"

# Fix import path for HF Space (different directory structure)
sed -i.bak 's/parent.parent.parent.parent/parent.parent/' "$DEPLOY_DIR/utils/data_loader.py"
rm -f "$DEPLOY_DIR/utils/data_loader.py.bak"

# Copy required source files
mkdir -p "$DEPLOY_DIR/src/boxing_gym/agents"
cp "$PROJECT_DIR/src/boxing_gym/agents/results_io.py" "$DEPLOY_DIR/src/boxing_gym/agents/"
touch "$DEPLOY_DIR/src/__init__.py"
touch "$DEPLOY_DIR/src/boxing_gym/__init__.py"
touch "$DEPLOY_DIR/src/boxing_gym/agents/__init__.py"

# Copy .streamlit config
mkdir -p "$DEPLOY_DIR/.streamlit"
cp "$PROJECT_DIR/.streamlit/config.toml" "$DEPLOY_DIR/.streamlit/"

# Create requirements.txt
cat > "$DEPLOY_DIR/requirements.txt" << 'EOF'
streamlit>=1.28.0
plotly>=5.18.0
wandb>=0.15.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
EOF

# Create Dockerfile for Streamlit
cat > "$DEPLOY_DIR/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose Streamlit port
EXPOSE 7860

# HF Spaces expects port 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
EOF

# Create HF Space README
cat > "$DEPLOY_DIR/README.md" << 'EOF'
---
title: BoxingGym Observatory
emoji: 🥊
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# BoxingGym Observatory

Multi-model LLM evaluation dashboard for Stanford's BoxingGym benchmark.

View sweep results, compare against paper baselines, and analyze model performance.
EOF

echo "=== Cloning HF Space repo ==="
cd /tmp
rm -rf hf-space-clone
git clone "https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}" hf-space-clone 2>/dev/null || {
    mkdir hf-space-clone
    cd hf-space-clone
    git init
    git remote add origin "https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"
}

echo "=== Copying files to HF repo ==="
cd /tmp/hf-space-clone
# Clear old files but keep .git
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -r "$DEPLOY_DIR"/* .

echo "=== Pushing to HF Space ==="
git add -A
git commit -m "Deploy dashboard $(date +%Y-%m-%d)" || echo "No changes to commit"
git push -u origin main --force

echo ""
echo "=== Done! ==="
echo "Dashboard: https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"
echo ""
echo "Don't forget to add secrets in Space Settings → Repository secrets:"
echo "  - WANDB_API_KEY"
echo "  - WANDB_ENTITY=your-entity"
echo "  - WANDB_PROJECT=boxing-gym"
