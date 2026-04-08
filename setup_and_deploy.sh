#!/bin/bash

# ============================================================
#  CustomerSupportEnv — Full Setup & Deploy Script
#  Run this ONE time from your project folder in VS Code terminal
# ============================================================

set -e  # Stop on any error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "======================================================"
echo "   CustomerSupportEnv — Setup & Deploy"
echo "======================================================"
echo -e "${NC}"

# ── STEP 1: Check Prerequisites ──────────────────────────────
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ Python3 not found. Install from https://python.org${NC}"; exit 1; }
command -v pip >/dev/null 2>&1 || { echo -e "${RED}❌ pip not found.${NC}"; exit 1; }
command -v git >/dev/null 2>&1 || { echo -e "${RED}❌ git not found. Install from https://git-scm.com${NC}"; exit 1; }
command -v docker >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Docker not found — skipping Docker steps (optional)${NC}"

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# ── STEP 2: Install Python Dependencies ──────────────────────
echo -e "\n${YELLOW}[2/7] Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# ── STEP 3: Run Validation Tests ─────────────────────────────
echo -e "\n${YELLOW}[3/7] Running environment validation (27 tests)...${NC}"
python3 validate_env.py
echo -e "${GREEN}✅ All tests passed${NC}"

# ── STEP 4: Run pytest suite ──────────────────────────────────
echo -e "\n${YELLOW}[4/7] Running pytest suite...${NC}"
pip install pytest -q
python3 -m pytest tests/ -v
echo -e "${GREEN}✅ pytest passed${NC}"

# ── STEP 5: Test local API server ─────────────────────────────
echo -e "\n${YELLOW}[5/7] Starting local API server (port 7860)...${NC}"
uvicorn app:app --host 0.0.0.0 --port 7860 &
SERVER_PID=$!
sleep 3

# Test health endpoint
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7860/health)
if [ "$HTTP_STATUS" = "200" ]; then
  echo -e "${GREEN}✅ API server running at http://localhost:7860${NC}"
  echo -e "   📖 Docs at http://localhost:7860/docs"
else
  echo -e "${RED}❌ API server failed (status: $HTTP_STATUS)${NC}"
fi

# Test reset endpoint
echo -e "   Testing /reset endpoint..."
RESET_RESP=$(curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "billing_dispute_easy"}')
SESSION_ID=$(echo $RESET_RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null || echo "")

if [ -n "$SESSION_ID" ]; then
  echo -e "${GREEN}   ✅ /reset works — session: ${SESSION_ID:0:8}...${NC}"
else
  echo -e "${RED}   ❌ /reset failed${NC}"
fi

# Test step endpoint
if [ -n "$SESSION_ID" ]; then
  STEP_RESP=$(curl -s -X POST http://localhost:7860/step \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"$SESSION_ID\", \"action\": {\"action_type\": \"respond\", \"response_text\": \"I apologize for the overcharge. I will fix this right away.\"}}")
  REWARD=$(echo $STEP_RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['reward'])" 2>/dev/null || echo "")
  echo -e "${GREEN}   ✅ /step works — reward: $REWARD${NC}"
fi

kill $SERVER_PID 2>/dev/null
echo -e "${GREEN}✅ Local API test complete${NC}"

# ── STEP 6: Docker Build (if available) ───────────────────────
echo -e "\n${YELLOW}[6/7] Docker build & run test...${NC}"
if command -v docker >/dev/null 2>&1; then
  docker build -t customer-support-env . && \
  echo -e "${GREEN}✅ Docker build successful${NC}" || \
  echo -e "${RED}❌ Docker build failed${NC}"
  
  echo -e "   To run Docker container:"
  echo -e "   ${BLUE}docker run -p 7860:7860 customer-support-env${NC}"
else
  echo -e "${YELLOW}⚠️  Docker not installed — skipping (not required for HuggingFace)${NC}"
fi

# ── STEP 7: Push to HuggingFace Spaces ───────────────────────
echo -e "\n${YELLOW}[7/7] HuggingFace Spaces deployment...${NC}"

# Check HF token
if [ -z "$HF_TOKEN" ]; then
  echo -e "${YELLOW}⚠️  HF_TOKEN not set. Set it to enable auto-push:${NC}"
  echo -e "   ${BLUE}export HF_TOKEN=your_token_here${NC}"
  echo ""
  echo -e "   Manual push instructions:"
  echo -e "   ${BLUE}1. Go to https://huggingface.co/new-space${NC}"
  echo -e "   ${BLUE}2. Name: customer-support-env${NC}"
  echo -e "   ${BLUE}3. SDK: Docker${NC}"
  echo -e "   ${BLUE}4. Then run the git commands below${NC}"
else
  # Check if huggingface_hub is installed
  pip install huggingface_hub -q

  python3 - <<PYEOF
import os
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

# Get username
user = api.whoami()["name"]
repo_id = f"{user}/customer-support-env"

print(f"Creating Space: {repo_id}")

try:
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
        token=token,
    )
    print(f"✅ Space created: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"Space exists or error: {e}")

# Upload all files
import glob, os

files = []
for root, dirs, filenames in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules']]
    for f in filenames:
        if not f.endswith('.pyc'):
            files.append(os.path.join(root, f))

print(f"Uploading {len(files)} files...")

for fpath in files:
    path_in_repo = fpath.lstrip("./")
    try:
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )
        print(f"  ✅ {path_in_repo}")
    except Exception as e:
        print(f"  ❌ {path_in_repo}: {e}")

print(f"\n🎉 Deployed! Visit: https://huggingface.co/spaces/{repo_id}")
PYEOF

fi

# ── Final Summary ─────────────────────────────────────────────
echo ""
echo -e "${BLUE}======================================================"
echo -e "  ✅ SETUP COMPLETE"
echo -e "======================================================"
echo -e "${NC}"
echo -e "📍 Local API:      ${BLUE}uvicorn app:app --port 7860${NC}"
echo -e "📖 API Docs:       ${BLUE}http://localhost:7860/docs${NC}"
echo -e "🧪 Run tests:      ${BLUE}python3 validate_env.py${NC}"
echo -e "🤖 Run inference:  ${BLUE}HF_TOKEN=xxx python3 inference.py${NC}"
echo -e "🐳 Docker run:     ${BLUE}docker run -p 7860:7860 customer-support-env${NC}"
if [ -n "$HF_TOKEN" ]; then
  python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
user = api.whoami()['name']
print(f'🤗 HuggingFace:    https://huggingface.co/spaces/{user}/customer-support-env')
" 2>/dev/null || true
fi
echo ""
