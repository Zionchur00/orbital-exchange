#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# create_release.sh -- attach the two packaged archives to a GitHub Release.
#
# Prerequisite: run ./publish_github.sh first (the repository must exist).
# Requires CodeBuddy GitHub connector authorization.
#
# Usage:
#   ./create_release.sh
#   TAG=v1.1.0 ./create_release.sh
#
# Release assets are the *correct* place for binary bundles (.zip).
# Do NOT commit large archives into the repository itself.
# ---------------------------------------------------------------------------
set -euo pipefail

TAG="${TAG:-v1.0.0}"
REPO_NAME="${REPO_NAME:-orbital-exchange}"
FULL_ZIP="/workspace/orbital-exchange.zip"       # 完整包（含成品）
PKG_ZIP="/workspace/orbital-exchange-src.zip"   # 源码包
SKILL_DIR="/root/.codebuddy/skills/github-connector"

source "$SKILL_DIR/scripts/get_token.sh" github >/dev/null 2>&1 || true
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "ERROR: no GitHub token. Re-authorize GitHub in CodeBuddy Settings -> Connectors."
  exit 1
fi

api() { curl -s -H "Authorization: Bearer ${GITHUB_TOKEN}" "$@"; }

USER=$(api https://api.github.com/user \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('login',''))")
[ -z "$USER" ] && { echo "ERROR: cannot read GitHub user."; exit 1; }
echo "GitHub account : $USER"
echo "Repository     : $USER/$REPO_NAME"
echo "Release tag    : $TAG"

# --- 1. tag and push -------------------------------------------------------
git tag -f "$TAG" >/dev/null 2>&1
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://oauth2:${GITHUB_TOKEN}@github.com/${USER}/${REPO_NAME}.git"
git push -u origin main
git push -f origin "$TAG"
echo "pushed main + tag $TAG"

# --- 2. create the release -------------------------------------------------
EXISTING=$(api "https://api.github.com/repos/${USER}/${REPO_NAME}/releases/tags/${TAG}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('id',''))")
if [ -n "$EXISTING" ]; then
  RELEASE_ID="$EXISTING"
  echo "release $TAG already exists (id=$RELEASE_ID), reusing it."
else
  RELEASE_ID=$(api -X POST -H "Content-Type: application/json" \
    "https://api.github.com/repos/${USER}/${REPO_NAME}/releases" \
    -d "{\"tag_name\":\"${TAG}\",\"name\":\"${TAG}\",\"body\":\"Orbital Exchange - validated three-body orbital-change simulation.\\\\n\\\\n- orbital-exchange.zip : full project (code + MP4/GIF/figures)\\\\n- orbital-exchange-src.zip : source-only package\"}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('id',''))")
  echo "created release $TAG (id=$RELEASE_ID)"
fi

# --- 3. upload the two archives as release assets --------------------------
upload() {
  local file="$1"
  [ -f "$file" ] || { echo "skip (missing): $file"; return 0; }
  echo "uploading $(basename "$file") ($(du -h "$file" | cut -f1))..."
  curl -s -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Content-Type: application/zip" \
    --data-binary @"$file" \
    "https://uploads.github.com/repos/${USER}/${REPO_NAME}/releases/${RELEASE_ID}/assets?name=$(basename "$file")" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('  ->', d.get('name', d.get('message')))"
}

upload "$FULL_ZIP"
upload "$PKG_ZIP"

echo
echo "Release ready -> https://github.com/${USER}/${REPO_NAME}/releases/tag/${TAG}"
