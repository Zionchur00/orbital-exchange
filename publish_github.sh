#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# publish_github.sh -- create the remote repository on GitHub and push.
#
# Requires the CodeBuddy GitHub connector to be authorized
# (CodeBuddy Settings -> Connectors -> GitHub).
#
# Usage:
#   ./publish_github.sh                 # private repo named "orbital-exchange"
#   PUBLIC=1 ./publish_github.sh        # public repo
#   REPO_NAME=my-repo ./publish_github.sh
#   FIX_AUTHOR=0 ./publish_github.sh    # keep the existing commit author
#
# The token is only ever referenced through $GITHUB_TOKEN and is never
# printed to the terminal.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_NAME="${REPO_NAME:-orbital-exchange}"
PRIVATE_JSON="true"; [ "${PUBLIC:-0}" = "1" ] && PRIVATE_JSON="false"
FIX_AUTHOR="${FIX_AUTHOR:-1}"
SKILL_DIR="/root/.codebuddy/skills/github-connector"

# --- 1. obtain token -------------------------------------------------------
source "$SKILL_DIR/scripts/get_token.sh" github >/dev/null 2>&1 || true
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "ERROR: no GitHub token available."
  echo "Please re-authorize GitHub in CodeBuddy: Settings -> Connectors -> GitHub."
  exit 1
fi

api() { curl -s -H "Authorization: Bearer ${GITHUB_TOKEN}" "$@"; }

# --- 2. identify the account ----------------------------------------------
USER=$(api https://api.github.com/user \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('login',''))")
if [ -z "$USER" ]; then
  echo "ERROR: could not read the authenticated GitHub user (token invalid/expired)."
  exit 1
fi
echo "GitHub account : $USER"
echo "Repository     : $USER/$REPO_NAME  (private=$PRIVATE_JSON)"

# --- 3. fix commit author (placeholder -> real account) --------------------
if [ "$FIX_AUTHOR" = "1" ]; then
  NAME=$(api https://api.github.com/user \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('name') or d.get('login'))")
  EMAIL=$(api https://api.github.com/user/emails \
    | python3 -c "import sys,json;d=json.load(sys.stdin);\
print(next((e['email'] for e in d if e.get('primary')), '') if isinstance(d,list) else '')")
  [ -z "$EMAIL" ] && EMAIL="${USER}@users.noreply.github.com"
  echo "Commit author  : $NAME <$EMAIL>"
  git config user.name "$NAME"
  git config user.email "$EMAIL"
  # rewrite author of the existing (unpushed) commits
  git commit --amend --reset-author --no-edit -q || true
fi

# --- 4. create the remote repository if it does not exist ------------------
STATUS=$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  "https://api.github.com/repos/${USER}/${REPO_NAME}")
if [ "$STATUS" = "200" ]; then
  echo "Remote repo already exists, reusing it."
else
  api -X POST -H "Content-Type: application/json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"${REPO_NAME}\",\"description\":\"Validated three-body orbital exchange dynamics\",\"private\":${PRIVATE_JSON}}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('Created:',d.get('full_name',d.get('message')))"
fi

# --- 5. push ---------------------------------------------------------------
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://oauth2:${GITHUB_TOKEN}@github.com/${USER}/${REPO_NAME}.git"
git push -u origin main

echo
echo "Done -> https://github.com/${USER}/${REPO_NAME}"
