"""PostToolUse guard: run `make check` only when a broker .py file was edited.

Wired in .claude/settings.json. Keeps the loop cheap — a no-op for unrelated
edits, and only the realtime check (which spends a couple OpenAI turns) runs
when broker Python actually changes. Exit 2 feeds the failure back to Claude
so the loop cannot close on red.
"""
import json
import os
import subprocess
import sys

# Repo root, derived from this file's location (<repo>/broker/tools/hook_check.py)
# so the hook is portable — no hardcoded personal path.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data = json.load(sys.stdin)
fp = (data.get("tool_input") or {}).get("file_path", "")
if "/broker/" in fp and fp.endswith(".py"):
    print(f"hook: broker edit ({fp}) -> make check", file=sys.stderr)
    r = subprocess.run(["make", "-C", REPO, "check"])
    sys.exit(2 if r.returncode else 0)
