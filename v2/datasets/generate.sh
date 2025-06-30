uv run tools.py --domain $domain --num-tools 20 --overwrite
uv run personas.py --domain $domain --num-personas 100 --overwrite
uv run scenarios.py --domain $domain --categories adaptive_tool_use --scenarios-per-persona 1 --overwrite