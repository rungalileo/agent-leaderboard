domain=finance
python tools.py --domain $domain --num-tools 20 --overwrite
python personas.py --domain $domain --num-personas 100 --overwrite
python scenarios.py --domain $domain --categories adaptive_tool_use --overwrite