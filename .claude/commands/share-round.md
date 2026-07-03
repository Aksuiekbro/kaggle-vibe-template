Manage sharing rounds between agents.

Usage:
- /share-round start — begin a sharing round
- /share-round status — check current sharing round state  
- /share-round end — close the sharing round

Commands:
- `python tools/share.py start` — copies best submissions + code to shared/best/
- `python tools/share.py status` — shows round state and derivatives
- `python tools/share.py end` — locks shared/best/, logs derivatives

During a sharing round, you can:
- Read other agents' code and submissions in shared/best/<agent>/
- Build on their work in YOUR workspace
- Submit improvements with --based-on to track provenance
