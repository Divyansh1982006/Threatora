"""Demonstration harness displaying the full Metasploit-style CLI session."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interactive_cli import ThreatoraConsole

def run():
    app = ThreatoraConsole()
    app.preloop()
    print()

    session_commands = [
        "status",
        "scan --live",
        "forecast --steps 5",
        "explain",
        "simulate -a BLOCK_MANAGEMENT_PORTS",
        "playbooks",
        "assets"
    ]

    for cmd in session_commands:
        print(f"\n{app.prompt}{cmd}")
        app.onecmd(cmd)

if __name__ == "__main__":
    run()
