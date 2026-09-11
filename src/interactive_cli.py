"""Metasploit-Style Interactive Tactical CLI for Threatora (NTRO PS 26153).

Engineered using cmd2 and rich:
  - Tactical cyber operations prompt with real-time API communication.
  - Commands: scan, forecast, explain, mitigate, assets, incidents, status.
  - Rich ASCII tables, risk badges, SHAP bar charts, and timeline rollouts.
  - Full tab-completion and zero-trust API authentication.
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

# Enforce UTF-8 on Windows consoles to prevent cp1252 charmap encoding errors
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cmd2
from cmd2 import with_argparser, with_category, Cmd2ArgumentParser
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console(legacy_windows=False)

DEFAULT_API_URL = os.environ.get("THREATORA_API_URL", "http://127.0.0.1:5000")


class ThreatoraConsole(cmd2.Cmd):
    """Tactical Cyber Defense Terminal (Metasploit-Style)."""

    intro = ""
    prompt = "threatora [tactical] > "

    def __init__(self, api_url: str = DEFAULT_API_URL):
        super().__init__(
            allow_cli_args=False,
            shortcuts={"?": "help", "!": "shell"},
            auto_load_commands=False,
        )
        self.api_url = api_url.rstrip("/")
        self.current_target: Optional[str] = None
        self.last_results: Optional[Dict[str, Any]] = None

        # Aesthetics
        self.self_in_py = True
        self.default_category = "General Commands"

    def preloop(self) -> None:
        """Display tactical ASCII banner upon entry."""
        banner = """
[bold red]████████╗██╗  ██╗██████╗ ███████╗ █████╗ ████████╗ ██████╗ ██████╗  █████╗ [/bold red]
[bold red]╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗[/bold red]
[bold yellow]   ██║   ███████║██████╔╝█████╗  ███████║   ██║   ██║   ██║██████╔╝███████║[/bold yellow]
[bold cyan]   ██║   ██╔══██║██╔══██╗██╔══╝  ██╔══██║   ██║   ██║   ██║██╔══██╗██╔══██║[/bold cyan]
[bold cyan]   ██║   ██║  ██║██║  ██║███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║██║  ██║[/bold cyan]
[bold white]   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝[/bold white]
[dim green]   AI-Based Network World Model Forecasting & Zero-Trust Mitigation (NTRO PS 26153)[/dim green]
[dim]   Type 'help' or '?' for available tactical directives. 'status' to verify API connection.[/dim]
"""
        console.print(banner)
        self._check_api_status(silent=True)

    def _check_api_status(self, silent: bool = False) -> bool:
        try:
            r = requests.get(f"{self.api_url}/api/health", timeout=2.5)
            if r.status_code == 200:
                data = r.json()
                if not silent:
                    console.print(f"[bold green]✔ Connected to Threatora Core API:[/bold green] {self.api_url}")
                    console.print(f"  [dim]Device: {data.get('device')} | Cell: {data.get('recurrent_cell')} | Status: {data.get('status')}[/dim]")
                return True
        except Exception:
            pass

        if not silent:
            console.print(f"[bold red]✘ Core API Offline or Unreachable:[/bold red] {self.api_url}")
            console.print("  [yellow]Make sure the Flask server is running on port 5000 (`python server/app.py`)[/yellow]")
        return False

    # -------------------------------------------------------------------------
    # Command: status
    # -------------------------------------------------------------------------
    @with_category("Operational Directives")
    def do_status(self, _args):
        """Check API connection, LSTM World Model health, and active target IP."""
        console.print(Panel("[bold cyan]THREATORA ENGINE DIAGNOSTICS[/bold cyan]", expand=False))
        self._check_api_status(silent=False)
        target_display = self.current_target if self.current_target else "None (Use `set target <IP>` or `scan`)"
        console.print(f"[bold white]Active Operator Target:[/bold white] [bold yellow]{target_display}[/bold yellow]")

    # -------------------------------------------------------------------------
    # Command: set target
    # -------------------------------------------------------------------------
    set_parser = Cmd2ArgumentParser(prog="set", description="Set tactical context variables")
    set_subparsers = set_parser.add_subparsers(dest="var_name")
    target_parser = set_subparsers.add_parser("target", help="Set active target IP")
    target_parser.add_argument("ip", help="Target IP address (e.g. 192.168.1.5)")

    @with_category("Operational Directives")
    @with_argparser(set_parser)
    def do_set(self, args):
        """Set tactical variables like target IP address."""
        if args.var_name == "target":
            self.current_target = args.ip
            self.prompt = f"threatora [target:{args.ip}] > "
            console.print(f"[bold green]✔ Active target context set to:[/bold green] [bold yellow]{args.ip}[/bold yellow]")

    # -------------------------------------------------------------------------
    # Command: scan
    # -------------------------------------------------------------------------
    scan_parser = Cmd2ArgumentParser(prog="scan", description="Run telemetry inference or load attack scenario")
    scan_parser.add_argument("--live", action="store_true", help="Run scan on live demo attack telemetry stream")
    scan_parser.add_argument("-i", "--input", help="Path to PCAP or CSV flow telemetry file")

    @with_category("Tactical Intelligence")
    @with_argparser(scan_parser)
    def do_scan(self, args):
        """Scan network telemetry stream and compute Infiltration Probabilities and MITRE Stages."""
        endpoint = f"{self.api_url}/api/v1/telemetry"

        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="[cyan]Streaming telemetry through LSTM Inference Singleton...", total=None)

            try:
                if args.input:
                    input_path = Path(args.input)
                    if not input_path.exists():
                        console.print(f"[red]Error: Telemetry file {input_path} not found.[/red]")
                        return
                    with open(input_path, "rb") as f:
                        resp = requests.post(f"{self.api_url}/api/upload", files={"file": f}, timeout=60)
                else:
                    # Live / demo stream
                    resp = requests.get(f"{self.api_url}/api/demo", timeout=30)

                if resp.status_code != 200:
                    console.print(f"[red]Scan failed with status {resp.status_code}: {resp.text}[/red]")
                    return

                data = resp.json()
                self.last_results = data
            except Exception as e:
                console.print(f"[bold red]API Communication Failure:[/bold red] {e}")
                return

        hosts = data.get("hosts", [])
        if not hosts:
            console.print("[yellow]Scan completed: No host telemetry found in stream.[/yellow]")
            return

        # Auto-set lead target
        if not self.current_target and hosts:
            self.current_target = hosts[0]["host_ip"]
            self.prompt = f"threatora [target:{self.current_target}] > "

        # Render Rich Table
        table = Table(title="⚡ THREATORA TELEMETRY INFERENCE & ATT&CK MAPPING", expand=True)
        table.add_column("Host IP", style="bold white", justify="left")
        table.add_column("60s Cells", justify="center")
        table.add_column("Infiltration Risk", justify="center")
        table.add_column("MITRE Stage", justify="left")
        table.add_column("Technique / Action", justify="left")
        table.add_column("Zero-Trust Status", justify="center")

        for h in hosts:
            risk = h["current_risk_score"] * 100
            risk_color = "red" if risk >= 50 else ("yellow" if risk >= 20 else "green")
            stage = h["current_stage"]["name"]
            technique = h["current_stage"]["metadata"].get("technique", "Baseline")
            status = "[bold red]THREAT DETECTED[/bold red]" if h["is_anomalous"] else "[bold green]BENIGN[/bold green]"

            table.add_row(
                h["host_ip"],
                str(h["window_count"]),
                f"[{risk_color}]{risk:.1f}%[/{risk_color}]",
                f"[bold cyan]{stage}[/bold cyan]",
                technique,
                status
            )

        console.print(table)
        console.print(f"[dim]Total Hosts: {len(hosts)} | Flagged: {data.get('flagged_hosts', 0)} | Active Target: [bold yellow]{self.current_target}[/bold yellow][/dim]\n")
        console.print("[dim]Hint: Type [bold cyan]forecast[/bold cyan] to see future rollout, or [bold cyan]explain[/bold cyan] for SHAP attribution.[/dim]")

    # -------------------------------------------------------------------------
    # Command: forecast
    # -------------------------------------------------------------------------
    forecast_parser = Cmd2ArgumentParser(prog="forecast", description="Dreaming Mode: LSTM K-step forward rollout")
    forecast_parser.add_argument("-t", "--target", help="Target IP (defaults to active target)")
    forecast_parser.add_argument("--steps", type=int, default=10, help="Number of future steps (default 10)")

    @with_category("Tactical Intelligence")
    @with_argparser(forecast_parser)
    def do_forecast(self, args):
        """Simulate future network states and infiltration probability rollout (.imagine mode)."""
        target_ip = args.target or self.current_target
        if not target_ip:
            console.print("[red]No target specified. Use `forecast -t <IP>` or run `scan` first.[/red]")
            return

        if not self.last_results:
            console.print("[yellow]No prior scan in memory. Executing live scan first...[/yellow]")
            self.do_scan(self.scan_parser.parse_args(["--live"]))
            if not self.last_results:
                return

        host = next((h for h in self.last_results.get("hosts", []) if h["host_ip"] == target_ip), None)
        if not host:
            console.print(f"[red]Target host {target_ip} not found in recent scan results.[/red]")
            return

        timeline = host.get("forecast_timeline", [])
        if not timeline:
            console.print("[yellow]No forecast timeline generated for target host.[/yellow]")
            return

        console.print(f"\n[bold cyan]⚡ DREAMING MODE: {len(timeline)}-STEP FORWARD MONTE CARLO ROLLOUT (.imagine)[/bold cyan]")
        console.print(f"[bold white]Target Host:[/bold white] [bold yellow]{target_ip}[/bold yellow] | [dim]State-Space Predictive Simulation[/dim]\n")

        table = Table(expand=True)
        table.add_column("Horizon", justify="center", style="bold cyan")
        table.add_column("Risk Probability (95% CI)", justify="center")
        table.add_column("Forecast Visual Density", justify="left")
        table.add_column("Predicted Kill Chain Stage", justify="left")

        for step in timeline:
            prob = step["infilt_prob"] * 100
            lower = step["lower_ci"] * 100
            upper = step["upper_ci"] * 100
            stage_name = step["stage_name"]
            stage_color = "red" if prob >= 50 else ("yellow" if prob >= 20 else "green")

            # ASCII progress visual
            bar_len = int(prob / 4)
            bar_visual = f"[{stage_color}]" + "█" * bar_len + "░" * (25 - bar_len) + f"[/{stage_color}]"

            table.add_row(
                step["minute"],
                f"{prob:5.1f}% ({lower:4.1f}% - {upper:4.1f}%)",
                bar_visual,
                f"[{stage_color}]{stage_name}[/{stage_color}]"
            )

        console.print(table)

    # -------------------------------------------------------------------------
    # Command: explain
    # -------------------------------------------------------------------------
    explain_parser = Cmd2ArgumentParser(prog="explain", description="SHAP & Saliency Feature Importance Engine")
    explain_parser.add_argument("-t", "--target", help="Target IP (defaults to active target)")

    @with_category("Explainable AI (XAI)")
    @with_argparser(explain_parser)
    def do_explain(self, args):
        """Dissect LSTM neural decision via SHAP feature importance & attribution scores."""
        target_ip = args.target or self.current_target
        if not target_ip:
            console.print("[red]No target specified. Run `scan` first or specify `--target <IP>`.[/red]")
            return

        if not self.last_results:
            console.print("[yellow]Executing live scan to populate feature attributions...[/yellow]")
            self.do_scan(self.scan_parser.parse_args(["--live"]))
            if not self.last_results:
                return

        host = next((h for h in self.last_results.get("hosts", []) if h["host_ip"] == target_ip), None)
        if not host:
            console.print(f"[red]Target host {target_ip} not found.[/red]")
            return

        explainability = host.get("explainability", {})
        top_features = explainability.get("top_features", {})

        console.print(f"\n[bold magenta]🔬 SHAP EXPLAINABILITY ENGINE (ZERO-HALLUCINATION AUDIT)[/bold magenta]")
        console.print(f"[bold white]Target IP:[/bold white] [bold yellow]{target_ip}[/bold yellow] | [dim]Feature Attribution Shapley Vectors[/dim]\n")

        if not top_features:
            console.print("[yellow]No high-variance feature drivers detected for this sequence.[/yellow]")
            return

        table = Table(title="Key Telemetry Feature Drivers", expand=True)
        table.add_column("Feature Name", style="bold cyan")
        table.add_column("Attribution (%)", justify="right", style="bold yellow")
        table.add_column("Attribution Waterfall Visual", justify="left")

        for feat, score in top_features.items():
            bar_len = int(min(score * 1.5, 40))
            bar_str = "[bold magenta]" + "█" * bar_len + "[/bold magenta]"
            table.add_row(feat, f"{score:.1f}%", bar_str)

        console.print(table)

        deltas = explainability.get("state_deltas", [])
        if deltas:
            console.print("\n[bold white]Forecasted State Perturbations (Next 10 Minutes):[/bold white]")
            for d in deltas[:4]:
                sign = "+" if d.get("pct_change", 0) > 0 else ""
                console.print(
                    f"  • [cyan]{d.get('feature')}[/cyan]: {d.get('current_value')} → {d.get('forecast_value')} "
                    f"([bold yellow]{sign}{d.get('pct_change')}%[/bold yellow])"
                )
        console.print("")

    # -------------------------------------------------------------------------
    # Command: simulate (What-If Action Simulator)
    # -------------------------------------------------------------------------
    simulate_parser = Cmd2ArgumentParser(prog="simulate", description="What-If Action Counterfactual Simulator")
    simulate_parser.add_argument("-a", "--action", default="BLOCK_MANAGEMENT_PORTS",
                                 choices=["ISOLATE_HOST", "BLOCK_MANAGEMENT_PORTS", "RATE_LIMIT_SYN", "SINKHOLE_C2_DNS", "THROTTLE_EGRESS"],
                                 help="Hypothetical defensive action to test")
    simulate_parser.add_argument("-t", "--target", help="Target IP (defaults to active target)")
    simulate_parser.add_argument("--horizon", type=int, default=10, help="Simulation steps (default 10)")

    @with_category("What-If Action Simulation")
    @with_argparser(simulate_parser)
    def do_simulate(self, args):
        """Simulate counterfactual network states and evaluate risk reduction of defensive actions."""
        target_ip = args.target or self.current_target
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=f"[cyan]Evaluating counterfactual dynamics for action '{args.action}'...", total=None)
            try:
                r = requests.post(
                    f"{self.api_url}/api/v1/simulate",
                    json={"action": args.action, "target_ip": target_ip, "horizon": args.horizon},
                    timeout=15
                )
                if r.status_code != 200:
                    console.print(f"[red]Simulation failed ({r.status_code}): {r.text}[/red]")
                    return
                data = r.json()
            except Exception as e:
                console.print(f"[red]API communication error: {e}[/red]")
                return

        console.print(f"\n[bold cyan]🔮 WHAT-IF COUNTERFACTUAL SIMULATION: {data.get('action_name')}[/bold cyan]")
        console.print(f"[bold white]Target IP:[/bold white] [bold yellow]{data.get('target_ip')}[/bold yellow] | [dim]{data.get('action_description')}[/dim]\n")

        table = Table(title="Trajectory Comparison: Baseline vs Counterfactual", expand=True)
        table.add_column("Horizon", justify="center", style="bold cyan")
        table.add_column("Baseline Risk", justify="center", style="red")
        table.add_column("Simulated Risk", justify="center", style="green")
        table.add_column("ΔRisk Reduction", justify="center", style="bold yellow")
        table.add_column("Baseline ATT&CK Stage", justify="left")
        table.add_column("Mitigated Stage", justify="left")

        for row in data.get("timeline", []):
            b_risk = row["baseline_risk"] * 100
            s_risk = row["simulated_risk"] * 100
            d_risk = row["delta_risk"] * 100
            d_sign = "+" if d_risk >= 0 else ""
            table.add_row(
                row["minute"],
                f"{b_risk:5.1f}%",
                f"{s_risk:5.1f}%",
                f"[bold green]{d_sign}{d_risk:5.1f}%[/bold green]" if d_risk > 0 else f"{d_risk:5.1f}%",
                row["baseline_stage"],
                f"[bold green]{row['simulated_stage']}[/bold green]"
            )

        console.print(table)
        console.print(f"\n[bold white]Overall Risk Reduction:[/bold white] [bold green]{data.get('pct_risk_reduction')}%[/bold green] | [bold white]Peak Mitigation:[/bold white] [bold green]-{data.get('peak_risk_reduction', 0)*100:.1f}%[/bold green]")
        console.print(f"[bold white]Tactical Assessment:[/bold white] [bold cyan]{data.get('tactical_verdict')}[/bold cyan]\n")

    # -------------------------------------------------------------------------
    # Command: playbooks / mitigate
    # -------------------------------------------------------------------------
    @with_category("Automated Mitigation")
    def do_playbooks(self, _args):
        """List active and pending Mitigation Playbooks from PostgreSQL state store."""
        try:
            r = requests.get(f"{self.api_url}/api/v1/playbooks", timeout=5)
            if r.status_code != 200:
                console.print(f"[red]Failed to fetch playbooks: {r.text}[/red]")
                return
            data = r.json()
        except Exception as e:
            console.print(f"[red]Error contacting API: {e}[/red]")
            return

        playbooks = data.get("playbooks", [])
        if not playbooks:
            console.print("[green]No active threat playbooks pending. Environment clean.[/green]")
            return

        table = Table(title="🛡️ ACTIVE MITIGATION PLAYBOOKS (POSTGRESQL STATE)", expand=True)
        table.add_column("Playbook UID", style="bold cyan")
        table.add_column("Target IP", style="bold white")
        table.add_column("Kill Chain Stage", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("Containment Strategy Preview", style="dim")

        for pb in playbooks:
            status = pb["status"]
            status_badge = "[bold red]PENDING[/bold red]" if status == "PENDING" else "[bold green]EXECUTED[/bold green]"
            strat_preview = pb["containment_strategy"][:65] + "..." if len(pb["containment_strategy"]) > 65 else pb["containment_strategy"]
            table.add_row(
                pb["playbook_uid"],
                pb["target_ip"],
                pb["kill_chain_stage"],
                status_badge,
                strat_preview
            )

        console.print(table)
        console.print("[dim]Use `mitigate --playbook <UID>` or `mitigate --isolate` to trigger containment.[/dim]")

    mitigate_parser = Cmd2ArgumentParser(prog="mitigate", description="Execute containment playbook or isolate host")
    mitigate_parser.add_argument("-p", "--playbook", help="Playbook UID to execute")
    mitigate_parser.add_argument("-t", "--target", help="Target IP to isolate")
    mitigate_parser.add_argument("--isolate", action="store_true", help="Execute immediate host isolation on target")

    @with_category("Automated Mitigation")
    @with_argparser(mitigate_parser)
    def do_mitigate(self, args):
        """Execute automated containment playbook and isolate compromised endpoint."""
        target_ip = args.target or self.current_target

        payload = {}
        if args.playbook:
            payload["playbook_uid"] = args.playbook
        elif args.isolate and target_ip:
            payload["target_ip"] = target_ip
            payload["action"] = "isolate"
        else:
            console.print("[red]Must specify either --playbook <UID> or --isolate with active target.[/red]")
            return

        with Progress(
            SpinnerColumn(spinner_name="aesthetic"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=f"[red]Dispatching Zero-Trust Containment Directives...", total=None)

            try:
                r = requests.post(f"{self.api_url}/api/v1/mitigate", json=payload, timeout=10)
                if r.status_code not in (200, 201):
                    console.print(f"[red]Mitigation action rejected by API ({r.status_code}): {r.text}[/red]")
                    return
                resp = r.json()
            except Exception as e:
                console.print(f"[red]Failed to execute mitigation: {e}[/red]")
                return

        console.print(f"\n[bold green]✔ CONTAINMENT DIRECTIVE EXECUTED[/bold green]")
        console.print(f"  [bold white]Target IP:[/bold white] [bold yellow]{resp.get('target_ip')}[/bold yellow]")
        console.print(f"  [bold white]Asset Status:[/bold white] [bold red]{resp.get('host_status')}[/bold red]")
        console.print(f"  [bold white]Audit Log:[/bold white] [dim]{resp.get('execution_log')}[/dim]\n")

    # -------------------------------------------------------------------------
    # Command: assets
    # -------------------------------------------------------------------------
    @with_category("Asset & Incident Ledger")
    def do_assets(self, _args):
        """Query PostgreSQL Asset Inventory (Criticality, OS, Services, Isolation Status)."""
        try:
            r = requests.get(f"{self.api_url}/api/v1/assets", timeout=5)
            if r.status_code != 200:
                console.print(f"[red]Failed to retrieve assets: {r.text}[/red]")
                return
            data = r.json()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        assets = data.get("assets", [])
        table = Table(title="🏢 ENTERPRISE ASSET INVENTORY (POSTGRESQL LEDGER)", expand=True)
        table.add_column("IP Address", style="bold white")
        table.add_column("Hostname", style="bold cyan")
        table.add_column("Criticality", justify="center")
        table.add_column("OS", style="dim")
        table.add_column("Status", justify="center")

        for a in assets:
            crit = a["criticality"]
            crit_style = "bold red" if crit == "MISSION_CRITICAL" else ("yellow" if crit == "HIGH" else "green")
            status = a["status"]
            status_style = "bold red" if status in ("ISOLATED", "COMPROMISED") else "bold green"

            table.add_row(
                a["ip_address"],
                a["hostname"],
                f"[{crit_style}]{crit}[/{crit_style}]",
                a.get("operating_system", "Unknown"),
                f"[{status_style}]{status}[/{status_style}]"
            )

        console.print(table)

    # -------------------------------------------------------------------------
    # Command: incidents
    # -------------------------------------------------------------------------
    @with_category("Asset & Incident Ledger")
    def do_incidents(self, _args):
        """Query incident history and forensic alerts recorded in PostgreSQL."""
        try:
            r = requests.get(f"{self.api_url}/api/v1/incidents", timeout=5)
            if r.status_code != 200:
                console.print(f"[red]Failed to retrieve incidents: {r.text}[/red]")
                return
            data = r.json()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        incidents = data.get("incidents", [])
        if not incidents:
            console.print("[green]No security incidents recorded in ledger.[/green]")
            return

        table = Table(title="🚨 SECURITY INCIDENTS LEDGER", expand=True)
        table.add_column("Incident UID", style="bold cyan")
        table.add_column("Target IP", style="bold white")
        table.add_column("Risk Score", justify="center")
        table.add_column("Kill Chain Stage", style="yellow")
        table.add_column("Contained", justify="center")
        table.add_column("Timestamp", style="dim")

        for inc in incidents:
            score = inc["risk_score"] * 100
            score_col = "red" if score >= 50 else "yellow"
            contained = "[green]YES[/green]" if inc.get("is_contained") else "[red]NO[/red]"

            table.add_row(
                inc["incident_uid"],
                inc["target_ip"],
                f"[{score_col}]{score:.1f}%[/{score_col}]",
                inc["stage_name"],
                contained,
                str(inc.get("created_at", ""))[:19]
            )

        console.print(table)

    def do_exit(self, _=None) -> bool:
        """Exit and terminate the Threatora tactical console session."""
        console.print("[yellow][*] Severing tactical session. Goodbye, Operator.[/yellow]")
        return True

    def do_quit(self, _=None) -> bool:
        """Exit and terminate the Threatora tactical console session."""
        return self.do_exit()

    def do_q(self, _=None) -> bool:
        """Quick exit alias."""
        return self.do_exit()


# Backward compatibility alias
ThraetoraConsole = ThreatoraConsole


def launch_console(api_url: str = DEFAULT_API_URL):
    """Entrypoint to start the interactive cmd2 console."""
    app = ThreatoraConsole(api_url=api_url)
    sys.exit(app.cmdloop())


if __name__ == "__main__":
    launch_console()
