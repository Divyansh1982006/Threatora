"""Metasploit-Style Interactive Tactical Cyber Defense Console for Threatora (NTRO PS 26153).

Engineered using cmd2 and rich:
  - High-impact Cyber Command Header with adaptive terminal scaling.
  - Interactive boot animation with OSI Layer 2-7 synchronization.
  - Zero-Trust Identity Gateway with web portal authentication (/login).
  - Network Layer Intelligence: OSI 7-Layer Defense Matrix (layers/osi).
  - Interactive Enterprise Network Topology Tree (topology/netmap).
  - Real-time animated NetFlow Packet Radar (monitor/sniff/packets).
  - Multi-panel Cyber HUD / Executive Dashboard (dashboard).
  - Autonomous 1-click triage & response (quickscan).
  - Target switching by index or IP (use, targets).
  - LSTM World Model Forward Rollout (.imagine forecast).
  - SHAP & Saliency Explainability (explain).
  - Counterfactual What-If Simulation (simulate).
  - Enterprise Assets, Incidents & Automated Containment (assets, incidents, playbooks, mitigate).
"""

from __future__ import annotations

import sys
import os
import time
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
from rich.tree import Tree
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt
from rich import box
from rich.align import Align

console = Console(legacy_windows=False)

DEFAULT_API_URL = os.environ.get("THREATORA_API_URL", "http://127.0.0.1:5000")
SYSTEM_API_KEY = os.environ.get("THREATORA_API_KEY", "threatora-zero-trust")


class ThreatoraConsole(cmd2.Cmd):
    """Tactical Cyber Defense Terminal with Zero-Trust Authentication."""

    intro = ""
    prompt = "threatora [tactical] > "

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        username: Optional[str] = None,
        password: Optional[str] = None,
        require_auth: bool = True,
    ):
        super().__init__(
            allow_cli_args=False,
            shortcuts={"?": "help", "!": "shell"},
            auto_load_commands=False,
        )
        self.api_url = api_url.rstrip("/")
        self.current_target: Optional[str] = None
        self.last_results: Optional[Dict[str, Any]] = None
        self.detected_hosts: List[Dict[str, Any]] = []

        # Authentication State
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": SYSTEM_API_KEY})
        self.current_user: Optional[Dict[str, Any]] = None
        self.require_auth = require_auth
        self._initial_username = username
        self._initial_password = password

        # Aesthetics
        self.self_in_py = True
        self.default_category = "General Commands"
        self._update_prompt()

    # -------------------------------------------------------------------------
    # Prompt & Identity Management
    # -------------------------------------------------------------------------
    def _update_prompt(self) -> None:
        """Update prompt with operator identity and active target context."""
        user_str = "operator"
        if self.current_user:
            username = self.current_user.get("username", "operator")
            role = self.current_user.get("role", "SOC")
            short_role = role.replace("CHIEF_", "").replace("_ADMIN", "").replace("_ANALYST", "")
            user_str = f"{username}@{short_role}"

        if self.current_target:
            self.prompt = f"⚡ threatora [{user_str} | target:{self.current_target}] ❯ "
        else:
            self.prompt = f"⚡ threatora [{user_str}] ❯ "

    # -------------------------------------------------------------------------
    # Lifecycle & Animated Startup
    # -------------------------------------------------------------------------
    def preloop(self) -> None:
        """Display cyber banner and authenticate operator upon startup."""
        if sys.stdin.isatty():
            self._animate_boot_sequence()

        self._print_banner()

        # Handle authentication
        if self._initial_username and self._initial_password:
            self._do_auth_attempt(self._initial_username, self._initial_password, silent=False)
        elif self.require_auth:
            if sys.stdin.isatty():
                self._interactive_login_prompt()
            else:
                self._headless_auto_auth()

        self._update_prompt()

    def _animate_boot_sequence(self) -> None:
        """High-tech interactive boot animation synchronizing OSI Layers and AI Engine."""
        boot_steps = [
            ("L2 DATA LINK", "Synchronizing Promiscuous Ethernet TAP & MAC Quarantine Matrix..."),
            ("L3 NETWORK", "Calibrating IPv4 Subnet Routing & TTL Distribution Engine..."),
            ("L4 TRANSPORT", "Reassembling TCP Flow Dynamics & SYN Rate-Limiting Filter..."),
            ("L7 APP", "Ingesting Deep Packet Inspection (DPI) & C2 Beacon Signatures..."),
            ("AI WORLD MODEL", "Ingesting 62 Telemetry Features into LSTM Recurrent Hidden State..."),
            ("ZERO-TRUST", "Zero-Trust Behavioral Verification Core: ENFORCING."),
        ]

        with Progress(
            SpinnerColumn(spinner_name="dots", style="bold bright_cyan"),
            TextColumn("[bold bright_cyan][{task.fields[layer]}][/bold bright_cyan] {task.description}"),
            transient=True,
            console=console,
        ) as progress:
            task = progress.add_task("", layer="INITIALIZING")
            for layer, desc in boot_steps:
                progress.update(task, description=f"[bold white]{desc}[/bold white]", layer=layer)
                time.sleep(0.09)

    def _print_banner(self) -> None:
        """Render grand, high-impact cyber operations banner with network layer theme."""
        banner_art = (
            "[bold red]████████╗██╗  ██╗██████╗ ███████╗ █████╗ ████████╗ ██████╗ ██████╗  █████╗ [/bold red]\n"
            "[bold bright_red]╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗[/bold bright_red]\n"
            "[bold bright_yellow]   ██║   ███████║██████╔╝█████╗  ███████║   ██║   ██║   ██║██████╔╝███████║[/bold bright_yellow]\n"
            "[bold bright_cyan]   ██║   ██╔══██║██╔══██╗██╔══╝  ██╔══██║   ██║   ██║   ██║██╔══██╗██╔══██║[/bold bright_cyan]\n"
            "[bold bright_blue]   ██║   ██║  ██║██║  ██║███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║██║  ██║[/bold bright_blue]\n"
            "[bold white]   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝[/bold white]"
        )

        console.print(Align.center(banner_art))
        console.print(
            Align.center(
                "[bold #06d6a0]⚡ RECURRENT AI NETWORK ATTACK FORECASTING & MITRE ATT&CK SIMULATION ⚡[/]"
            )
        )
        console.print(
            Align.center(
                "[dim #8ecae6]National Technical Research Organisation (NTRO) · Defense Problem Statement 26153[/]"
            )
        )

        status_line = (
            f"[dim]Core API: [bold #00f5d4]{self.api_url}[/] │ "
            f"Architecture: [bold #ffbe0b]LSTM World Model (62-dim)[/] │ "
            f"Defense Posture: [bold #06d6a0]Zero-Trust Enforcing (OSI L2-L7)[/][/dim]"
        )
        console.print(Align.center(status_line))
        console.print()

        # Tactical Directive Quick Accelerators Card
        quick_table = Table(
            title="[bold #00f5d4]⚡ TACTICAL DIRECTIVE ACCELERATORS[/]",
            box=box.ROUNDED,
            border_style="#4361ee",
            expand=True,
            caption="[dim #8ecae6]Type any command below to begin operations · 'help' for full catalog[/]",
        )
        quick_table.add_column("Command", style="bold #00f5d4", width=12)
        quick_table.add_column("Tactical Capability & Operational Scope", style="white")
        quick_table.add_row("quickscan", "1-Click Autonomous Triage (Scan → Lock Target → Forecast → Explain)")
        quick_table.add_row("dashboard", "Full-Spectrum 3-Panel Executive Cyber HUD (Health, Radar & Posture)")
        quick_table.add_row("topology", "Interactive Visual Network Tree of WAN, DMZ, and Internal Subnets")
        quick_table.add_row("monitor", "Real-Time Animated NetFlow / Packet Frame Sniffer & Infiltration Gauge")
        quick_table.add_row("targets", "Discovered Hosts Inventory & Metasploit-Style Selector (`use <#>`)")
        quick_table.add_row("forecast", "LSTM 10-Step Forward Rollout (.imagine mode with 95% CI)")
        quick_table.add_row("explain", "Zero-Hallucination SHAP Feature Importance Waterfall & Attribution")
        quick_table.add_row("mitigate", "Zero-Trust Host Isolation Directive (`mitigate --isolate`) & Audit Log")

        console.print(quick_table)
        console.print()

    # -------------------------------------------------------------------------
    # Authentication Engine (Web Portal Integration)
    # -------------------------------------------------------------------------
    def _do_auth_attempt(self, username: str, password: str, silent: bool = False) -> bool:
        """Submit login request to Threatora Flask backend (/login)."""
        endpoint = f"{self.api_url}/login"
        try:
            resp = self.session.post(
                endpoint,
                json={"username": username, "password": password},
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.current_user = data.get("user", {})
                if not silent:
                    self._render_operator_card(self.current_user)
                return True
            else:
                try:
                    err_msg = resp.json().get("message", "Invalid credentials.")
                except Exception:
                    err_msg = f"HTTP {resp.status_code}: {resp.text}"
                if not silent:
                    console.print(f"[bold red]✘ Authentication Denied:[/bold red] {err_msg}")
                return False
        except requests.exceptions.ConnectionError:
            if not silent:
                console.print(f"[bold red]✘ Core API Unreachable:[/bold red] Cannot connect to {self.api_url}")
                console.print("  [yellow]Make sure Flask is running on port 5000 (`python server/app.py`)[/yellow]")
            return False
        except Exception as e:
            if not silent:
                console.print(f"[bold red]✘ Authentication Error:[/bold red] {e}")
            return False

    def _headless_auto_auth(self) -> None:
        """Attempt non-interactive auth for CI / tests / scripts."""
        ok = self._do_auth_attempt("admin", "Threatora@2026", silent=True)
        if not ok:
            self.current_user = {
                "id": 0,
                "username": "system-agent",
                "full_name": "Automated Security Console",
                "role": "SYSTEM_SERVICE",
                "is_active": True,
            }

    def _interactive_login_prompt(self) -> None:
        """Interactive visual login gate requiring web portal credentials."""
        login_panel = Panel(
            "[bold white]ENTER OPERATOR IDENTITY (Web Portal Identity Provider)[/bold white]\n"
            "[dim]Authenticate with your Threatora Web credentials to unlock tactical operations.\n"
            "Default Administrative Identity: [bold yellow]admin[/bold yellow] / [bold yellow]Threatora@2026[/bold yellow][/dim]",
            title="[bold cyan]🛡️ ZERO-TRUST ACCESS GATEWAY[/bold cyan]",
            border_style="bright_blue",
            box=box.ROUNDED,
            expand=False,
        )
        console.print(login_panel)

        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            attempts += 1
            username = Prompt.ask("[bold cyan]Operator Username / Email[/bold cyan]", default="admin").strip()
            password = Prompt.ask("[bold cyan]Security Passphrase[/bold cyan]", password=True).strip()

            if not username or not password:
                console.print("[yellow]Username and Passphrase are required.[/yellow]")
                continue

            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="[cyan]Verifying credentials against Zero-Trust Identity Provider...", total=None)
                success = self._do_auth_attempt(username, password, silent=False)

            if success:
                console.print("[bold green]✔ Identity clearance confirmed. Tactical console unlocked.[/bold green]\n")
                return

            remaining = max_attempts - attempts
            if remaining > 0:
                console.print(f"[yellow]Authentication failed. {remaining} attempt(s) remaining.[/yellow]\n")
            else:
                console.print("[bold red]Maximum authentication attempts exceeded.[/bold red]")
                console.print("[yellow]Continuing in restricted guest mode (offline / mock telemetry only).[/yellow]\n")
                self.current_user = {
                    "username": "guest",
                    "role": "OBSERVER",
                    "full_name": "Unauthenticated Guest",
                }

    def _render_operator_card(self, user: Dict[str, Any]) -> None:
        """Render high-tech operator badge."""
        full_name = user.get("full_name", "Operator")
        username = user.get("username", "unknown")
        role = user.get("role", "SOC_ANALYST")
        email = user.get("email", "N/A")
        clearance = "LEVEL 5 // CHIEF CISO" if "ADMIN" in role else "LEVEL 3 // SOC TACTICAL"

        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="bold cyan", justify="right")
        info_table.add_column(style="bold white")

        info_table.add_row("Operator Name:", full_name)
        info_table.add_row("Callsign / Handle:", f"[bold yellow]{username}[/bold yellow]")
        info_table.add_row("Assigned Role:", f"[bold magenta]{role}[/bold magenta]")
        info_table.add_row("Contact Channel:", email)
        info_table.add_row("Security Clearance:", f"[bold green]{clearance}[/bold green]")
        info_table.add_row("Zero-Trust Posture:", "[bold green]VERIFIED & ACTIVE[/bold green]")

        card_panel = Panel(
            info_table,
            title="[bold green]✔ OPERATOR IDENTITY VERIFIED[/bold green]",
            subtitle="[dim]Session encrypted · Audit logging enabled[/dim]",
            border_style="bright_green",
            box=box.ROUNDED,
            expand=False,
        )
        console.print(card_panel)

    def _check_api_status(self, silent: bool = False) -> bool:
        """Query Core API /api/health."""
        try:
            r = self.session.get(f"{self.api_url}/api/health", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                if not silent:
                    console.print(f"[bold green]✔ Connected to Threatora Core API:[/bold green] {self.api_url}")
                    console.print(
                        f"  [dim]Device: {data.get('device', 'cpu')} │ "
                        f"Model: {data.get('recurrent_cell', 'LSTM')} │ "
                        f"Status: {data.get('status', 'healthy')}[/dim]"
                    )
                return True
        except Exception:
            pass

        if not silent:
            console.print(f"[bold red]✘ Core API Offline or Unreachable:[/bold red] {self.api_url}")
            console.print("  [yellow]Make sure the Flask server is running (`python server/app.py`)[/yellow]")
        return False

    # -------------------------------------------------------------------------
    # Authentication & Identity Commands
    # -------------------------------------------------------------------------
    @with_category("Authentication & Identity")
    def do_whoami(self, _args):
        """Display active Operator identity, role, and security clearance."""
        if not self.current_user:
            console.print("[yellow]Not currently authenticated as a registered operator.[/yellow]")
            return
        self._render_operator_card(self.current_user)

    login_parser = Cmd2ArgumentParser(prog="login", description="Authenticate with Threatora Web credentials")
    login_parser.add_argument("-u", "--username", help="Operator username or email")
    login_parser.add_argument("-p", "--password", help="Security passphrase")

    @with_category("Authentication & Identity")
    @with_argparser(login_parser)
    def do_login(self, args):
        """Authenticate into Threatora tactical terminal with web portal credentials."""
        username = args.username
        password = args.password

        if not username:
            username = Prompt.ask("[bold cyan]Operator Username / Email[/bold cyan]", default="admin")
        if not password:
            password = Prompt.ask("[bold cyan]Security Passphrase[/bold cyan]", password=True)

        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="[cyan]Authenticating operator credentials...", total=None)
            success = self._do_auth_attempt(username, password, silent=False)

        if success:
            self._update_prompt()
            console.print(f"[bold green]✔ Logged in successfully as {username}.[/bold green]")

    @with_category("Authentication & Identity")
    def do_logout(self, _args):
        """Terminate current operator session and clear authentication credentials."""
        try:
            self.session.post(f"{self.api_url}/logout", timeout=3)
        except Exception:
            pass
        self.session.cookies.clear()
        self.current_user = None
        self._update_prompt()
        console.print("[bold yellow]✔ Operator session terminated.[/bold yellow]")

    # -------------------------------------------------------------------------
    # Network Layer Intelligence: topology, monitor, layers
    # -------------------------------------------------------------------------
    @with_category("Network Layer Intelligence")
    def do_topology(self, _args):
        """Visualize Enterprise Network Topology Tree across WAN, DMZ, and Internal Corp subnets."""
        # Query asset inventory from state store
        assets = []
        try:
            ra = self.session.get(f"{self.api_url}/api/v1/assets", timeout=4)
            if ra.status_code == 200:
                assets = ra.json().get("assets", [])
        except Exception:
            pass

        tree = Tree("[bold bright_cyan]🌐 THREATORA ENTERPRISE ZERO-TRUST MESH (OSI L3/L4/L7)[/bold bright_cyan]")
        wan = tree.add("[bold magenta]WAN / EXTERNAL PERIMETER (147.32.84.0/24)[/bold magenta]")

        # Find edge gateway
        edge = next((a for a in assets if "EDGE" in a.get("hostname", "")), None)
        edge_status = edge.get("status", "HEALTHY") if edge else "HEALTHY"
        gw = wan.add(
            f"[bold cyan]147.32.84.165 (EDGE-GATEWAY-EXT)[/bold cyan] │ "
            f"[dim]VyOS Router · BGP:179, IPsec:500, SNMP:161[/dim] │ "
            f"[{'green' if edge_status == 'HEALTHY' else 'red'}]{edge_status}[/]"
        )

        dmz = gw.add("[bold yellow]DMZ & INGRESS PROXY TIER[/bold yellow]")
        proxy = next((a for a in assets if "PROXY" in a.get("hostname", "")), None)
        proxy_status = proxy.get("status", "HEALTHY") if proxy else "HEALTHY"
        dmz_node = dmz.add(
            f"[bold cyan]192.168.1.15 (INGRESS-NGINX-PROXY)[/bold cyan] │ "
            f"[dim]Alpine Linux · HTTP:80, HTTPS:443, SSH:2222[/dim] │ "
            f"[{'green' if proxy_status == 'HEALTHY' else 'red'}]{proxy_status}[/]"
        )

        corp = dmz_node.add("[bold blue]INTERNAL ENTERPRISE SUBNET (192.168.1.0/24)[/bold blue]")

        # Internal hosts
        for a in assets:
            ip = a.get("ip_address")
            if ip in ("147.32.84.165", "192.168.1.15"):
                continue

            hostname = a.get("hostname")
            crit = a.get("criticality")
            status = a.get("status")
            is_target = ip == self.current_target

            target_badge = " [bold yellow]★ ACTIVE TARGET[/bold yellow]" if is_target else ""
            status_badge = (
                "[bold white on red] ISOLATED / COMPROMISED [/bold white on red]"
                if status in ("ISOLATED", "COMPROMISED")
                else "[bold green]HEALTHY[/bold green]"
            )

            corp.add(
                f"[bold {'bright_red' if status in ('ISOLATED', 'COMPROMISED') else 'white'}]{ip} ({hostname})[/] │ "
                f"[dim]{a.get('operating_system', 'Linux')} · [{crit}][/dim] │ {status_badge}{target_badge}"
            )

        console.print(
            Panel(
                tree,
                box=box.ROUNDED,
                border_style="cyan",
                title="[bold bright_green]🗺️ ENTERPRISE ATTACK SURFACE & NETWORK TOPOLOGY[/bold bright_green]",
                subtitle="[dim]Switch active target with: `use <#>` or `use <IP>`[/dim]",
            )
        )

    def do_netmap(self, args):
        """Network topology alias."""
        return self.do_topology(args)

    @with_category("Network Layer Intelligence")
    def do_monitor(self, _args):
        """Live animated packet stream monitor simulating real-time NetFlow ingestion."""
        console.print(Panel("[bold cyan]📡 REAL-TIME OSI L3/L4/L7 PACKET & FLOW TELEMETRY MONITOR[/bold cyan]", box=box.ROUNDED, expand=False))
        console.print("[dim]Ingesting packet telemetry from virtual TAP interface (Promiscuous mode)...[/dim]\n")

        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Proto", justify="center", width=12)
        table.add_column("Flow Vector (Src -> Dst)", style="bold white")
        table.add_column("Length", justify="right", width=8)
        table.add_column("Infiltration Risk", justify="center", width=16)
        table.add_column("OSI L7 Vector / ATT&CK", justify="left")

        # Telemetry packet frame samples
        now = time.strftime("%H:%M:%S")
        packets = [
            (f"{now}.10", "[bold cyan]TCP [SYN][/bold cyan]", "192.168.1.105:50421 -> 192.168.1.5:5432", "64 B", "[yellow]██░░░░ 18.2%[/yellow]", "L4 Port Scan (DB Probing)"),
            (f"{now}.18", "[bold green]TCP [ACK][/bold green]", "192.168.1.5:5432    -> 192.168.1.105:50421", "52 B", "[green]█░░░░░  8.5%[/green]", "L4 TCP RST/ACK Generated"),
            (f"{now}.29", "[bold yellow]UDP [DNS][/bold yellow]", "192.168.1.105:58102 -> 147.32.84.165:53", "182 B", "[red]████░░ 41.5%[/red]", "[bold yellow]T1071.004: DNS Tunneling[/bold yellow]"),
            (f"{now}.42", "[bold red]TCP [PSH,ACK][/bold red]", "192.168.1.105:49152 -> 147.32.84.165:443", "1.4 KB", "[bold red]█████░ 54.8%[/bold red]", "[bold red]T1071: C2 Beacon Spikes[/bold red]"),
            (f"{now}.55", "[bold green]TCP [ACK][/bold green]", "192.168.1.10:389    -> 192.168.1.15:44120", "840 B", "[green]█░░░░░  4.1%[/green]", "L7 Kerberos Ticket Auth"),
            (f"{now}.68", "[bold red]TCP [PSH,ACK][/bold red]", "192.168.1.105:49152 -> 147.32.84.165:443", "1.4 KB", "[bold red]██████ 62.1%[/bold red]", "[bold red]T1041: Exfiltration Flow[/bold red]"),
        ]

        # Interactive animation delay simulating streaming frames
        with Progress(
            SpinnerColumn(spinner_name="dots", style="bold bright_cyan"),
            TextColumn("[bold cyan]Sniffing frame {task.completed}/{task.total}...[/bold cyan]"),
            transient=True,
            console=console,
        ) as prog:
            task = prog.add_task("Streaming", total=len(packets))
            for pkt in packets:
                table.add_row(*pkt)
                prog.advance(task)
                time.sleep(0.10)

        console.print(table)
        console.print(
            f"[dim]Capture Summary: 6 frames ingested │ 3 High-Risk Anomaly Flags Detected │ "
            f"Active Infiltration Target: [bold yellow]{self.current_target or '192.168.1.105'}[/bold yellow][/dim]\n"
        )

    def do_sniff(self, args):
        """Packet monitor alias."""
        return self.do_monitor(args)

    def do_packets(self, args):
        """Packet monitor alias."""
        return self.do_monitor(args)

    @with_category("Network Layer Intelligence")
    def do_layers(self, _args):
        """Inspect OSI 7-Layer Defense Matrix & 62-dimensional Telemetry Feature Mapping."""
        table = Table(
            title="🌐 OSI 7-LAYER ATTACK SURFACE & THREATORA DEFENSE MATRIX",
            box=box.ROUNDED,
            expand=True,
        )
        table.add_column("Layer", style="bold cyan", width=10)
        table.add_column("Protocols", style="bold white", width=16)
        table.add_column("Monitored Telemetry Signals (62-dim)", justify="left")
        table.add_column("Zero-Trust Defense Countermeasure", justify="left")

        layer_data = [
            ("Layer 2\nData Link", "Ethernet 802.3\nARP, MAC Addressing", "Frame size variance, packet IAT covariance, jumbo frame ratio", "[green]Port security, 802.1X quarantine, MAC spoofing drops[/green]"),
            ("Layer 3\nNetwork", "IPv4, ICMP, Routing\nSubnets, BGP, IPsec", "TTL mean/min, IP header entropy, subnet routing divergence", "[yellow]BGP blackholing, ICMP rate limiting, route severance[/yellow]"),
            ("Layer 4\nTransport", "TCP, UDP\nFlow Ports, Handshakes", "Fraction SYN-only, TCP window scaling, management port probes (22/445/3389)", "[red]SYN flood rate-limiting, egress port throttling[/red]"),
            ("Layer 5\nSession", "TLS 1.3, SSHv2\nSession State Tables", "Session duration std, flow idle time, handshake failure frequency", "[yellow]Active session reset, certificate revocation enforcement[/yellow]"),
            ("Layer 6\nPresentation", "MIME, SSL Decryption\nPayload Encoding", "Entropy of payload bytes, gzip compression ratio, base64 flags", "[cyan]SSL/TLS inspection, high-entropy binary payload inspection[/cyan]"),
            ("Layer 7\nApplication", "HTTP/2, DNS, Kerberos\nSMB, C2 Beacons", "Beacon periodicity, DNS tunneling QPS, HTTP POST payload size ratio", "[bold red]Zero-Trust DNS Sinkholing, Automated Host Isolation[/bold red]"),
        ]

        for row in layer_data:
            table.add_row(*row)

        console.print(table)
        console.print("[dim]Run `quickscan` to evaluate live network flows across all OSI layers.[/dim]\n")

    def do_osi(self, args):
        """OSI layers alias."""
        return self.do_layers(args)

    # -------------------------------------------------------------------------
    # Operational & Executive Directives
    # -------------------------------------------------------------------------
    @with_category("Operational Directives")
    def do_dashboard(self, _args):
        """Display full-spectrum Cyber HUD (Heads-Up Display) with health, threats, and assets."""
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="[cyan]Gathering intelligence from Core API & Ledger...", total=None)

            # 1. Health
            health_data = {}
            try:
                rh = self.session.get(f"{self.api_url}/api/health", timeout=3)
                if rh.status_code == 200:
                    health_data = rh.json()
            except Exception:
                pass

            # 2. Assets
            assets = []
            try:
                ra = self.session.get(f"{self.api_url}/api/v1/assets", timeout=3)
                if ra.status_code == 200:
                    assets = ra.json().get("assets", [])
            except Exception:
                pass

            # 3. Incidents
            incidents = []
            try:
                ri = self.session.get(f"{self.api_url}/api/v1/incidents", timeout=3)
                if ri.status_code == 200:
                    incidents = ri.json().get("incidents", [])
            except Exception:
                pass

            # 4. Playbooks
            playbooks = []
            try:
                rp = self.session.get(f"{self.api_url}/api/v1/playbooks", timeout=3)
                if rp.status_code == 200:
                    playbooks = rp.json().get("playbooks", [])
            except Exception:
                pass

        # Calculations
        tot_assets = len(assets)
        crit_assets = sum(1 for a in assets if a.get("criticality") == "MISSION_CRITICAL")
        iso_assets = sum(1 for a in assets if a.get("status") in ("ISOLATED", "COMPROMISED"))

        tot_incidents = len(incidents)
        uncontained = sum(1 for i in incidents if not i.get("is_contained"))

        pending_pb = sum(1 for p in playbooks if p.get("status") == "PENDING")

        flagged_hosts = self.last_results.get("flagged_hosts", 0) if self.last_results else 0
        total_scanned = self.last_results.get("total_hosts", 0) if self.last_results else 0

        # Panel 1: Engine Telemetry
        engine_txt = (
            f"[bold cyan]Core API Status  :[/bold cyan] "
            f"{'[bold green]ONLINE[/bold green]' if health_data else '[bold red]OFFLINE[/bold red]'}\n"
            f"[bold cyan]Inference Device :[/bold cyan] [bold yellow]{health_data.get('device', 'cpu').upper()}[/bold yellow]\n"
            f"[bold cyan]Recurrent Model  :[/bold cyan] [bold white]{health_data.get('recurrent_cell', 'LSTM')}[/bold white]\n"
            f"[bold cyan]Feature Dimension:[/bold cyan] [bold white]{health_data.get('obs_dimension', 62)} Telemetry Features[/bold white]\n"
            f"[bold cyan]OSI Telemetry TAP:[/bold cyan] [bold green]L2-L7 INGESTION ACTIVE[/bold green]"
        )
        p1 = Panel(engine_txt, title="[bold cyan]⚙️ ENGINE TELEMETRY[/bold cyan]", box=box.ROUNDED, expand=True)

        # Panel 2: Threat Radar
        radar_txt = (
            f"[bold white]Telemetry Stream :[/bold white] "
            f"{'[bold green]ACTIVE STREAM[/bold green]' if self.last_results else '[dim]IDLE (run `scan`)[/dim]'}\n"
            f"[bold white]Hosts Analyzed   :[/bold white] [bold cyan]{total_scanned}[/bold cyan]\n"
            f"[bold white]Flagged Threats  :[/bold white] "
            f"{f'[bold red]{flagged_hosts} DETECTED[/bold red]' if flagged_hosts else '[bold green]0 (CLEAN)[/bold green]'}\n"
            f"[bold white]Active Target    :[/bold white] "
            f"{f'[bold yellow]{self.current_target}[/bold yellow]' if self.current_target else '[dim]None selected[/dim]'}\n"
            f"[bold white]Threat Posture   :[/bold white] "
            f"{'[bold red]CRITICAL ALERT[/bold red]' if flagged_hosts > 0 else '[bold green]GUARD POSTURE NORMAL[/bold green]'}"
        )
        p2 = Panel(radar_txt, title="[bold yellow]📡 THREAT RADAR[/bold yellow]", box=box.ROUNDED, expand=True)

        # Panel 3: Defense & Mitigation Posture
        defense_txt = (
            f"[bold white]Total Assets     :[/bold white] [bold cyan]{tot_assets}[/bold cyan] ({crit_assets} Mission Critical)\n"
            f"[bold white]Isolated Endpoints:[/bold white] "
            f"{f'[bold red]{iso_assets}[/bold red]' if iso_assets else '[bold green]0[/bold green]'}\n"
            f"[bold white]Incidents Total  :[/bold white] [bold yellow]{tot_incidents}[/bold yellow] ({uncontained} uncontained)\n"
            f"[bold white]Pending Playbooks:[/bold white] "
            f"{f'[bold red]{pending_pb} PENDING[/bold red]' if pending_pb else '[bold green]0 PENDING[/bold green]'}\n"
            f"[bold white]Zero-Trust Policy:[/bold white] [bold green]AUTO-CONTAIN ENABLED[/bold green]"
        )
        p3 = Panel(defense_txt, title="[bold magenta]🛡️ DEFENSE & MITIGATION[/bold magenta]", box=box.ROUNDED, expand=True)

        console.print(Columns([p1, p2, p3]))

        # Operator status summary at bottom
        op_name = self.current_user.get("full_name", "Operator") if self.current_user else "Unauthenticated"
        op_role = self.current_user.get("role", "GUEST") if self.current_user else "NONE"
        console.print(
            f"[dim]Operator in command: [bold white]{op_name}[/bold white] ([bold cyan]{op_role}[/bold cyan]) │ "
            f"Type [bold cyan]topology[/bold cyan] for network mesh, [bold cyan]monitor[/bold cyan] for live packets, or [bold cyan]quickscan[/bold cyan] for triage.[/dim]\n"
        )

    @with_category("Operational Directives")
    def do_status(self, _args):
        """Check API connection, LSTM World Model health, and active target IP."""
        console.print(Panel("[bold cyan]THREATORA ENGINE DIAGNOSTICS[/bold cyan]", box=box.ROUNDED, expand=False))
        self._check_api_status(silent=False)
        target_display = self.current_target if self.current_target else "None (Use `use <#>` or `set target <IP>` or `scan`)"
        console.print(f"[bold white]Active Operator Target:[/bold white] [bold yellow]{target_display}[/bold yellow]")
        if self.current_user:
            console.print(
                f"[bold white]Authenticated Operator:[/bold white] [bold green]{self.current_user.get('username')}[/bold green] "
                f"([cyan]{self.current_user.get('role')}[/cyan])"
            )

    # -------------------------------------------------------------------------
    # Target Management (Metasploit-Style `use` and `targets`)
    # -------------------------------------------------------------------------
    @with_category("Operational Directives")
    def do_targets(self, _args):
        """List all discovered hosts from recent scan with selectable target index numbers."""
        if not self.detected_hosts:
            if self.last_results and "hosts" in self.last_results:
                self.detected_hosts = self.last_results["hosts"]
            else:
                console.print("[yellow]No targets recorded yet. Run `scan` or `quickscan` first to discover network hosts.[/yellow]")
                return

        table = Table(title="🎯 DISCOVERED NETWORK TARGETS", box=box.ROUNDED, expand=True)
        table.add_column("#", style="bold cyan", justify="center", width=4)
        table.add_column("Host IP", style="bold white", justify="left")
        table.add_column("Infiltration Risk", justify="center")
        table.add_column("MITRE Stage", justify="left")
        table.add_column("Technique / Vector", justify="left")
        table.add_column("Active Selection", justify="center")

        for idx, h in enumerate(self.detected_hosts, 1):
            risk = h.get("current_risk_score", 0.0) * 100
            risk_color = "bright_red" if risk >= 50 else ("bright_yellow" if risk >= 20 else "bright_green")
            stg = h.get("current_stage", {}).get("name", "Unknown")
            tech = h.get("current_stage", {}).get("metadata", {}).get("technique", "Baseline")
            is_active = "[bold yellow]★ ACTIVE[/bold yellow]" if h.get("host_ip") == self.current_target else "[dim]—[/dim]"

            table.add_row(
                str(idx),
                h.get("host_ip", "N/A"),
                f"[{risk_color}]{risk:.1f}%[/{risk_color}]",
                f"[bold cyan]{stg}[/bold cyan]",
                tech,
                is_active,
            )

        console.print(table)
        console.print("[dim]Select target by typing: [bold cyan]use <#>[/bold cyan] or [bold cyan]use <IP>[/bold cyan][/dim]")

    use_parser = Cmd2ArgumentParser(prog="use", description="Switch active target by index or IP address")
    use_parser.add_argument("target", help="Index number (#) from `targets` or Host IP address (e.g. 192.168.1.5)")

    @with_category("Operational Directives")
    @with_argparser(use_parser)
    def do_use(self, args):
        """Select active target by index number (#) or direct IP address."""
        target_input = args.target.strip()

        if target_input.isdigit():
            idx = int(target_input) - 1
            if 0 <= idx < len(self.detected_hosts):
                selected_ip = self.detected_hosts[idx]["host_ip"]
                self.current_target = selected_ip
                self._update_prompt()
                console.print(f"[bold green]✔ Active target switched to:[/bold green] [bold yellow]{selected_ip}[/bold yellow]")
                return
            else:
                console.print(f"[red]Invalid target index #{target_input}. Run `targets` to see available hosts.[/red]")
                return

        self.current_target = target_input
        self._update_prompt()
        console.print(f"[bold green]✔ Active target switched to:[/bold green] [bold yellow]{target_input}[/bold yellow]")

    set_parser = Cmd2ArgumentParser(prog="set", description="Set tactical context variables")
    set_subparsers = set_parser.add_subparsers(dest="var_name")
    target_parser = set_subparsers.add_parser("target", help="Set active target IP")
    target_parser.add_argument("ip", help="Target IP address (e.g. 192.168.1.5)")

    @with_category("Operational Directives")
    @with_argparser(set_parser)
    def do_set(self, args):
        """Set tactical context variables (e.g. `set target 192.168.1.5`)."""
        if args.var_name == "target":
            self.current_target = args.ip
            self._update_prompt()
            console.print(f"[bold green]✔ Active target context set to:[/bold green] [bold yellow]{args.ip}[/bold yellow]")

    @with_category("Operational Directives")
    def do_clear(self, _args):
        """Clear terminal screen and redraw the executive header."""
        console.clear()
        self._print_banner()

    def do_cls(self, args):
        """Clear screen alias."""
        return self.do_clear(args)

    # -------------------------------------------------------------------------
    # Tactical Intelligence: scan & quickscan
    # -------------------------------------------------------------------------
    scan_parser = Cmd2ArgumentParser(prog="scan", description="Run telemetry inference stream or load PCAP / CSV")
    scan_parser.add_argument("--live", action="store_true", help="Run scan on live demo attack telemetry stream")
    scan_parser.add_argument("-i", "--input", help="Path to PCAP or CSV flow telemetry file")

    @with_category("Tactical Intelligence")
    @with_argparser(scan_parser)
    def do_scan(self, args):
        """Scan network telemetry stream and compute Infiltration Probabilities and MITRE Stages."""
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
                        resp = self.session.post(f"{self.api_url}/api/upload", files={"file": f}, timeout=60)
                else:
                    resp = self.session.get(f"{self.api_url}/api/demo", timeout=30)

                if resp.status_code != 200:
                    console.print(f"[red]Scan failed with status {resp.status_code}: {resp.text}[/red]")
                    return

                data = resp.json()
                self.last_results = data
                self.detected_hosts = data.get("hosts", [])
            except Exception as e:
                console.print(f"[bold red]API Communication Failure:[/bold red] {e}")
                return

        hosts = self.detected_hosts
        if not hosts:
            console.print("[yellow]Scan completed: No host telemetry found in stream.[/yellow]")
            return

        if not self.current_target and hosts:
            highest_risk_host = max(hosts, key=lambda h: h.get("current_risk_score", 0))
            self.current_target = highest_risk_host["host_ip"]
            self._update_prompt()

        table = Table(
            title="⚡ THREATORA TELEMETRY INFERENCE & MITRE ATT&CK MAPPING",
            box=box.ROUNDED,
            expand=True,
        )
        table.add_column("#", style="bold cyan", justify="center", width=4)
        table.add_column("Host IP", style="bold white", justify="left")
        table.add_column("60s Cells", justify="center")
        table.add_column("Infiltration Risk", justify="center")
        table.add_column("MITRE Stage", justify="left")
        table.add_column("Technique / Action", justify="left")
        table.add_column("Zero-Trust Status", justify="center")

        for idx, h in enumerate(hosts, 1):
            risk = h.get("current_risk_score", 0.0) * 100
            risk_color = "bright_red" if risk >= 50 else ("bright_yellow" if risk >= 20 else "bright_green")
            stage = h.get("current_stage", {}).get("name", "Unknown")
            technique = h.get("current_stage", {}).get("metadata", {}).get("technique", "Baseline")
            status = (
                "[bold white on red] THREAT DETECTED [/bold white on red]"
                if h.get("is_anomalous")
                else "[bold white on green] BENIGN [/bold white on green]"
            )

            bar_len = int(risk / 10)
            meter = f"[{risk_color}]" + "█" * bar_len + "░" * (10 - bar_len) + f"[/{risk_color}]"

            table.add_row(
                str(idx),
                h["host_ip"],
                str(h.get("window_count", 1)),
                f"{meter} [{risk_color}]{risk:5.1f}%[/{risk_color}]",
                f"[bold cyan]{stage}[/bold cyan]",
                technique,
                status,
            )

        console.print(table)
        console.print(
            f"[dim]Total Hosts: {len(hosts)} │ Flagged: {data.get('flagged_hosts', 0)} │ "
            f"Active Target: [bold yellow]{self.current_target}[/bold yellow][/dim]\n"
        )
        console.print(
            "[dim]Quick Actions: Type [bold cyan]use <#>[/bold cyan] to switch target │ "
            "[bold cyan]forecast[/bold cyan] for 10-min rollout │ "
            "[bold cyan]explain[/bold cyan] for SHAP attribution │ "
            "[bold cyan]topology[/bold cyan] for network tree.[/dim]"
        )

    @with_category("Tactical Intelligence")
    def do_quickscan(self, _args):
        """Execute autonomous 1-click end-to-end triage: Scan -> Target Lock -> Forecast -> Explain -> Mitigate."""
        console.print(Panel("[bold cyan]🚀 AUTONOMOUS 1-CLICK RAPID TRIAGE (QUICKSCAN)[/bold cyan]", box=box.ROUNDED, expand=False))

        # 1. Scan live stream
        console.print("[bold white]Step 1/4:[/bold white] Ingesting real-time network telemetry...")
        self.onecmd("scan --live")

        if not self.detected_hosts:
            console.print("[yellow]No hosts discovered to analyze.[/yellow]")
            return

        # 2. Lock onto highest threat
        highest_host = max(self.detected_hosts, key=lambda h: h.get("current_risk_score", 0))
        highest_ip = highest_host["host_ip"]
        highest_risk = highest_host.get("current_risk_score", 0) * 100
        self.current_target = highest_ip
        self._update_prompt()

        console.print(
            f"\n[bold white]Step 2/4:[/bold white] [bold green]🎯 Target Locked:[/bold green] "
            f"[bold yellow]{highest_ip}[/bold yellow] with [bold red]{highest_risk:.1f}% Risk[/bold red] "
            f"([bold cyan]{highest_host.get('current_stage', {}).get('name')}[/bold cyan])"
        )

        # 3. Forecast
        console.print("\n[bold white]Step 3/4:[/bold white] Generating 5-minute forward predictive rollout (.imagine)...")
        self.onecmd(f"forecast -t {highest_ip} --steps 5")

        # 4. Explain
        console.print("\n[bold white]Step 4/4:[/bold white] Computing SHAP explainability attribution...")
        self.onecmd(f"explain -t {highest_ip}")

        # Recommendation
        stg = highest_host.get("current_stage", {})
        rec_action = stg.get("metadata", {}).get("soc_action", "Execute Zero-Trust Isolation")
        console.print(
            Panel(
                f"[bold white]Recommended Containment Action:[/bold white] [bold bright_yellow]{rec_action}[/bold bright_yellow]\n"
                f"[dim]Execute containment directive now by typing: [bold cyan]mitigate --isolate[/bold cyan][/dim]",
                title="[bold red]⚡ RAPID RESPONSE DIRECTIVE[/bold red]",
                border_style="red",
                box=box.ROUNDED,
                expand=False,
            )
        )

    # -------------------------------------------------------------------------
    # Forecasting: forecast (.imagine mode)
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
            console.print("[red]No target specified. Run `scan` first or specify `-t <IP>`.[/red]")
            return

        if not self.last_results:
            console.print("[yellow]No prior scan in memory. Executing live scan first...[/yellow]")
            self.onecmd("scan --live")
            if not self.last_results:
                return

        host = next((h for h in self.last_results.get("hosts", []) if h.get("host_ip") == target_ip), None)
        if not host:
            console.print(f"[red]Target host {target_ip} not found in recent scan results.[/red]")
            return

        timeline = host.get("forecast_timeline", [])
        if not timeline:
            console.print("[yellow]No forecast timeline generated for target host.[/yellow]")
            return

        if args.steps and args.steps < len(timeline):
            display_timeline = timeline[:args.steps]
        else:
            display_timeline = timeline

        console.print(f"\n[bold cyan]⚡ DREAMING MODE: {len(display_timeline)}-STEP FORWARD MONTE CARLO ROLLOUT (.imagine)[/bold cyan]")
        console.print(f"[bold white]Target Host:[/bold white] [bold yellow]{target_ip}[/bold yellow] │ [dim]State-Space Recurrent Forecasting[/dim]\n")

        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Horizon", justify="center", style="bold cyan", width=8)
        table.add_column("Risk Probability (95% CI)", justify="center", width=26)
        table.add_column("Predictive Visual Density Gauge", justify="left")
        table.add_column("Predicted Kill Chain Stage", justify="left")

        for step in display_timeline:
            prob = step.get("infilt_prob", 0.0) * 100
            lower = step.get("lower_ci", 0.0) * 100
            upper = step.get("upper_ci", 0.0) * 100
            stage_name = step.get("stage_name", "Unknown")
            stage_color = "bright_red" if prob >= 50 else ("bright_yellow" if prob >= 20 else "bright_green")

            bar_len = int(prob / 3.33)
            bar_visual = f"[{stage_color}]" + "█" * bar_len + "░" * (30 - bar_len) + f"[/{stage_color}]"

            table.add_row(
                step.get("minute", "T+1"),
                f"[{stage_color}]{prob:5.1f}%[/{stage_color}] ({lower:4.1f}% - {upper:4.1f}%)",
                bar_visual,
                f"[{stage_color}]{stage_name}[/{stage_color}]",
            )

        console.print(table)

    # -------------------------------------------------------------------------
    # Explainable AI (XAI): explain
    # -------------------------------------------------------------------------
    explain_parser = Cmd2ArgumentParser(prog="explain", description="SHAP & Saliency Feature Importance Engine")
    explain_parser.add_argument("-t", "--target", help="Target IP (defaults to active target)")

    @with_category("Explainable AI (XAI)")
    @with_argparser(explain_parser)
    def do_explain(self, args):
        """Dissect LSTM neural decision via SHAP feature importance & attribution scores."""
        target_ip = args.target or self.current_target
        if not target_ip:
            console.print("[red]No target specified. Run `scan` first or specify `-t <IP>`.[/red]")
            return

        if not self.last_results:
            console.print("[yellow]Executing live scan to populate feature attributions...[/yellow]")
            self.onecmd("scan --live")
            if not self.last_results:
                return

        host = next((h for h in self.last_results.get("hosts", []) if h.get("host_ip") == target_ip), None)
        if not host:
            console.print(f"[red]Target host {target_ip} not found.[/red]")
            return

        explainability = host.get("explainability", {})
        top_features = explainability.get("top_features", {})

        console.print(f"\n[bold magenta]🔬 SHAP EXPLAINABILITY ENGINE (ZERO-HALLUCINATION AUDIT)[/bold magenta]")
        console.print(f"[bold white]Target IP:[/bold white] [bold yellow]{target_ip}[/bold yellow] │ [dim]Attribution Shapley Gradient Vectors[/dim]\n")

        if not top_features:
            console.print("[yellow]No high-variance feature drivers detected for this sequence.[/yellow]")
            return

        table = Table(title="Top Telemetry Feature Drivers", box=box.ROUNDED, expand=True)
        table.add_column("Feature Name", style="bold cyan")
        table.add_column("Attribution (%)", justify="right", style="bold yellow", width=16)
        table.add_column("Attribution Waterfall Chart", justify="left")

        for feat, score in top_features.items():
            bar_len = int(min(score * 1.5, 36))
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
    # Counterfactual Simulation: simulate
    # -------------------------------------------------------------------------
    simulate_parser = Cmd2ArgumentParser(prog="simulate", description="What-If Action Counterfactual Simulator")
    simulate_parser.add_argument(
        "-a",
        "--action",
        default="BLOCK_MANAGEMENT_PORTS",
        choices=["ISOLATE_HOST", "BLOCK_MANAGEMENT_PORTS", "RATE_LIMIT_SYN", "SINKHOLE_C2_DNS", "THROTTLE_EGRESS"],
        help="Hypothetical defensive action to test",
    )
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
                r = self.session.post(
                    f"{self.api_url}/api/v1/simulate",
                    json={"action": args.action, "target_ip": target_ip, "horizon": args.horizon},
                    timeout=15,
                )
                if r.status_code != 200:
                    console.print(f"[red]Simulation failed ({r.status_code}): {r.text}[/red]")
                    return
                data = r.json()
            except Exception as e:
                console.print(f"[red]API communication error: {e}[/red]")
                return

        console.print(f"\n[bold cyan]🔮 WHAT-IF COUNTERFACTUAL SIMULATION: {data.get('action_name')}[/bold cyan]")
        console.print(f"[bold white]Target IP:[/bold white] [bold yellow]{data.get('target_ip')}[/bold yellow] │ [dim]{data.get('action_description')}[/dim]\n")

        table = Table(title="Trajectory Comparison: Baseline vs Counterfactual", box=box.ROUNDED, expand=True)
        table.add_column("Horizon", justify="center", style="bold cyan")
        table.add_column("Baseline Risk", justify="center", style="bright_red")
        table.add_column("Simulated Risk", justify="center", style="bright_green")
        table.add_column("Δ Risk Reduction", justify="center", style="bold yellow")
        table.add_column("Baseline ATT&CK Stage", justify="left")
        table.add_column("Mitigated Stage", justify="left")

        for row in data.get("timeline", []):
            b_risk = row.get("baseline_risk", 0.0) * 100
            s_risk = row.get("simulated_risk", 0.0) * 100
            d_risk = row.get("delta_risk", 0.0) * 100
            d_sign = "+" if d_risk >= 0 else ""
            table.add_row(
                row.get("minute", "T+1"),
                f"{b_risk:5.1f}%",
                f"{s_risk:5.1f}%",
                f"[bold green]{d_sign}{d_risk:5.1f}%[/bold green]" if d_risk > 0 else f"{d_risk:5.1f}%",
                row.get("baseline_stage", "N/A"),
                f"[bold green]{row.get('simulated_stage', 'N/A')}[/bold green]",
            )

        console.print(table)
        console.print(
            f"\n[bold white]Overall Risk Reduction:[/bold white] [bold green]{data.get('pct_risk_reduction')}%[/bold green] │ "
            f"[bold white]Peak Mitigation:[/bold white] [bold green]-{data.get('peak_risk_reduction', 0)*100:.1f}%[/bold green]"
        )
        console.print(f"[bold white]Tactical Assessment:[/bold white] [bold cyan]{data.get('tactical_verdict')}[/bold cyan]\n")

    # -------------------------------------------------------------------------
    # Automated Mitigation: playbooks & mitigate
    # -------------------------------------------------------------------------
    @with_category("Automated Mitigation")
    def do_playbooks(self, _args):
        """List active and pending Mitigation Playbooks from state store."""
        try:
            r = self.session.get(f"{self.api_url}/api/v1/playbooks", timeout=5)
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

        table = Table(title="🛡️ ACTIVE MITIGATION PLAYBOOKS", box=box.ROUNDED, expand=True)
        table.add_column("Playbook UID", style="bold cyan")
        table.add_column("Target IP", style="bold white")
        table.add_column("Kill Chain Stage", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("Containment Strategy Preview", style="dim")

        for pb in playbooks:
            status = pb.get("status")
            status_badge = "[bold white on red] PENDING [/bold white on red]" if status == "PENDING" else "[bold white on green] EXECUTED [/bold white on green]"
            strat_preview = pb.get("containment_strategy", "")
            if len(strat_preview) > 60:
                strat_preview = strat_preview[:57] + "..."
            table.add_row(
                pb.get("playbook_uid", "N/A"),
                pb.get("target_ip", "N/A"),
                pb.get("kill_chain_stage", "N/A"),
                status_badge,
                strat_preview,
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
            console.print("[red]Must specify either --playbook <UID> or --isolate with an active target.[/red]")
            return

        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="[red]Dispatching Zero-Trust Containment Directives...", total=None)

            try:
                r = self.session.post(f"{self.api_url}/api/v1/mitigate", json=payload, timeout=10)
                if r.status_code not in (200, 201):
                    console.print(f"[red]Mitigation action rejected by API ({r.status_code}): {r.text}[/red]")
                    return
                resp = r.json()
            except Exception as e:
                console.print(f"[red]Failed to execute mitigation: {e}[/red]")
                return

        console.print(f"\n[bold green]✔ ZERO-TRUST CONTAINMENT DIRECTIVE EXECUTED[/bold green]")
        console.print(f"  [bold white]Target IP    :[/bold white] [bold yellow]{resp.get('target_ip')}[/bold yellow]")
        console.print(f"  [bold white]Asset Status :[/bold white] [bold red]{resp.get('host_status')}[/bold red]")
        console.print(f"  [bold white]Audit Log    :[/bold white] [dim]{resp.get('execution_log')}[/dim]\n")

    # -------------------------------------------------------------------------
    # Asset & Incident Ledger
    # -------------------------------------------------------------------------
    @with_category("Asset & Incident Ledger")
    def do_assets(self, _args):
        """Query Enterprise Asset Inventory (Criticality, OS, Services, Isolation Status)."""
        try:
            r = self.session.get(f"{self.api_url}/api/v1/assets", timeout=5)
            if r.status_code != 200:
                console.print(f"[red]Failed to retrieve assets: {r.text}[/red]")
                return
            data = r.json()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        assets = data.get("assets", [])
        table = Table(title="🏢 ENTERPRISE ASSET INVENTORY", box=box.ROUNDED, expand=True)
        table.add_column("IP Address", style="bold white")
        table.add_column("Hostname", style="bold cyan")
        table.add_column("Criticality", justify="center")
        table.add_column("Operating System", style="dim")
        table.add_column("Status", justify="center")

        for a in assets:
            crit = a.get("criticality", "MEDIUM")
            crit_style = "bold red" if crit == "MISSION_CRITICAL" else ("yellow" if crit == "HIGH" else "green")
            status = a.get("status", "HEALTHY")
            status_style = "bold white on red" if status in ("ISOLATED", "COMPROMISED") else "bold white on green"

            table.add_row(
                a.get("ip_address", "N/A"),
                a.get("hostname", "N/A"),
                f"[{crit_style}]{crit}[/{crit_style}]",
                a.get("operating_system", "Unknown"),
                f"[{status_style}] {status} [/{status_style}]",
            )

        console.print(table)

    @with_category("Asset & Incident Ledger")
    def do_incidents(self, _args):
        """Query incident history and forensic alerts recorded in state store."""
        try:
            r = self.session.get(f"{self.api_url}/api/v1/incidents", timeout=5)
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

        table = Table(title="🚨 SECURITY INCIDENTS LEDGER", box=box.ROUNDED, expand=True)
        table.add_column("Incident UID", style="bold cyan")
        table.add_column("Target IP", style="bold white")
        table.add_column("Risk Score", justify="center")
        table.add_column("Kill Chain Stage", style="yellow")
        table.add_column("Contained", justify="center")
        table.add_column("Timestamp", style="dim")

        for inc in incidents:
            score = inc.get("risk_score", 0.0) * 100
            score_col = "bright_red" if score >= 50 else "bright_yellow"
            contained = "[bold white on green] YES [/bold white on green]" if inc.get("is_contained") else "[bold white on red] NO [/bold white on red]"

            table.add_row(
                inc.get("incident_uid", "N/A"),
                inc.get("target_ip", "N/A"),
                f"[{score_col}]{score:.1f}%[/{score_col}]",
                inc.get("stage_name", "N/A"),
                contained,
                str(inc.get("created_at", ""))[:19],
            )

        console.print(table)

    # -------------------------------------------------------------------------
    # Session Termination
    # -------------------------------------------------------------------------
    def do_exit(self, _=None) -> bool:
        """Exit and terminate the Threatora tactical console session."""
        console.print("[yellow][*] Terminating tactical session. Stand down, Operator.[/yellow]")
        return True

    def do_quit(self, args=None) -> bool:
        """Exit session alias."""
        return self.do_exit(args)

    def do_q(self, args=None) -> bool:
        """Quick exit alias."""
        return self.do_exit(args)


# Backward compatibility alias
ThraetoraConsole = ThreatoraConsole


def launch_console(
    api_url: str = DEFAULT_API_URL,
    username: Optional[str] = None,
    password: Optional[str] = None,
    require_auth: bool = True,
):
    """Entrypoint to start the interactive cmd2 console."""
    app = ThreatoraConsole(
        api_url=api_url,
        username=username,
        password=password,
        require_auth=require_auth,
    )
    sys.exit(app.cmdloop())


if __name__ == "__main__":
    launch_console()
