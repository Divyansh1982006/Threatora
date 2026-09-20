/**
 * Threatora Network Topology Interactive Force Graph
 * Real-Time Visual Defense & Compromise Progression Radar
 * 100% English Interface
 */

class CyberNetworkTopology {
  constructor(canvasId, inspectorId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.inspector = document.getElementById(inspectorId);

    this.nodes = [];
    this.links = [];
    this.particles = [];
    this.selectedNode = null;
    this.hoveredNode = null;

    this.isDragging = false;
    this.draggedNode = null;
    this.offset = { x: 0, y: 0 };
    this.scale = 1;

    this.animFrameId = null;

    this.initCanvasSize();
    this.bindEvents();
    this.loadTopology();
  }

  initCanvasSize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.displayWidth = rect.width;
    this.displayHeight = rect.height;
  }

  async loadTopology() {
    try {
      const res = await fetch('/api/v1/topology');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.status === 'success' && data.nodes && data.nodes.length > 0) {
        this.processTopologyData(data.nodes, data.links);

        // Dynamically update HUD Overlay and Header
        const subnets = Array.from(new Set(data.nodes.map(n => n.subnet).filter(Boolean)));
        const gws = data.nodes.filter(n => n.type === 'gateway').map(n => n.ip);
        const compromised = data.nodes.filter(n => n.status === 'COMPROMISED' || n.status === 'THREAT_ACTOR');

        const hudSub = document.getElementById('hudSubnet');
        const hudGw = document.getElementById('hudGateway');
        const hudCount = document.getElementById('hudNodesCount');
        const topoDesc = document.getElementById('topoSubnetDesc');

        if (hudSub) hudSub.innerHTML = `<strong>SUBNET:</strong> ${subnets.join(', ') || 'Enterprise LAN'}`;
        if (hudGw) hudGw.innerHTML = `<strong>PERIMETER GATEWAY:</strong> ${gws.join(', ') || 'Internal Router'}`;
        if (hudCount) hudCount.innerHTML = `<strong>ACTIVE NODES:</strong> ${data.nodes.length} Endpoints // ${compromised.length} Compromised // ${data.links.length} Active Channels`;
        if (topoDesc) topoDesc.innerText = `Visualizing active capture '${data.capture_file || 'Telemetry'}': ${data.nodes.length} endpoints, ${data.links.length} communication links, ${compromised.length} threat nodes.`;
      } else {
        this.loadFallbackTopology();
      }
    } catch (err) {
      console.warn("Could not load topology data, using baseline topology:", err);
      this.loadFallbackTopology();
    }
  }

  loadFallbackTopology() {
    const mockNodes = [
      { id: "147.32.84.165", ip: "147.32.84.165", hostname: "EDGE-GATEWAY-01", type: "gateway", status: "HEALTHY", risk_score: 0.14, stage_name: "Benign", criticality: "MISSION_CRITICAL" },
      { id: "192.168.1.10", ip: "192.168.1.10", hostname: "DC-CORP-AD01", type: "domain_controller", status: "HEALTHY", risk_score: 0.12, stage_name: "Benign", criticality: "HIGH" },
      { id: "192.168.1.5", ip: "192.168.1.5", hostname: "SRV-DATABASE-01", type: "database", status: "HEALTHY", risk_score: 0.08, stage_name: "Benign", criticality: "HIGH" },
      { id: "192.168.1.105", ip: "192.168.1.105", hostname: "DEV-WORKSTATION-05", type: "workstation", status: "COMPROMISED", risk_score: 0.75, stage_name: "Lateral Movement", criticality: "MEDIUM" },
      { id: "198.51.100.42", ip: "198.51.100.42", hostname: "EXT-C2-ADVERSARY", type: "external", status: "THREAT_ACTOR", risk_score: 0.99, stage_name: "Command & Control", criticality: "ADVERSARY" }
    ];

    const mockLinks = [
      { source: "147.32.84.165", target: "192.168.1.10", threat: "normal" },
      { source: "147.32.84.165", target: "192.168.1.5", threat: "normal" },
      { source: "147.32.84.165", target: "192.168.1.105", threat: "medium" },
      { source: "192.168.1.105", target: "192.168.1.10", threat: "critical" },
      { source: "192.168.1.105", target: "192.168.1.5", threat: "critical" },
      { source: "192.168.1.105", target: "198.51.100.42", threat: "critical" }
    ];

    this.processTopologyData(mockNodes, mockLinks);
  }

  processTopologyData(rawNodes, rawLinks) {
    const cx = this.displayWidth / 2;
    const cy = this.displayHeight / 2;

    const angleStep = (Math.PI * 2) / (rawNodes.length || 1);
    this.nodes = rawNodes.map((n, i) => {
      let x = cx;
      let y = cy;

      if (n.type === 'gateway') {
        x = cx;
        y = cy - 40;
      } else if (n.type === 'external') {
        x = cx + 220;
        y = cy - 140;
      } else {
        const rad = 150;
        const ang = angleStep * i;
        x = cx + Math.cos(ang) * rad;
        y = cy + Math.sin(ang) * rad + 30;
      }

      return {
        ...n,
        x: x,
        y: y,
        radius: n.type === 'gateway' ? 24 : (n.type === 'external' ? 22 : 18),
        pulse: Math.random() * Math.PI * 2
      };
    });

    const nodeMap = new Map();
    this.nodes.forEach(n => nodeMap.set(n.id, n));

    this.links = rawLinks.map(l => {
      return {
        sourceNode: nodeMap.get(l.source),
        targetNode: nodeMap.get(l.target),
        threat: l.threat || 'normal',
        port: l.port || 80,
        proto: l.proto || 'TCP'
      };
    }).filter(l => l.sourceNode && l.targetNode);

    // Initialize flowing particles
    this.particles = [];
    for (let i = 0; i < 24; i++) {
      const link = this.links[Math.floor(Math.random() * this.links.length)];
      if (link) {
        this.particles.push({
          link: link,
          progress: Math.random(),
          speed: 0.004 + Math.random() * 0.008
        });
      }
    }

    if (!this.animFrameId) {
      this.animate();
    }
  }

  bindEvents() {
    window.addEventListener('resize', () => {
      this.initCanvasSize();
    });

    this.canvas.addEventListener('mousedown', (e) => {
      const pos = this.getMousePos(e);
      const clicked = this.findNodeAt(pos.x, pos.y);

      if (clicked) {
        this.isDragging = true;
        this.draggedNode = clicked;
        this.selectNode(clicked);
      }
    });

    window.addEventListener('mousemove', (e) => {
      const pos = this.getMousePos(e);
      if (this.isDragging && this.draggedNode) {
        this.draggedNode.x = pos.x;
        this.draggedNode.y = pos.y;
      } else {
        const hover = this.findNodeAt(pos.x, pos.y);
        if (hover !== this.hoveredNode) {
          this.hoveredNode = hover;
          this.canvas.style.cursor = hover ? 'pointer' : 'grab';
        }
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
      this.draggedNode = null;
    });

    // Close inspector button if available
    const btnClose = document.getElementById('closeNodeInspector');
    if (btnClose) {
      btnClose.addEventListener('click', () => {
        this.inspector.classList.remove('open');
        this.selectedNode = null;
      });
    }
  }

  getMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }

  findNodeAt(x, y) {
    for (const n of this.nodes) {
      const dist = Math.hypot(n.x - x, n.y - y);
      if (dist <= n.radius + 6) return n;
    }
    return null;
  }

  selectNode(node) {
    this.selectedNode = node;
    if (!this.inspector) return;

    this.inspector.classList.add('open');
    document.getElementById('inspHostName').innerText = node.hostname;
    document.getElementById('inspIp').innerText = node.ip;
    document.getElementById('inspSubnet').innerText = node.subnet || '192.168.1.0/24';
    document.getElementById('inspCrit').innerText = node.criticality;

    const riskEl = document.getElementById('inspRisk');
    const riskPct = (node.risk_score * 100).toFixed(1) + '%';
    riskEl.innerText = riskPct;
    riskEl.style.color = node.risk_score > 0.5 ? 'var(--neon-crimson)' : 'var(--neon-emerald)';

    const stageEl = document.getElementById('inspStage');
    stageEl.innerText = node.stage_name;

    const statusBadge = document.getElementById('inspStatus');
    statusBadge.className = node.status === 'ISOLATED' ? 'badge-isolated' : 
                           (node.status === 'COMPROMISED' || node.risk_score > 0.5 ? 'badge-threat' : 'badge-benign');
    statusBadge.innerText = node.status;

    // Direct 1-Click isolate from inspector
    const btnIsolate = document.getElementById('inspBtnIsolate');
    if (btnIsolate) {
      if (node.status === 'ISOLATED') {
        btnIsolate.disabled = true;
        btnIsolate.innerText = "Host Already Isolated";
      } else {
        btnIsolate.disabled = false;
        btnIsolate.innerText = `Quarantine ${node.ip}`;
        btnIsolate.onclick = () => {
          if (window.triggerDirectIsolation) {
            window.triggerDirectIsolation(node.ip);
          }
        };
      }
    }
  }

  animate() {
    this.ctx.clearRect(0, 0, this.displayWidth, this.displayHeight);

    // 1. Draw Links
    this.links.forEach(l => {
      this.ctx.beginPath();
      this.ctx.moveTo(l.sourceNode.x, l.sourceNode.y);
      this.ctx.lineTo(l.targetNode.x, l.targetNode.y);

      if (l.threat === 'critical') {
        this.ctx.strokeStyle = 'rgba(255, 0, 85, 0.45)';
        this.ctx.lineWidth = 2.2;
      } else if (l.threat === 'medium') {
        this.ctx.strokeStyle = 'rgba(255, 183, 3, 0.35)';
        this.ctx.lineWidth = 1.6;
      } else {
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
        this.ctx.lineWidth = 1.2;
      }
      this.ctx.stroke();
    });

    // 2. Draw Flowing Particles
    this.particles.forEach(p => {
      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;

      const s = p.link.sourceNode;
      const t = p.link.targetNode;
      const px = s.x + (t.x - s.x) * p.progress;
      const py = s.y + (t.y - s.y) * p.progress;

      this.ctx.beginPath();
      this.ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      this.ctx.fillStyle = p.link.threat === 'critical' ? '#F25623' : '#FFA07A';
      this.ctx.shadowBlur = 6;
      this.ctx.shadowColor = p.link.threat === 'critical' ? '#F25623' : 'rgba(242, 86, 35, 0.5)';
      this.ctx.fill();
      this.ctx.shadowBlur = 0;
    });

    // 3. Draw Nodes
    this.nodes.forEach(n => {
      n.pulse += 0.04;
      const isThreat = n.status === 'COMPROMISED' || n.risk_score >= 0.5;
      const isIsolated = n.status === 'ISOLATED';
      const isSelected = this.selectedNode === n;

      // Glow Halo
      const pulseRadius = n.radius + 4 + Math.sin(n.pulse) * 3;
      this.ctx.beginPath();
      this.ctx.arc(n.x, n.y, pulseRadius, 0, Math.PI * 2);
      if (isThreat) {
        this.ctx.fillStyle = 'rgba(242, 86, 35, 0.25)';
      } else if (isIsolated) {
        this.ctx.fillStyle = 'rgba(77, 77, 77, 0.18)';
      } else {
        this.ctx.fillStyle = 'rgba(242, 86, 35, 0.12)';
      }
      this.ctx.fill();

      // Node Body Circle
      this.ctx.beginPath();
      this.ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = isIsolated ? '#242424' : '#171717';
      this.ctx.fill();

      // Node Border
      this.ctx.lineWidth = isSelected ? 3 : 2;
      this.ctx.strokeStyle = isThreat ? '#F25623' : (isIsolated ? '#4D4D4D' : '#F25623');
      if (isSelected) this.ctx.strokeStyle = '#FFFFFF';
      this.ctx.stroke();

      // Node Icon / Type Glyph
      this.ctx.fillStyle = isThreat ? '#F25623' : (isIsolated ? '#DEDEDE' : '#FFA07A');
      this.ctx.font = '10px "Plus Jakarta Sans", sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';

      let glyph = '💻';
      if (n.type === 'gateway') glyph = '🌐';
      if (n.type === 'domain_controller') glyph = '🛡️';
      if (n.type === 'database') glyph = '🗄️';
      if (n.type === 'external') glyph = '⚠️';
      if (isIsolated) glyph = '🔒';

      this.ctx.fillText(glyph, n.x, n.y);

      // Node Label Text
      this.ctx.fillStyle = '#f1f5f9';
      this.ctx.font = '10px "JetBrains Mono", monospace';
      this.ctx.fillText(n.ip, n.x, n.y + n.radius + 12);

      this.ctx.fillStyle = isThreat ? '#ff3366' : '#94a3b8';
      this.ctx.font = '9px "Plus Jakarta Sans", sans-serif';
      this.ctx.fillText(n.hostname, n.x, n.y + n.radius + 24);
    });

    this.animFrameId = requestAnimationFrame(() => this.animate());
  }
}

window.CyberNetworkTopology = CyberNetworkTopology;
