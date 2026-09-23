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
      // 1. Check for active telemetry state from sessionStorage / localStorage or /api/telemetry/active
      let activeTelemetry = null;
      try {
        const stored = sessionStorage.getItem('threatora_active_telemetry') || localStorage.getItem('threatora_active_telemetry');
        if (stored) activeTelemetry = JSON.parse(stored);
      } catch (e) {}

      if (!activeTelemetry) {
        try {
          const atRes = await fetch('/api/telemetry/active');
          if (atRes.ok) {
            const atData = await atRes.json();
            if (atData && atData.status === 'success') activeTelemetry = atData;
          }
        } catch (e) {}
      }

      // Check attack / benign status:
      // CASE A: IF BENIGN (is_attack === false or defcon_level === 5 or peak_threat_score < 30)
      // CASE B: IF ATTACK (is_attack === true or peak_threat_score >= 65)
      let isAttack = false;
      if (activeTelemetry) {
        if (activeTelemetry.is_attack !== undefined) {
          isAttack = Boolean(activeTelemetry.is_attack);
        } else {
          const peakThreat = (activeTelemetry.kpis?.peak_risk_pct !== undefined ? activeTelemetry.kpis.peak_risk_pct :
                             (activeTelemetry.summary?.peak_risk !== undefined ? activeTelemetry.summary.peak_risk * 100 :
                             (activeTelemetry.peak_threat !== undefined ? activeTelemetry.peak_threat : 0.0)));
          const defcon = activeTelemetry.kpis?.defcon !== undefined ? activeTelemetry.kpis.defcon :
                         (activeTelemetry.summary?.defcon !== undefined ? activeTelemetry.summary.defcon : 5);
          isAttack = (peakThreat >= 65.0 && defcon < 5);
        }
      }
      const isBenign = !isAttack;

      let rawNodes = [];
      let rawLinks = [];
      let captureFile = 'Active Telemetry';

      if (activeTelemetry && activeTelemetry.topology_graph && activeTelemetry.topology_graph.nodes && activeTelemetry.topology_graph.nodes.length > 0) {
        rawNodes = activeTelemetry.topology_graph.nodes;
        rawLinks = activeTelemetry.topology_graph.edges || activeTelemetry.topology_graph.links || [];
        captureFile = activeTelemetry.file_label || activeTelemetry.meta?.file_name || 'Active Telemetry';
      } else if (activeTelemetry && activeTelemetry.meta && activeTelemetry.meta.topology && activeTelemetry.meta.topology.nodes && activeTelemetry.meta.topology.nodes.length > 0) {
        rawNodes = activeTelemetry.meta.topology.nodes;
        rawLinks = activeTelemetry.meta.topology.links || activeTelemetry.meta.topology.edges || [];
        captureFile = activeTelemetry.file_label || activeTelemetry.meta?.file_name || 'Active Telemetry';
      } else {
        const res = await fetch('/api/v1/topology');
        if (res.ok) {
          const data = await res.json();
          if (data.status === 'success' && data.nodes && data.nodes.length > 0) {
            rawNodes = data.nodes;
            rawLinks = data.links || data.edges || [];
            captureFile = data.capture_file || 'Active Telemetry';
          }
        }
      }

      if (rawNodes.length > 0) {
        let nodes = rawNodes;
        let links = rawLinks;

        if (isBenign) {
          // CASE A: IF BENIGN:
          // Render full active network graph from the uploaded session
          // All nodes MUST be styled as HEALTHY (#00E676 / green, risk: 0.0%, no red alert badges)
          // Communication routes MUST be rendered as normal ambient flows (#00E5FF or #64748B, normal particle speed, NO red lines)
          nodes = nodes.map(n => ({
            ...n,
            status: n.status === 'ISOLATED' ? 'ISOLATED' : 'HEALTHY',
            is_compromised: false,
            is_adversary: false,
            risk_score: 0.0,
            stage_name: 'Benign',
            technique: 'Normal Baseline'
          }));
          links = links.map(l => ({
            ...l,
            threat: 'normal',
            is_attack_route: false,
          }));
        } else {
          // CASE B: IF ATTACK:
          // Check for any isolated hosts in activeTelemetry.isolated_hosts or status
          const isolatedList = activeTelemetry?.isolated_hosts || [];
          nodes = nodes.map(n => {
            const isIso = isolatedList.includes(n.ip) || isolatedList.includes(n.id) || n.status === 'ISOLATED' || n.is_isolated;
            if (isIso) {
              return {
                ...n,
                status: 'ISOLATED',
                is_isolated: true,
                is_compromised: false,
                risk_score: 0.0,
                stage_name: 'Air-gapped',
                technique: 'Isolated'
              };
            }
            return n;
          });

          // Any link connected to an isolated node MUST be normal/neutralized
          const isoSet = new Set(nodes.filter(n => n.status === 'ISOLATED' || n.is_isolated).map(n => n.ip || n.id));
          links = links.map(l => {
            const s = String(l.source?.ip || l.source);
            const t = String(l.target?.ip || l.target);
            if (isoSet.has(s) || isoSet.has(t)) {
              return {
                ...l,
                threat: 'normal',
                is_attack_route: false,
              };
            }
            return l;
          });
        }

        this.processTopologyData(nodes, links);

        // Top-left HUD badge and Overlay
        const subnets = Array.from(new Set(nodes.map(n => n.subnet).filter(Boolean)));
        const gws = nodes.filter(n => n.type === 'gateway').map(n => n.ip);
        const compromised = nodes.filter(n => (n.status === 'COMPROMISED' || n.is_compromised) && n.status !== 'ISOLATED' && !n.is_isolated);

        const hudSub = document.getElementById('hudSubnet');
        const hudGw = document.getElementById('hudGateway');
        const hudCount = document.getElementById('hudNodesCount');
        const topoDesc = document.getElementById('topoSubnetDesc');

        if (hudSub) hudSub.innerHTML = `<strong>SUBNET:</strong> ${subnets.join(', ') || 'Enterprise LAN'}`;
        if (hudGw) hudGw.innerHTML = `<strong>PERIMETER GATEWAY:</strong> ${gws.join(', ') || 'Internal Router'}`;
        if (hudCount) {
          if (isBenign || compromised.length === 0) {
            hudCount.innerHTML = `<strong>ACTIVE NODES:</strong> ${nodes.length} Endpoints // 0 Compromised // Normal Infrastructure Traffic`;
          } else {
            hudCount.innerHTML = `<strong>ACTIVE NODES:</strong> ${nodes.length} Endpoints // ${compromised.length} Compromised // Infiltration Route Detected`;
          }
        }
        if (topoDesc) {
          if (isBenign || compromised.length === 0) {
            topoDesc.innerText = `Visualizing active capture '${captureFile}': ${nodes.length} endpoints operating within normal baseline parameters. Zero anomalous intrusions detected.`;
          } else {
            topoDesc.innerText = `Visualizing active capture '${captureFile}': ${nodes.length} endpoints, ${links.length} communication links, ${compromised.length} threat nodes.`;
          }
        }

        // Right-side inspector behavior
        if (isBenign) {
          if (this.nodes.length > 0) {
            this.selectNode(this.nodes[0]);
          }
        } else {
          // Auto-select the active compromised node, or isolated node, or first node
          const compNode = this.nodes.find(n => (n.is_compromised || n.status === 'COMPROMISED') && n.status !== 'ISOLATED' && !n.is_isolated) ||
                           this.nodes.find(n => n.status === 'ISOLATED' || n.is_isolated) ||
                           this.nodes[0];
          if (compNode) {
            this.selectNode(compNode);
          }
        }
      } else {
        this.loadFallbackTopology(isBenign);
      }
    } catch (err) {
      console.warn("Could not load topology data, using baseline topology:", err);
      this.loadFallbackTopology(false);
    }
  }

  loadFallbackTopology(isBenign = false) {
    let mockNodes = [
      { id: "147.32.84.165", ip: "147.32.84.165", hostname: "EDGE-GATEWAY-01", type: "gateway", status: "HEALTHY", risk_score: 0.05, stage_name: "Benign", criticality: "MISSION_CRITICAL" },
      { id: "192.168.1.10", ip: "192.168.1.10", hostname: "DC-CORP-AD01", type: "domain_controller", status: "HEALTHY", risk_score: 0.08, stage_name: "Benign", criticality: "HIGH" },
      { id: "192.168.1.5", ip: "192.168.1.5", hostname: "SRV-DATABASE-01", type: "database", status: "HEALTHY", risk_score: 0.04, stage_name: "Benign", criticality: "HIGH" },
      { id: "192.168.1.105", ip: "192.168.1.105", hostname: "DEV-WORKSTATION-05", type: "workstation", status: isBenign ? "HEALTHY" : "COMPROMISED", is_compromised: !isBenign, risk_score: isBenign ? 0.0 : 0.88, stage_name: isBenign ? "Benign" : "Initial Access", technique: isBenign ? "Normal Baseline" : "TA0001 Exploit Public-Facing App", criticality: "MEDIUM" },
    ];

    if (!isBenign) {
      mockNodes.push({ id: "198.51.100.42", ip: "198.51.100.42", hostname: "EXT-C2-ADVERSARY", type: "external", status: "THREAT_ACTOR", is_adversary: true, risk_score: 0.99, stage_name: "Command & Control", technique: "TA0011 C2 Beaconing", criticality: "ADVERSARY" });
    }

    let mockLinks = [
      { source: "147.32.84.165", target: "192.168.1.10", threat: "normal", is_attack_route: false },
      { source: "147.32.84.165", target: "192.168.1.5", threat: "normal", is_attack_route: false },
      { source: "147.32.84.165", target: "192.168.1.105", threat: isBenign ? "normal" : "critical", is_attack_route: !isBenign },
    ];

    if (!isBenign) {
      mockLinks.push({ source: "192.168.1.105", target: "198.51.100.42", threat: "critical", is_attack_route: true });
    }

    this.processTopologyData(mockNodes, mockLinks);

    const hudCount = document.getElementById('hudNodesCount');
    if (hudCount) {
      if (isBenign) {
        hudCount.innerHTML = `<strong>ACTIVE NODES:</strong> ${mockNodes.length} Endpoints // 0 Compromised // Normal Infrastructure Traffic`;
      } else {
        hudCount.innerHTML = `<strong>ACTIVE NODES:</strong> ${mockNodes.length} Endpoints // 1 Compromised // Infiltration Route Detected`;
      }
    }
    const victim = mockNodes.find(n => n.is_compromised || n.status === 'COMPROMISED') || mockNodes[0];
    if (victim) this.selectNode(victim);
  }

  drawRoundedRect(x, y, width, height, radius) {
    this.ctx.beginPath();
    this.ctx.moveTo(x + radius, y);
    this.ctx.lineTo(x + width - radius, y);
    this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    this.ctx.lineTo(x + width, y + height - radius);
    this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    this.ctx.lineTo(x + radius, y + height);
    this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    this.ctx.lineTo(x, y + radius);
    this.ctx.quadraticCurveTo(x, y, x + radius, y);
    this.ctx.closePath();
  }

  processTopologyData(rawNodes, rawLinks) {
    const cx = this.displayWidth / 2;
    const cy = this.displayHeight / 2;

    // 1. Consolidate devices with multiple network identities into single consolidated icons
    const consolidatedMap = new Map();
    (rawNodes || []).forEach(n => {
      const parts = String(n.ip || n.id).split(/[,; ]+/).filter(Boolean);
      const primaryIp = parts[0] || n.id;
      const isGw = n.type === 'gateway' || (n.hostname && n.hostname.toUpperCase().includes('GATEWAY'));
      const key = isGw ? 'GATEWAY_PRIMARY' : primaryIp;

      if (!consolidatedMap.has(key)) {
        consolidatedMap.set(key, {
          ...n,
          id: n.id || primaryIp,
          ip: primaryIp,
          ips: parts.length > 0 ? parts : [primaryIp],
          type: isGw ? 'gateway' : (n.type || 'workstation'),
        });
      } else {
        const existing = consolidatedMap.get(key);
        parts.forEach(p => {
          if (!existing.ips.includes(p)) existing.ips.push(p);
        });
        if ((n.risk_score || 0) > (existing.risk_score || 0)) {
          existing.risk_score = n.risk_score;
          existing.status = n.status;
          existing.stage_name = n.stage_name;
          existing.technique = n.technique;
        }
      }
    });

    const uniqueNodes = Array.from(consolidatedMap.values());

    // 2. Separate into Gateways, Externals, and Internal Endpoints
    const gateways = uniqueNodes.filter(n => n.type === 'gateway');
    const externals = uniqueNodes.filter(n => n.type === 'external');
    const endpoints = uniqueNodes.filter(n => n.type !== 'gateway' && n.type !== 'external');

    // Place Gateways top-center with distinct boundary separation
    gateways.forEach((gw, idx) => {
      const offset = (idx - (gateways.length - 1) / 2) * 110;
      gw.x = cx + offset;
      gw.y = cy - 90;
      gw.radius = 24;
      gw.pulse = Math.random() * Math.PI * 2;
    });

    // Place External nodes in the upper-right WAN zone
    externals.forEach((ext, idx) => {
      const offset = (idx - (externals.length - 1) / 2) * 100;
      ext.x = cx + 240 + offset;
      ext.y = cy - 130;
      ext.radius = 22;
      ext.pulse = Math.random() * Math.PI * 2;
    });

    // Place Endpoints on a spacious, comfortable circular orbit
    const nEndpoints = Math.max(endpoints.length, 1);
    const orbitRadius = Math.max(165, Math.min(260, 42 * Math.sqrt(nEndpoints)));
    endpoints.forEach((ep, i) => {
      const angle = (2 * Math.PI * i) / nEndpoints - Math.PI / 2 + 0.2;
      ep.x = cx + Math.cos(angle) * orbitRadius;
      ep.y = (cy + 25) + Math.sin(angle) * (orbitRadius * 0.82);
      ep.radius = 18;
      ep.pulse = Math.random() * Math.PI * 2;
    });

    this.nodes = [...gateways, ...externals, ...endpoints];

    // 3. Force relaxation to ensure strict boundary separation and prevent any collision
    const minDist = 82; // Minimum distance between any two node centers
    for (let iter = 0; iter < 80; iter++) {
      for (let i = 0; i < this.nodes.length; i++) {
        for (let j = i + 1; j < this.nodes.length; j++) {
          const a = this.nodes[i];
          const b = this.nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.hypot(dx, dy) || 1;
          if (dist < minDist) {
            const overlap = (minDist - dist) * 0.5;
            const fx = (dx / dist) * overlap;
            const fy = (dy / dist) * overlap;
            b.x += fx;
            b.y += fy;
            a.x -= fx;
            a.y -= fy;
          }
        }
      }
    }

    // Clamp coordinates comfortably inside canvas
    const pad = 45;
    this.nodes.forEach(n => {
      n.x = Math.max(pad, Math.min(this.displayWidth - pad, n.x));
      n.y = Math.max(pad, Math.min(this.displayHeight - pad, n.y));
    });

    // 4. Map links to resolved node IDs or consolidated primary IP
    const nodeMap = new Map();
    this.nodes.forEach(n => {
      nodeMap.set(n.id, n);
      nodeMap.set(n.ip, n);
      if (n.ips) {
        n.ips.forEach(ip => nodeMap.set(ip, n));
      }
      if (n.type === 'gateway') {
        nodeMap.set('GATEWAY', n);
        nodeMap.set('147.32.84.165', n);
      }
    });

    this.links = (rawLinks || []).map(l => {
      const srcNode = nodeMap.get(l.source);
      const tgtNode = nodeMap.get(l.target);
      const isAttRoute = Boolean(l.is_attack_route || l.threat === 'critical');
      return {
        sourceNode: srcNode,
        targetNode: tgtNode,
        threat: isAttRoute ? 'critical' : (l.threat || 'normal'),
        is_attack_route: isAttRoute,
        port: l.port || 80,
        proto: l.proto || l.protocol || 'TCP'
      };
    }).filter(l => l.sourceNode && l.targetNode && l.sourceNode !== l.targetNode);

    // Initialize flowing particles
    this.particles = [];
    for (let i = 0; i < 28; i++) {
      const link = this.links[Math.floor(Math.random() * this.links.length)];
      if (link) {
        const isAttackRoute = link.threat === 'critical' || link.is_attack_route;
        this.particles.push({
          link: link,
          progress: Math.random(),
          speed: isAttackRoute ? (0.018 + Math.random() * 0.014) : (0.004 + Math.random() * 0.006)
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
          if (hover) {
            this.canvas.title = `${hover.hostname || hover.ip} (${hover.ip}) - Subnet: ${hover.subnet || 'N/A'}`;
          } else {
            this.canvas.title = '';
          }
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
    document.getElementById('inspHostName').innerText = node.hostname || node.label || node.ip;
    document.getElementById('inspIp').innerText = node.ip;
    document.getElementById('inspSubnet').innerText = node.subnet || '192.168.1.0/24';
    document.getElementById('inspCrit').innerText = node.criticality || 'MEDIUM';

    const riskEl = document.getElementById('inspRisk');
    const stageEl = document.getElementById('inspStage');
    const statusBadge = document.getElementById('inspStatus');

    const isIso = Boolean(node.status === 'ISOLATED' || node.is_isolated);
    const isComp = Boolean((node.status === 'COMPROMISED' || node.is_compromised) && !isIso);
    const isAdv = Boolean((node.status === 'THREAT_ACTOR' || node.is_adversary) && !isIso);

    if (isIso) {
      statusBadge.className = 'badge-isolated';
      statusBadge.innerText = 'ISOLATED';
      statusBadge.style.color = '#A0AEC0';
      statusBadge.style.borderColor = '#4D4D4D';
      statusBadge.style.background = 'rgba(77, 77, 77, 0.3)';
      riskEl.innerText = '0.0%';
      riskEl.style.color = 'var(--text-muted)';
      stageEl.innerText = 'Air-gapped';
    } else if (isComp) {
      statusBadge.className = 'badge-threat';
      statusBadge.innerText = 'COMPROMISED';
      statusBadge.style.color = '#FF5252';
      statusBadge.style.borderColor = 'rgba(255, 82, 82, 0.4)';
      statusBadge.style.background = 'rgba(255, 82, 82, 0.15)';
      riskEl.innerText = ((node.risk_score || 0.85) * 100).toFixed(1) + '%';
      riskEl.style.color = '#FF5252';
      stageEl.innerText = node.stage_name || 'Active Infiltration';
    } else if (isAdv) {
      statusBadge.className = 'badge-threat';
      statusBadge.innerText = 'THREAT ACTOR';
      statusBadge.style.color = '#FF7043';
      statusBadge.style.borderColor = 'rgba(255, 112, 67, 0.4)';
      statusBadge.style.background = 'rgba(255, 112, 67, 0.15)';
      riskEl.innerText = ((node.risk_score || 0.95) * 100).toFixed(1) + '%';
      riskEl.style.color = '#FF7043';
      stageEl.innerText = 'Adversary Recon';
    } else {
      statusBadge.className = 'badge-benign';
      statusBadge.innerText = 'Status: HEALTHY // No Compromise';
      statusBadge.style.color = '#00E676';
      statusBadge.style.borderColor = 'rgba(0, 230, 118, 0.4)';
      statusBadge.style.background = 'rgba(0, 230, 118, 0.1)';
      riskEl.innerText = '0.0%';
      riskEl.style.color = '#00E676';
      stageEl.innerText = 'Normal Baseline';
    }

    // Direct 1-Click isolate from inspector
    const btnIsolate = document.getElementById('inspBtnIsolate');
    if (btnIsolate) {
      if (isIso) {
        btnIsolate.disabled = true;
        btnIsolate.className = 'btn-cyber';
        btnIsolate.style.borderColor = '#00E676';
        btnIsolate.style.color = '#00E676';
        btnIsolate.innerText = "✔ Host Isolated";
      } else {
        btnIsolate.disabled = false;
        btnIsolate.className = 'btn-cyber btn-cyber-danger';
        btnIsolate.style.borderColor = '';
        btnIsolate.style.color = '';
        btnIsolate.innerText = `Quarantine ${node.ip}`;
        btnIsolate.onclick = async () => {
          btnIsolate.disabled = true;
          btnIsolate.innerText = "Isolating...";
          if (window.triggerDirectIsolation) {
            await window.triggerDirectIsolation(node.ip);
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

      const isSrcIso = Boolean(l.sourceNode.status === 'ISOLATED' || l.sourceNode.is_isolated);
      const isTgtIso = Boolean(l.targetNode.status === 'ISOLATED' || l.targetNode.is_isolated);
      const isIsoRoute = isSrcIso || isTgtIso;

      const isAttack = (l.is_attack_route || l.threat === 'critical') && !isIsoRoute;

      if (isAttack) {
        this.ctx.strokeStyle = '#FF5252';
        this.ctx.lineWidth = 2.4;
      } else if (l.threat === 'medium' && !isIsoRoute) {
        this.ctx.strokeStyle = 'rgba(255, 183, 3, 0.45)';
        this.ctx.lineWidth = 1.6;
      } else {
        this.ctx.strokeStyle = 'rgba(0, 229, 255, 0.25)';
        this.ctx.lineWidth = 1.2;
      }
      this.ctx.stroke();
    });

    // 2. Draw Flowing Particles
    this.particles.forEach(p => {
      const isSrcIso = Boolean(p.link.sourceNode.status === 'ISOLATED' || p.link.sourceNode.is_isolated);
      const isTgtIso = Boolean(p.link.targetNode.status === 'ISOLATED' || p.link.targetNode.is_isolated);
      const isIsoRoute = isSrcIso || isTgtIso;

      const isAttackRoute = (p.link.threat === 'critical' || p.link.is_attack_route) && !isIsoRoute;

      p.progress += (isAttackRoute ? p.speed : 0.005);
      if (p.progress > 1) p.progress = 0;

      const s = p.link.sourceNode;
      const t = p.link.targetNode;
      const px = s.x + (t.x - s.x) * p.progress;
      const py = s.y + (t.y - s.y) * p.progress;

      this.ctx.beginPath();
      this.ctx.arc(px, py, isAttackRoute ? 3.0 : 2.5, 0, Math.PI * 2);
      this.ctx.fillStyle = isAttackRoute ? '#FF1744' : '#00E676';
      this.ctx.shadowBlur = isAttackRoute ? 8 : 4;
      this.ctx.shadowColor = isAttackRoute ? '#FF1744' : 'rgba(0, 230, 118, 0.5)';
      this.ctx.fill();
      this.ctx.shadowBlur = 0;
    });

    // 3. Draw Nodes
    this.nodes.forEach(n => {
      n.pulse += 0.04;
      const isIsolated = Boolean(n.status === 'ISOLATED' || n.is_isolated);
      const isThreat = Boolean((n.status === 'COMPROMISED' || n.is_compromised) && !isIsolated);
      const isAdversary = Boolean((n.status === 'THREAT_ACTOR' || n.is_adversary) && !isIsolated);
      const isSelected = this.selectedNode === n;

      // Glow Halo
      const pulseRadius = n.radius + 4 + Math.sin(n.pulse) * 3;
      this.ctx.beginPath();
      this.ctx.arc(n.x, n.y, pulseRadius, 0, Math.PI * 2);
      if (isThreat) {
        this.ctx.fillStyle = 'rgba(255, 82, 82, 0.25)';
      } else if (isAdversary) {
        this.ctx.fillStyle = 'rgba(255, 112, 67, 0.25)';
      } else if (isIsolated) {
        this.ctx.fillStyle = 'rgba(77, 77, 77, 0.18)';
      } else {
        this.ctx.fillStyle = 'rgba(0, 230, 118, 0.22)';
      }
      this.ctx.fill();

      // Node Body Circle
      this.ctx.beginPath();
      this.ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = isIsolated ? '#242424' : '#171717';
      this.ctx.fill();

      // Node Border
      this.ctx.lineWidth = isSelected ? 3 : 2;
      if (isThreat) {
        this.ctx.strokeStyle = '#FF5252';
      } else if (isAdversary) {
        this.ctx.strokeStyle = '#FF7043';
      } else if (isIsolated) {
        this.ctx.strokeStyle = '#4D4D4D';
      } else {
        this.ctx.strokeStyle = '#00E676';
      }
      if (isSelected) this.ctx.strokeStyle = '#FFFFFF';
      this.ctx.stroke();

      // Node Icon / Type Glyph
      this.ctx.fillStyle = isThreat ? '#FF5252' : (isAdversary ? '#FF7043' : (isIsolated ? '#A0AEC0' : '#00E676'));
      this.ctx.font = '10px "Plus Jakarta Sans", sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';

      let glyph = '💻';
      if (n.type === 'gateway') glyph = '🌐';
      if (n.type === 'domain_controller') glyph = '🛡️';
      if (n.type === 'database') glyph = '🗄️';
      if (n.type === 'external' || isAdversary) glyph = '⚠️';
      if (isThreat) glyph = '🚨';
      if (isIsolated) glyph = '🔒';

      this.ctx.fillText(glyph, n.x, n.y);

      // Consolidated single-line label with subtle background badge
      let label = n.ip;
      if (n.type === 'gateway') {
        const count = (n.ips && n.ips.length > 1) ? n.ips.length : 1;
        label = count > 1 ? `GATEWAY (${n.ips[0]}) +${count - 1} subnets` : `GATEWAY (${n.ip})`;
      } else if (n.type === 'external' || isAdversary) {
        label = `EXT-C2 (${n.ip})`;
      } else {
        const shortHost = (n.hostname && n.hostname.length > 14) ? n.hostname.substring(0, 13) + '…' : (n.hostname || 'NODE');
        label = `${shortHost} · ${n.ip}`;
      }

      this.ctx.font = '10px "JetBrains Mono", monospace';
      const textMetrics = this.ctx.measureText(label);
      const badgeW = textMetrics.width + 16;
      const badgeH = 18;
      const badgeX = n.x - (badgeW / 2);
      const badgeY = n.y + n.radius + 6;

      // Subtle background pill badge preventing text collisions
      this.ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
      this.ctx.strokeStyle = isThreat ? 'rgba(255, 82, 82, 0.6)' : (isAdversary ? 'rgba(255, 112, 67, 0.6)' : (isSelected ? '#FFFFFF' : (isIsolated ? 'rgba(150, 150, 150, 0.3)' : 'rgba(255, 255, 255, 0.14)')));
      this.ctx.lineWidth = 1;
      this.drawRoundedRect(badgeX, badgeY, badgeW, badgeH, 4);
      this.ctx.fill();
      this.ctx.stroke();

      // Clean single-line text label
      this.ctx.fillStyle = isThreat ? '#FF5252' : (isAdversary ? '#FF7043' : (isIsolated ? '#A0AEC0' : '#F1F5F9'));
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(label, n.x, badgeY + (badgeH / 2));
    });

    this.animFrameId = requestAnimationFrame(() => this.animate());
  }

  isolateNode(targetIp) {
    if (!targetIp) return;
    const target = this.nodes.find(n => n.ip === targetIp || n.id === targetIp);
    if (target) {
      target.status = 'ISOLATED';
      target.is_isolated = true;
      target.is_compromised = false;
      target.risk_score = 0.0;
      target.stage_name = 'Air-gapped';
      target.technique = 'Isolated';
    }
    this.links.forEach(l => {
      const s = String(l.sourceNode?.ip || l.sourceNode?.id || l.source);
      const t = String(l.targetNode?.ip || l.targetNode?.id || l.target);
      if (s === targetIp || t === targetIp) {
        l.threat = 'normal';
        l.is_attack_route = false;
      }
    });
    this.particles.forEach(p => {
      const s = String(p.link?.sourceNode?.ip || p.link?.sourceNode?.id);
      const t = String(p.link?.targetNode?.ip || p.link?.targetNode?.id);
      if (s === targetIp || t === targetIp) {
        p.speed = 0.005;
      }
    });

    const compromised = this.nodes.filter(n => (n.status === 'COMPROMISED' || n.is_compromised) && n.status !== 'ISOLATED' && !n.is_isolated);
    const hudCount = document.getElementById('hudNodesCount');
    if (hudCount) {
      if (compromised.length === 0) {
        hudCount.innerHTML = `<strong>ACTIVE NODES:</strong> ${this.nodes.length} Endpoints // 0 Compromised // Normal Infrastructure Traffic`;
      } else {
        hudCount.innerHTML = `<strong>ACTIVE NODES:</strong> ${this.nodes.length} Endpoints // ${compromised.length} Compromised // Infiltration Route Detected`;
      }
    }
    if (this.selectedNode && (this.selectedNode.ip === targetIp || this.selectedNode.id === targetIp)) {
      this.selectNode(target || this.selectedNode);
    }
  }
}

window.CyberNetworkTopology = CyberNetworkTopology;
