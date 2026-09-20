/**
 * Threatora 3D Live Network Topology Radar Visualizer
 * NTRO Defense Problem Statement 26153 // Interactive Canvas 3D Engine
 */

class ThreatoraRadar {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext('2d');
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;

    // Camera / Rotation angles
    this.rotX = 0.45; // Tilt
    this.rotY = 0.0;  // Azimuth
    this.zoom = 1.0;
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    // Sweep scanline angle
    this.sweepAngle = 0;
    this.sweepSpeed = 0.015;

    // Burst mode animation
    this.burstIntensity = 0;

    // Topology Nodes
    this.nodes = [
      { id: 'gw-01', name: 'PERIMETER-GW-01', ip: '10.0.0.1', type: 'gateway', r: 180, theta: 0.2, speed: 0.002, color: '#06b6d4', size: 7, risk: 0.12 },
      { id: 'dev-05', name: 'DEV-WORKSTATION-05', ip: '192.168.1.105', type: 'compromised', r: 130, theta: 1.8, speed: -0.003, color: '#ef4444', size: 9, risk: 0.748 },
      { id: 'prod-01', name: 'PROD-API-SRV-01', ip: '192.168.2.10', type: 'server', r: 150, theta: 3.4, speed: 0.0015, color: '#10b981', size: 7, risk: 0.25 },
      { id: 'db-shard', name: 'DB-PRIMARY-SHARD', ip: '192.168.3.50', type: 'database', r: 90, theta: 4.6, speed: -0.0025, color: '#f59e0b', size: 8, risk: 0.45 },
      { id: 'work-12', name: 'OFFICE-HOST-12', ip: '192.168.1.112', type: 'workstation', r: 160, theta: 5.5, speed: 0.001, color: '#10b981', size: 6, risk: 0.08 },
    ];

    // Edges with animated packet pulses
    this.edges = [
      { from: 'gw-01', to: 'dev-05', threat: 'suspicious', pulses: [] },
      { from: 'dev-05', to: 'prod-01', threat: 'attack', pulses: [] },
      { from: 'prod-01', to: 'db-shard', threat: 'normal', pulses: [] },
      { from: 'gw-01', to: 'work-12', threat: 'normal', pulses: [] },
    ];

    this.initEvents();
    this.resize();
    this.animate();
  }

  initEvents() {
    window.addEventListener('resize', () => this.resize());

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastMouseX;
      const dy = e.clientY - this.lastMouseY;
      this.rotY += dx * 0.008;
      this.rotX += dy * 0.008;
      // Clamp vertical tilt
      this.rotX = Math.max(0.1, Math.min(1.2, this.rotX));
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom -= e.deltaY * 0.001;
      this.zoom = Math.max(0.6, Math.min(1.8, this.zoom));
    }, { passive: false });
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = this.width * window.devicePixelRatio;
    this.canvas.height = this.height * window.devicePixelRatio;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  project(x, y, z) {
    // 3D rotation around Y and X
    const cosY = Math.cos(this.rotY);
    const sinY = Math.sin(this.rotY);
    const x1 = x * cosY - z * sinY;
    const z1 = z * cosY + x * sinY;

    const cosX = Math.cos(this.rotX);
    const sinX = Math.sin(this.rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = z1 * cosX + y * sinX;

    const fov = 400;
    const scale = (fov / (fov + z2)) * this.zoom;
    const projX = this.width / 2 + x1 * scale;
    const projY = this.height / 2 + y2 * scale;

    return { x: projX, y: projY, z: z2, scale };
  }

  triggerBurst() {
    this.burstIntensity = 1.0;
    // Spawn rapid pulses on all edges
    this.edges.forEach(e => {
      e.pulses.push({ progress: 0, speed: 0.035 + Math.random() * 0.02 });
      e.pulses.push({ progress: 0.3, speed: 0.035 + Math.random() * 0.02 });
    });
  }

  update() {
    this.sweepAngle += this.sweepSpeed;
    if (this.burstIntensity > 0) {
      this.burstIntensity = Math.max(0, this.burstIntensity - 0.02);
    }

    // Update nodes orbit
    this.nodes.forEach(n => {
      n.theta += n.speed * (1 + this.burstIntensity * 2);
    });

    // Spawn periodic packet pulses
    if (Math.random() < 0.04 + this.burstIntensity * 0.1) {
      const edge = this.edges[Math.floor(Math.random() * this.edges.length)];
      edge.pulses.push({ progress: 0, speed: 0.015 + Math.random() * 0.01 });
    }

    // Advance pulses
    this.edges.forEach(e => {
      for (let i = e.pulses.length - 1; i >= 0; i--) {
        e.pulses[i].progress += e.pulses[i].speed * (1 + this.burstIntensity * 1.5);
        if (e.pulses[i].progress >= 1) {
          e.pulses.splice(i, 1);
        }
      }
    });
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const cx = this.width / 2;
    const cy = this.height / 2;

    // 1. Concentric Radar Rings (Projected as 3D ellipses)
    const rings = [60, 110, 160, 210];
    rings.forEach((r, idx) => {
      ctx.beginPath();
      const segments = 64;
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        const pt = this.project(Math.cos(theta) * r, 0, Math.sin(theta) * r);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = idx === rings.length - 1 
        ? 'rgba(6, 182, 212, 0.35)' 
        : 'rgba(30, 35, 45, 0.8)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Range labels
      if (idx === rings.length - 1) {
        const pLabel = this.project(r + 10, 0, 0);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.5)';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillText('RADAR 10km // PERIMETER', pLabel.x, pLabel.y);
      }
    });

    // 2. Sweeping Radar Beam
    ctx.save();
    const beamEnd = this.project(Math.cos(this.sweepAngle) * 210, 0, Math.sin(this.sweepAngle) * 210);
    const center = this.project(0, 0, 0);
    
    // Draw sweeping line
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(beamEnd.x, beamEnd.y);
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Beam sweep gradient trail
    const trailSegments = 16;
    for (let i = 1; i <= trailSegments; i++) {
      const a1 = this.sweepAngle - (i - 1) * 0.03;
      const a2 = this.sweepAngle - i * 0.03;
      const p1 = this.project(Math.cos(a1) * 210, 0, Math.sin(a1) * 210);
      const p2 = this.project(Math.cos(a2) * 210, 0, Math.sin(a2) * 210);

      ctx.beginPath();
      ctx.moveTo(center.x, center.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.closePath();
      const alpha = (1 - i / trailSegments) * 0.15;
      ctx.fillStyle = `rgba(6, 182, 212, ${alpha})`;
      ctx.fill();
    }
    ctx.restore();

    // 3. Draw Edges and Animated Packet Pulses
    const nodeMap = {};
    this.nodes.forEach(n => {
      const x = Math.cos(n.theta) * n.r;
      const z = Math.sin(n.theta) * n.r;
      nodeMap[n.id] = { node: n, pt: this.project(x, 0, z) };
    });

    this.edges.forEach(e => {
      const from = nodeMap[e.from];
      const to = nodeMap[e.to];
      if (!from || !to) return;

      // Draw edge line
      ctx.beginPath();
      ctx.moveTo(from.pt.x, from.pt.y);
      ctx.lineTo(to.pt.x, to.pt.y);
      if (e.threat === 'attack') {
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.65)';
        ctx.lineWidth = 1.8;
      } else if (e.threat === 'suspicious') {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
        ctx.lineWidth = 1.2;
      } else {
        ctx.strokeStyle = 'rgba(30, 35, 45, 0.8)';
        ctx.lineWidth = 1;
      }
      ctx.stroke();

      // Draw pulses along edge
      e.pulses.forEach(p => {
        const px = from.pt.x + (to.pt.x - from.pt.x) * p.progress;
        const py = from.pt.y + (to.pt.y - from.pt.y) * p.progress;

        ctx.beginPath();
        ctx.arc(px, py, 3 * from.pt.scale, 0, Math.PI * 2);
        ctx.fillStyle = e.threat === 'attack' ? '#ef4444' : '#06b6d4';
        ctx.shadowColor = e.threat === 'attack' ? '#ef4444' : '#06b6d4';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      });
    });

    // 4. Draw Orbiting Nodes
    this.nodes.forEach(n => {
      const data = nodeMap[n.id];
      if (!data) return;
      const pt = data.pt;

      // Outer glowing ring
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, (n.size + 4) * pt.scale, 0, Math.PI * 2);
      ctx.strokeStyle = n.color;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Inner solid node
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, n.size * pt.scale, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.shadowColor = n.color;
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Text label
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(n.name, pt.x + 12 * pt.scale, pt.y - 4 * pt.scale);

      ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.fillText(`${n.ip} (${Math.round(n.risk * 100)}% Risk)`, pt.x + 12 * pt.scale, pt.y + 8 * pt.scale);
    });
  }

  animate() {
    this.update();
    this.draw();
    requestAnimationFrame(() => this.animate());
  }
}

// Global hook
let globalRadar = null;
window.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('radarCanvas')) {
    globalRadar = new ThreatoraRadar('radarCanvas');
  }
});

window.triggerRadarBurst = function() {
  if (globalRadar) {
    globalRadar.triggerBurst();
  }
};
