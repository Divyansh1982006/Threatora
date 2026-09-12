/**
 * Threatora Interactive 3D Cyber Network Globe Canvas
 * High-Performance 60FPS 3D Mathematical Projection Engine
 * Uses Signature Swatch Colors: #F25623 (Orange), #DEDEDE, #4D4D4D, #171717
 */

(function() {
  'use strict';

  function initCyberGlobe() {
    const canvas = document.getElementById('cyberGlobe3d');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width, height, dpr;
    let radius = 160;

    // Mouse & Touch Interaction Physics
    let rotX = 0.25;
    let rotY = 0;
    let targetRotX = 0.2;
    let targetRotY = 0;
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let autoRotateSpeed = 0.004;

    // Nodes on 3D Sphere (spherical coords: lat, lon)
    const nodes = [
      { lat: 0.35, lon: 0.8, label: 'GATEWAY 147.32.84.165', type: 'perimeter', pulse: 0 },
      { lat: -0.2, lon: 1.5, label: 'HOST 192.168.1.105', type: 'threat', risk: '74.8%', pulse: 0 },
      { lat: 0.6, lon: -0.9, label: 'DC-CORP-AD01', type: 'core', pulse: 0 },
      { lat: -0.45, lon: -0.4, label: 'SRV-DATABASE-01', type: 'database', pulse: 0 },
      { lat: 0.1, lon: -2.1, label: 'DMZ-WEB-PROXY', type: 'proxy', pulse: 0 },
      { lat: -0.6, lon: 2.3, label: 'BACKUP-COLD-VAULT', type: 'storage', pulse: 0 },
      { lat: 0.5, lon: 2.8, label: 'WORKSTATION-12', type: 'client', pulse: 0 },
      { lat: -0.1, lon: -1.2, label: 'C2-SUSPECT 81.2.225.29', type: 'c2', risk: '89.2%', pulse: 0 }
    ];

    // Edges between nodes (indices)
    const edges = [
      { from: 0, to: 1, traffic: 1.0 },
      { from: 1, to: 2, traffic: 0.8 },
      { from: 0, to: 4, traffic: 0.5 },
      { from: 2, to: 3, traffic: 0.6 },
      { from: 4, to: 1, traffic: 0.7 },
      { from: 1, to: 7, traffic: 0.9 },
      { from: 2, to: 6, traffic: 0.4 },
      { from: 0, to: 5, traffic: 0.3 }
    ];

    // Flying packets
    const packets = [];
    for (let i = 0; i < 16; i++) {
      packets.push({
        edgeIdx: Math.floor(Math.random() * edges.length),
        progress: Math.random(),
        speed: 0.005 + Math.random() * 0.008,
        color: Math.random() > 0.4 ? '#F25623' : '#FFFFFF'
      });
    }

    // Shockwaves on click/ping
    const shockwaves = [];

    // Spherical grid points (latitudes and longitudes)
    const gridPoints = [];
    const latLines = [-60, -40, -20, 0, 20, 40, 60];
    const lonSteps = 24;

    latLines.forEach(latDeg => {
      const latRad = (latDeg * Math.PI) / 180;
      for (let i = 0; i < lonSteps; i++) {
        const lonRad = (i / lonSteps) * Math.PI * 2;
        gridPoints.push({
          x: Math.cos(latRad) * Math.cos(lonRad),
          y: Math.sin(latRad),
          z: Math.cos(latRad) * Math.sin(lonRad),
          latDeg: latDeg
        });
      }
    });

    function resize() {
      const rect = canvas.getBoundingClientRect();
      dpr = window.devicePixelRatio || 1;
      width = rect.width;
      height = rect.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
      radius = Math.min(width, height) * 0.34;
    }

    window.addEventListener('resize', resize);
    resize();

    // Mouse Interaction Handlers
    canvas.addEventListener('mousedown', function(e) {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    });

    window.addEventListener('mouseup', function() {
      isDragging = false;
    });

    window.addEventListener('mousemove', function(e) {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      if (isDragging) {
        const deltaX = e.clientX - prevMouseX;
        const deltaY = e.clientY - prevMouseY;
        targetRotY += deltaX * 0.007;
        targetRotX += deltaY * 0.007;
        prevMouseX = e.clientX;
        prevMouseY = e.clientY;
      } else if (mouseX >= 0 && mouseX <= rect.width && mouseY >= 0 && mouseY <= rect.height) {
        // Subtle tilt with mouse hover
        const normX = (mouseX / rect.width - 0.5) * 2;
        const normY = (mouseY / rect.height - 0.5) * 2;
        targetRotX = 0.25 - normY * 0.3;
        targetRotY += normX * 0.003;
      }
    });

    // Mobile & Tablet Touch Interaction Handlers
    canvas.addEventListener('touchstart', function(e) {
      if (e.touches.length === 1) {
        isDragging = true;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      }
    }, { passive: true });

    window.addEventListener('touchend', function() {
      isDragging = false;
    }, { passive: true });

    canvas.addEventListener('touchmove', function(e) {
      if (isDragging && e.touches.length === 1) {
        const deltaX = e.touches[0].clientX - prevMouseX;
        const deltaY = e.touches[0].clientY - prevMouseY;
        targetRotY += deltaX * 0.008;
        targetRotX += deltaY * 0.008;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      }
    }, { passive: true });

    // Interactive Click / Tap to spawn shockwave & ping
    canvas.addEventListener('click', function(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      spawnShockwave(x, y);

      // Trigger active pulse on threat node
      nodes[1].pulse = 1.0;
      updateInteractiveTooltip(nodes[1]);
    });

    function spawnShockwave(x, y) {
      shockwaves.push({
        x: x || width / 2,
        y: y || height / 2,
        r: 10,
        maxR: 90,
        alpha: 0.9
      });
    }

    window.triggerGlobeSurge = function() {
      spawnShockwave(width / 2, height / 2);
      nodes[1].pulse = 1.0;
      nodes[7].pulse = 1.0;
      packets.forEach(p => { p.speed *= 2.5; });
      setTimeout(() => {
        packets.forEach(p => { p.speed /= 2.5; });
      }, 1500);
      updateInteractiveTooltip({
        label: 'DATASET TEST BURST: host-becomes-infected.csv',
        risk: '74.8%',
        type: 'INFERENCE_STREAM'
      });
    };

    function updateInteractiveTooltip(node) {
      const readout = document.getElementById('globeReadout');
      if (readout) {
        readout.innerHTML = `
          <div style="color: #F25623; font-weight: 700; font-size: 0.76rem; letter-spacing: 0.5px;">
            ${node.label}
          </div>
          <div style="font-size: 0.70rem; color: #DEDEDE;">
            STATUS: <strong style="color: #F25623;">STREAM ACTIVE</strong> | RISK: <strong style="color: #F25623;">${node.risk || 'NORMAL'}</strong> | 60s WINDOW
          </div>
        `;
        readout.style.opacity = '1';
      }
    }

    // 3D Point Projection Helper
    function project(x, y, z, cx, cy, r) {
      // Rotation around Y axis
      let x1 = x * Math.cos(rotY) - z * Math.sin(rotY);
      let z1 = x * Math.sin(rotY) + z * Math.cos(rotY);

      // Rotation around X axis
      let y2 = y * Math.cos(rotX) - z1 * Math.sin(rotX);
      let z2 = y * Math.sin(rotX) + z1 * Math.cos(rotX);

      // Perspective scale
      const cameraDistance = 3.2;
      const perspective = cameraDistance / (cameraDistance - z2);
      return {
        px: cx + x1 * r * perspective,
        py: cy + y2 * r * perspective,
        pz: z2,
        visible: z2 > -0.6,
        scale: perspective
      };
    }

    // Main 60 FPS Render Loop
    let time = 0;
    function render() {
      time += 0.015;

      // Smooth rotation interpolation
      targetRotY += autoRotateSpeed;
      rotX += (targetRotX - rotX) * 0.08;
      rotY += (targetRotY - rotY) * 0.08;

      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;

      // Draw subtle background ambient glow
      const ambientGlow = ctx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 1.5);
      ambientGlow.addColorStop(0, 'rgba(242, 86, 35, 0.09)');
      ambientGlow.addColorStop(0.5, 'rgba(242, 86, 35, 0.02)');
      ambientGlow.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = ambientGlow;
      ctx.fillRect(0, 0, width, height);

      // Draw Outer Holographic Reticle HUD Rings
      ctx.save();
      ctx.translate(cx, cy);

      // Outer Ring 1 (Rotating Clockwise)
      ctx.rotate(time * 0.25);
      ctx.beginPath();
      ctx.arc(0, 0, radius * 1.25, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(77, 77, 77, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 16]);
      ctx.stroke();

      // Outer Ring 2 with Accent Arc (Rotating Counter-Clockwise)
      ctx.rotate(-time * 0.5);
      ctx.beginPath();
      ctx.arc(0, 0, radius * 1.36, 0.3, Math.PI * 0.8);
      ctx.strokeStyle = 'rgba(242, 86, 35, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.stroke();

      // Degree Tick Marks
      for (let a = 0; a < 8; a++) {
        const angle = (a / 8) * Math.PI * 2;
        const tx1 = Math.cos(angle) * (radius * 1.22);
        const ty1 = Math.sin(angle) * (radius * 1.22);
        const tx2 = Math.cos(angle) * (radius * 1.28);
        const ty2 = Math.sin(angle) * (radius * 1.28);
        ctx.beginPath();
        ctx.moveTo(tx1, ty1);
        ctx.lineTo(tx2, ty2);
        ctx.strokeStyle = 'rgba(222, 222, 222, 0.4)';
        ctx.stroke();
      }
      ctx.restore();

      // Project Grid Points (Wireframe Sphere)
      ctx.fillStyle = 'rgba(222, 222, 222, 0.22)';
      for (let i = 0; i < gridPoints.length; i++) {
        const gp = gridPoints[i];
        const proj = project(gp.x, gp.y, gp.z, cx, cy, radius);
        if (proj.visible) {
          const alpha = Math.max(0.08, (proj.pz + 1) / 2);
          const size = 1.1 * proj.scale;
          ctx.fillStyle = gp.latDeg === 0 
            ? `rgba(242, 86, 35, ${alpha * 0.65})` 
            : `rgba(222, 222, 222, ${alpha * 0.35})`;
          ctx.fillRect(proj.px - size / 2, proj.py - size / 2, size, size);
        }
      }

      // Compute 3D node positions
      const projectedNodes = nodes.map((n, idx) => {
        const nx = Math.cos(n.lat) * Math.cos(n.lon);
        const ny = Math.sin(n.lat);
        const nz = Math.cos(n.lat) * Math.sin(n.lon);
        const proj = project(nx, ny, nz, cx, cy, radius);
        return { ...n, ...proj, idx };
      });

      // Draw Edges (Network Arcs)
      edges.forEach(e => {
        const p1 = projectedNodes[e.from];
        const p2 = projectedNodes[e.to];
        if (!p1 || !p2) return;

        // Determine edge visibility & alpha based on z-depth
        const avgZ = (p1.pz + p2.pz) / 2;
        if (avgZ > -0.7) {
          const alpha = Math.max(0.12, (avgZ + 1) * 0.45);
          ctx.beginPath();
          ctx.moveTo(p1.px, p1.py);

          // Quadratic curve elevated above the sphere surface for 3D curved arc look
          const midX = (p1.px + p2.px) / 2;
          const midY = (p1.py + p2.py) / 2;
          const dist = Math.hypot(p2.px - p1.px, p2.py - p1.py);
          const lift = Math.min(28, dist * 0.18);
          const ctrlX = midX + (midX - cx) * 0.15;
          const ctrlY = midY - lift;

          ctx.quadraticCurveTo(ctrlX, ctrlY, p2.px, p2.py);
          ctx.strokeStyle = (p1.type === 'threat' || p2.type === 'threat')
            ? `rgba(242, 86, 35, ${alpha * 0.85})`
            : `rgba(77, 77, 77, ${alpha * 0.75})`;
          ctx.lineWidth = (p1.type === 'threat' || p2.type === 'threat') ? 1.5 : 1;
          ctx.stroke();
        }
      });

      // Animate & Draw Packets
      packets.forEach(pkt => {
        pkt.progress += pkt.speed;
        if (pkt.progress > 1) pkt.progress = 0;

        const e = edges[pkt.edgeIdx];
        const p1 = projectedNodes[e.from];
        const p2 = projectedNodes[e.to];
        if (!p1 || !p2) return;

        const avgZ = (p1.pz + p2.pz) / 2;
        if (avgZ > -0.6) {
          const t = pkt.progress;
          // Interpolate along the curved arc
          const midX = (p1.px + p2.px) / 2;
          const midY = (p1.py + p2.py) / 2;
          const dist = Math.hypot(p2.px - p1.px, p2.py - p1.py);
          const lift = Math.min(28, dist * 0.18);
          const ctrlX = midX + (midX - cx) * 0.15;
          const ctrlY = midY - lift;

          // Bezier interpolation formula
          const curX = (1 - t) * (1 - t) * p1.px + 2 * (1 - t) * t * ctrlX + t * t * p2.px;
          const curY = (1 - t) * (1 - t) * p1.py + 2 * (1 - t) * t * ctrlY + t * t * p2.py;

          ctx.beginPath();
          ctx.arc(curX, curY, 2.5 * p1.scale, 0, Math.PI * 2);
          ctx.fillStyle = pkt.color;
          ctx.shadowColor = pkt.color;
          ctx.shadowBlur = 8;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });

      // Draw Nodes
      projectedNodes.forEach(n => {
        if (!n.visible) return;

        const isThreat = n.type === 'threat';
        const nodeColor = isThreat ? '#F25623' : (n.type === 'perimeter' ? '#F25623' : '#DEDEDE');
        const nodeSize = (isThreat ? 5.5 : 3.8) * n.scale;

        // Outer glow halo
        ctx.beginPath();
        ctx.arc(n.px, n.py, nodeSize * (isThreat ? 2.8 : 1.8), 0, Math.PI * 2);
        ctx.fillStyle = isThreat ? 'rgba(242, 86, 35, 0.28)' : 'rgba(222, 222, 222, 0.12)';
        ctx.fill();

        // Node core
        ctx.beginPath();
        ctx.arc(n.px, n.py, nodeSize, 0, Math.PI * 2);
        ctx.fillStyle = nodeColor;
        ctx.fill();

        // Extra pulse animation if triggered
        if (n.pulse > 0) {
          n.pulse -= 0.02;
          ctx.beginPath();
          ctx.arc(n.px, n.py, nodeSize + (1 - n.pulse) * 35, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(242, 86, 35, ${n.pulse})`;
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Threat beacon label on high-depth nodes
        if (isThreat && n.pz > -0.1) {
          ctx.font = '600 9px Inter, sans-serif';
          ctx.fillStyle = '#F25623';
          ctx.fillText('192.168.1.105 [ALERT 74.8%]', n.px + 10, n.py + 3);
        }
      });

      // Draw Shockwaves
      for (let s = shockwaves.length - 1; s >= 0; s--) {
        const sw = shockwaves[s];
        sw.r += 2.2;
        sw.alpha -= 0.022;

        if (sw.alpha <= 0 || sw.r >= sw.maxR) {
          shockwaves.splice(s, 1);
        } else {
          ctx.beginPath();
          ctx.arc(sw.x, sw.y, sw.r, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(242, 86, 35, ${sw.alpha})`;
          ctx.lineWidth = 1.8;
          ctx.stroke();
        }
      }

      if (isVisible) {
        animFrameId = requestAnimationFrame(render);
      }
    }

    let isVisible = true;
    let animFrameId = null;
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          isVisible = entry.isIntersecting;
          if (isVisible && !animFrameId) {
            animFrameId = requestAnimationFrame(render);
          }
        });
      }, { threshold: 0.05 });
      observer.observe(canvas);
    } else {
      animFrameId = requestAnimationFrame(render);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCyberGlobe);
  } else {
    initCyberGlobe();
  }
})();
