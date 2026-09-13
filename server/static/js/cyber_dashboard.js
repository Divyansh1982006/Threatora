/**
 * Threatora Cyber Dashboard Client Controller
 * 100% English Interface
 */

// Global State
let currentActiveHostIp = "192.168.1.105";
let currentPlaybookUid = null;
let currentRoleInfo = window.INITIAL_ROLE_INFO || {};
let topologyInstance = null;

// Default Mock Timeline Rollout (10-minute forward simulation)
const defaultMockTimeline = [
  { minute: "+1m", infilt_prob: 0.512, lower_ci: 0.465, upper_ci: 0.560, stage_name: "Initial Access", stage_color: "#3b82f6" },
  { minute: "+2m", infilt_prob: 0.548, lower_ci: 0.490, upper_ci: 0.605, stage_name: "Initial Access", stage_color: "#3b82f6" },
  { minute: "+3m", infilt_prob: 0.583, lower_ci: 0.520, upper_ci: 0.645, stage_name: "Lateral Movement", stage_color: "#ffb703" },
  { minute: "+4m", infilt_prob: 0.637, lower_ci: 0.570, upper_ci: 0.700, stage_name: "Lateral Movement", stage_color: "#ffb703" },
  { minute: "+5m", infilt_prob: 0.681, lower_ci: 0.610, upper_ci: 0.750, stage_name: "Lateral Movement", stage_color: "#ffb703" },
  { minute: "+6m", infilt_prob: 0.724, lower_ci: 0.650, upper_ci: 0.795, stage_name: "Command & Control", stage_color: "#a855f7" },
  { minute: "+7m", infilt_prob: 0.748, lower_ci: 0.670, upper_ci: 0.825, stage_name: "Command & Control", stage_color: "#a855f7" },
  { minute: "+8m", infilt_prob: 0.765, lower_ci: 0.690, upper_ci: 0.840, stage_name: "Exfiltration", stage_color: "#ff0055" },
  { minute: "+9m", infilt_prob: 0.791, lower_ci: 0.710, upper_ci: 0.870, stage_name: "Exfiltration", stage_color: "#ff0055" },
  { minute: "+10m", infilt_prob: 0.816, lower_ci: 0.730, upper_ci: 0.900, stage_name: "Exfiltration", stage_color: "#ff0055" }
];

window.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Tab Navigation
  initTabNavigation();

  // 2. Draw Forward Rollout Timeline Chart
  drawTimelineChart(defaultMockTimeline);

  // 3. Render Initial What-If Simulation Comparison Chart
  renderInitialPlotlyChart();

  // 4. Initialize Network Topology Map
  topologyInstance = new CyberNetworkTopology('topologyCanvas', 'nodeInspector');

  // 5. Connect to Live Health Check
  pollHealthStatus();
  setInterval(pollHealthStatus, 30000);
});

/* Tab Switching Logic */
function initTabNavigation() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabViews = document.querySelectorAll('.tab-view');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetViewId = btn.getAttribute('data-view');

      tabButtons.forEach(b => b.classList.remove('active'));
      tabViews.forEach(v => v.classList.remove('active'));

      btn.classList.add('active');
      const targetView = document.getElementById(targetViewId);
      if (targetView) {
        targetView.classList.add('active');

        // Trigger resize if topology tab opened
        if (targetViewId === 'viewTopology' && topologyInstance) {
          setTimeout(() => topologyInstance.initCanvasSize(), 50);
        }
        // Trigger plot relayout if simulation tab opened
        if (targetViewId === 'viewSimulation') {
          setTimeout(() => Plotly.Plots.resize('plotlySimChart'), 50);
        }
      }
    });
  });
}

function switchTab(viewId) {
  const btn = document.querySelector(`.tab-btn[data-view="${viewId}"]`);
  if (btn) btn.click();
}

/* ========================================================================= */
/* ADVANCED 10-MINUTE INFILTRATION TRAJECTORY VISUALIZATION ENGINE           */
/* Smooth Cubic Splines, Monte Carlo Cloud, Breach Line & Interactive HUD    */
/* ========================================================================= */

window.trajectoryState = {
  data: defaultMockTimeline,
  hoverIndex: null,
  showCI: true,
  showThreshold: true,
  pulseT: 0,
  isHovered: false,
  mouseX: 0,
  mouseY: 0,
  listenersAttached: false
};

function toggleTrajectoryCI() {
  window.trajectoryState.showCI = !window.trajectoryState.showCI;
  const btn = document.getElementById('toggleCiBtn');
  if (btn) btn.classList.toggle('active', window.trajectoryState.showCI);
  renderTrajectoryCanvas();
}

function toggleTrajectoryThreshold() {
  window.trajectoryState.showThreshold = !window.trajectoryState.showThreshold;
  const btn = document.getElementById('toggleBreachBtn');
  if (btn) btn.classList.toggle('active', window.trajectoryState.showThreshold);
  renderTrajectoryCanvas();
}

/* Catmull-Rom to Cubic Bezier Smooth Spline Renderer */
function traceSmoothSpline(ctx, pts) {
  if (!pts || pts.length === 0) return;
  if (pts.length === 1) {
    ctx.moveTo(pts[0].x, pts[0].y);
    return;
  }
  ctx.moveTo(pts[0].x, pts[0].y);
  if (pts.length === 2) {
    ctx.lineTo(pts[1].x, pts[1].y);
    return;
  }
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = i > 0 ? pts[i - 1] : pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = i < pts.length - 2 ? pts[i + 2] : p2;

    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
}

/* High-Definition Interactive Trajectory Render Function */
function renderTrajectoryCanvas() {
  const canvas = document.getElementById('timelineChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const padding = { top: 32, right: 42, bottom: 38, left: 48 };
  const graphW = w - padding.left - padding.right;
  const graphH = h - padding.top - padding.bottom;

  const timeline = window.trajectoryState.data || defaultMockTimeline;
  const n = timeline.length;
  if (n < 2) return;

  ctx.clearRect(0, 0, w, h);

  // 1. Cyber Ambient Background
  const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
  bgGrad.addColorStop(0, '#131313');
  bgGrad.addColorStop(1, '#171717');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  // 2. Cyber Telemetry Grid Lines with '+' Intersection Markers
  const yDivisions = [0, 0.25, 0.5, 0.75, 1.0];
  ctx.lineWidth = 1;

  yDivisions.forEach(val => {
    const y = padding.top + (1 - val) * graphH;
    ctx.strokeStyle = val === 0 ? '#383838' : 'rgba(77, 77, 77, 0.22)';
    ctx.setLineDash(val === 0 ? [] : [3, 6]);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();

    // Y-Axis Labels
    ctx.setLineDash([]);
    ctx.fillStyle = '#8E8E8E';
    ctx.font = "600 10px Inter, sans-serif";
    ctx.textAlign = 'right';
    ctx.fillText((val * 100).toFixed(0) + '%', padding.left - 10, y + 3.5);
  });

  // Calculate coordinates for all points
  const xStep = graphW / (n - 1);
  const meanPts = [];
  const upperPts = [];
  const lowerPts = [];

  for (let i = 0; i < n; i++) {
    const pt = timeline[i];
    const x = padding.left + i * xStep;
    const yMean = padding.top + (1 - Math.max(0, Math.min(1, pt.infilt_prob))) * graphH;
    const yUpper = padding.top + (1 - Math.max(0, Math.min(1, pt.upper_ci || (pt.infilt_prob + 0.06)))) * graphH;
    const yLower = padding.top + (1 - Math.max(0, Math.min(1, pt.lower_ci || (pt.infilt_prob - 0.06)))) * graphH;

    meanPts.push({ x, y: yMean, pt, i });
    upperPts.push({ x, y: yUpper });
    lowerPts.push({ x, y: yLower });

    // Vertical Timeline Grid Line
    ctx.strokeStyle = 'rgba(77, 77, 77, 0.18)';
    ctx.setLineDash([2, 8]);
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, h - padding.bottom);
    ctx.stroke();

    // Cross markers at grid crossings
    yDivisions.forEach(val => {
      const yCross = padding.top + (1 - val) * graphH;
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(222, 222, 222, 0.2)';
      ctx.fillRect(x - 1, yCross - 1, 2, 2);
    });

    // X-Axis Minute Labels
    ctx.setLineDash([]);
    ctx.fillStyle = (window.trajectoryState.hoverIndex === i) ? '#F25623' : '#A0A0A0';
    ctx.font = (window.trajectoryState.hoverIndex === i) ? "700 10px 'JetBrains Mono', monospace" : "500 10px 'JetBrains Mono', monospace";
    ctx.textAlign = 'center';
    ctx.fillText(pt.minute, x, h - padding.bottom + 18);
  }

  // 3. Draw 70% Breach Threshold Horizon Line
  if (window.trajectoryState.showThreshold) {
    const breachY = padding.top + (1 - 0.70) * graphH;

    // Breach danger zone subtle background tint
    const dangerTint = ctx.createLinearGradient(0, padding.top, 0, breachY);
    dangerTint.addColorStop(0, 'rgba(242, 86, 35, 0.08)');
    dangerTint.addColorStop(1, 'rgba(242, 86, 35, 0.00)');
    ctx.fillStyle = dangerTint;
    ctx.fillRect(padding.left, padding.top, graphW, breachY - padding.top);

    // Dotted red/orange breach line
    ctx.strokeStyle = '#F25623';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, breachY);
    ctx.lineTo(w - padding.right, breachY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Breach line badge
    ctx.fillStyle = '#F25623';
    ctx.font = "700 9px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right';
    ctx.fillText('CRITICAL BREACH THRESHOLD [70%]', w - padding.right, breachY - 6);
  }

  // 4. Draw 95% Monte Carlo Confidence Cloud (Smooth Spline)
  if (window.trajectoryState.showCI) {
    ctx.save();
    ctx.beginPath();
    traceSmoothSpline(ctx, upperPts);
    // Reverse lower points to close ribbon
    const lowerReversed = [...lowerPts].reverse();
    ctx.lineTo(lowerReversed[0].x, lowerReversed[0].y);
    traceSmoothSpline(ctx, lowerReversed);
    ctx.closePath();

    const ciGrad = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    ciGrad.addColorStop(0, 'rgba(242, 86, 35, 0.08)');
    ciGrad.addColorStop(0.5, 'rgba(242, 86, 35, 0.04)');
    ciGrad.addColorStop(1, 'rgba(242, 86, 35, 0.01)');
    ctx.fillStyle = ciGrad;
    ctx.fill();

    // Boundary stroke
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // 5. Draw Luminous Area Fill Under Mean Trajectory Curve
  ctx.save();
  ctx.beginPath();
  traceSmoothSpline(ctx, meanPts);
  ctx.lineTo(meanPts[meanPts.length - 1].x, h - padding.bottom);
  ctx.lineTo(meanPts[0].x, h - padding.bottom);
  ctx.closePath();

  const areaGrad = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
  areaGrad.addColorStop(0, 'rgba(242, 86, 35, 0.15)');
  areaGrad.addColorStop(0.6, 'rgba(242, 86, 35, 0.05)');
  areaGrad.addColorStop(1, 'rgba(18, 18, 18, 0.0)');
  ctx.fillStyle = areaGrad;
  ctx.fill();
  ctx.restore();

  // 6. Draw Smooth Main Trajectory Curve
  ctx.save();
  ctx.beginPath();
  traceSmoothSpline(ctx, meanPts);
  ctx.strokeStyle = '#F25623';
  ctx.lineWidth = 3;
  ctx.shadowColor = 'rgba(242, 86, 35, 0.65)';
  ctx.shadowBlur = 12;
  ctx.stroke();
  ctx.restore();

  // 7. Draw Nodes & MITRE ATT&CK Phase Dots
  meanPts.forEach(p => {
    const isHovered = window.trajectoryState.hoverIndex === p.i;
    const pt = p.pt;
    const stageColor = pt.stage_color || '#F25623';

    // Expanding beacon ring on hovered node
    if (isHovered) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(242, 86, 35, 0.25)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Outer Halo
    ctx.beginPath();
    ctx.arc(p.x, p.y, isHovered ? 7 : 5.5, 0, Math.PI * 2);
    ctx.fillStyle = stageColor;
    ctx.shadowColor = stageColor;
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Inner White Core
    ctx.beginPath();
    ctx.arc(p.x, p.y, isHovered ? 3.5 : 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
  });

  // 8. Draw Laser Crosshair & Interactive Tactical HUD Tooltip
  if (window.trajectoryState.hoverIndex !== null && window.trajectoryState.hoverIndex < meanPts.length) {
    const target = meanPts[window.trajectoryState.hoverIndex];
    const pt = target.pt;

    // Laser Vertical Scanline
    ctx.strokeStyle = 'rgba(242, 86, 35, 0.75)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(target.x, padding.top);
    ctx.lineTo(target.x, h - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Glassmorphic HUD Tooltip
    const tooltipW = 170;
    const tooltipH = 82;
    let tooltipX = target.x - (tooltipW / 2);
    let tooltipY = target.y - tooltipH - 12;

    // Clamp inside canvas boundary
    if (tooltipX < padding.left) tooltipX = padding.left;
    if (tooltipX + tooltipW > w - padding.right) tooltipX = w - padding.right - tooltipW;
    
    // If tooltip goes above chart, flip it below
    if (tooltipY < 10) {
      tooltipY = target.y + 12;
    }

    // Tooltip Box Container
    ctx.save();
    ctx.fillStyle = 'rgba(20, 20, 20, 0.96)';
    ctx.strokeStyle = '#F25623';
    ctx.lineWidth = 1.2;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.85)';
    ctx.shadowBlur = 18;

    ctx.beginPath();
    ctx.roundRect(tooltipX, tooltipY, tooltipW, tooltipH, 8);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Tooltip Header
    ctx.fillStyle = '#F25623';
    ctx.font = "700 9px 'JetBrains Mono', monospace";
    ctx.textAlign = 'left';
    ctx.fillText(`⏱️ HORIZON: ${pt.minute}`, tooltipX + 8, tooltipY + 16);

    // Divider Line
    ctx.strokeStyle = 'rgba(77, 77, 77, 0.5)';
    ctx.beginPath();
    ctx.moveTo(tooltipX + 8, tooltipY + 22);
    ctx.lineTo(tooltipX + tooltipW - 8, tooltipY + 22);
    ctx.stroke();

    // Risk Value
    const riskPct = (pt.infilt_prob * 100).toFixed(1);
    ctx.fillStyle = pt.infilt_prob >= 0.7 ? '#F25623' : '#DEDEDE';
    ctx.font = "700 11px Inter, sans-serif";
    ctx.fillText(`Risk: ${riskPct}% ${pt.infilt_prob >= 0.7 ? '⚠ CRITICAL' : ''}`, tooltipX + 8, tooltipY + 38);

    // ATT&CK Stage
    ctx.fillStyle = '#DEDEDE';
    ctx.font = "500 9px Inter, sans-serif";
    ctx.fillText(`Phase: ${pt.stage_name || 'Lateral Movement'}`, tooltipX + 8, tooltipY + 52);

    // 95% CI Range
    const lowerPct = ((pt.lower_ci || pt.infilt_prob - 0.05) * 100).toFixed(1);
    const upperPct = ((pt.upper_ci || pt.infilt_prob + 0.05) * 100).toFixed(1);
    ctx.fillStyle = '#8E8E8E';
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillText(`95% CI: [${lowerPct}% — ${upperPct}%]`, tooltipX + 8, tooltipY + 66);
    ctx.restore();
  }
}

/* Attach Interactive Mouse/Touch Events to Canvas */
function attachTrajectoryEvents() {
  const canvas = document.getElementById('timelineChart');
  if (!canvas || window.trajectoryState.listenersAttached) return;

  function handlePointer(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const padding = { top: 32, right: 42, bottom: 38, left: 48 };
    const graphW = rect.width - padding.left - padding.right;
    const n = (window.trajectoryState.data || defaultMockTimeline).length;

    if (x >= padding.left - 15 && x <= rect.width - padding.right + 15) {
      const xStep = graphW / (n - 1);
      const relativeX = x - padding.left;
      const closestIdx = Math.max(0, Math.min(n - 1, Math.round(relativeX / xStep)));
      if (window.trajectoryState.hoverIndex !== closestIdx) {
        window.trajectoryState.hoverIndex = closestIdx;
        renderTrajectoryCanvas();
      }
    } else if (window.trajectoryState.hoverIndex !== null) {
      window.trajectoryState.hoverIndex = null;
      renderTrajectoryCanvas();
    }
  }

  canvas.addEventListener('mousemove', e => {
    handlePointer(e.clientX, e.clientY);
  });

  canvas.addEventListener('mouseleave', () => {
    window.trajectoryState.hoverIndex = null;
    renderTrajectoryCanvas();
  });

  canvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      handlePointer(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  canvas.addEventListener('touchmove', e => {
    if (e.touches.length === 1) {
      handlePointer(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  canvas.addEventListener('touchend', () => {
    setTimeout(() => {
      window.trajectoryState.hoverIndex = null;
      renderTrajectoryCanvas();
    }, 1500);
  }, { passive: true });

  window.trajectoryState.listenersAttached = true;
}

/* Public Entry Point to Update and Draw Timeline */
function drawTimelineChart(timeline) {
  if (timeline && timeline.length > 0) {
    window.trajectoryState.data = timeline;

    // Update Peak Risk Badge in Panel Header
    const maxPt = timeline.reduce((max, pt) => pt.infilt_prob > max.infilt_prob ? pt : max, timeline[0]);
    const peakBadge = document.getElementById('peakRiskBadge');
    if (peakBadge && maxPt) {
      peakBadge.innerText = `PEAK: ${(maxPt.infilt_prob * 100).toFixed(1)}% @ ${maxPt.minute}`;
      peakBadge.style.color = maxPt.infilt_prob >= 0.7 ? '#F25623' : '#10b981';
      peakBadge.style.borderColor = maxPt.infilt_prob >= 0.7 ? '#F25623' : '#10b981';
    }
  }

  attachTrajectoryEvents();
  renderTrajectoryCanvas();
}

window.addEventListener('resize', () => {
  renderTrajectoryCanvas();
});

/* What-If Simulation Plotly Comparison Chart */
function renderInitialPlotlyChart() {
  const chartDiv = document.getElementById('plotlySimChart');
  if (!chartDiv) return;

  const xVals = defaultMockTimeline.map(t => t.minute);
  const baseRisk = [74.8, 76.5, 78.2, 80.0, 81.5, 82.8, 84.1, 85.0, 86.2, 87.5];
  const simRisk =  [74.8, 62.0, 56.4, 51.2, 49.0, 48.2, 47.8, 47.5, 47.1, 46.8];

  const traceBase = {
    x: xVals,
    y: baseRisk,
    name: 'Baseline (No Action)',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: '#F25623', width: 2.5, dash: 'dot' },
    marker: { size: 6, color: '#F25623' }
  };

  const traceSim = {
    x: xVals,
    y: simRisk,
    name: 'Simulated Intervention',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: '#10b981', width: 2.5 },
    marker: { size: 7, symbol: 'diamond', color: '#10b981' }
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#DEDEDE', family: 'JetBrains Mono, monospace', size: 11 },
    margin: { t: 20, r: 25, b: 35, l: 45 },
    xaxis: { gridcolor: '#333333', zeroline: false },
    yaxis: { gridcolor: '#333333', zeroline: false, title: 'Infiltration Risk (%)', range: [0, 100] },
    legend: { orientation: 'h', y: 1.15, x: 0.05, font: { color: '#FFFFFF' } }
  };

  Plotly.newPlot('plotlySimChart', [traceBase, traceSim], layout, { responsive: true, displayModeBar: false });
}

/* Trigger What-If Counterfactual Simulation */
async function runWhatIfSimulation() {
  const action = document.getElementById('simActionSelect').value;
  const summaryBadge = document.getElementById('simSummaryBadge');
  const verdictBox = document.getElementById('simVerdictBox');

  summaryBadge.innerText = "Simulating counterfactual dynamics...";
  verdictBox.style.display = 'none';

  try {
    const res = await fetch('/api/v1/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: action,
        target_ip: currentActiveHostIp,
        horizon: 10
      })
    });

    const data = await res.json();
    if (res.status === 403) {
      showAccessDenied('can_simulate', data.message);
      summaryBadge.innerText = "";
      return;
    }
    if (data.status !== 'success') {
      alert("Simulation error: " + data.message);
      summaryBadge.innerText = "";
      return;
    }

    summaryBadge.innerText = `✔ ${data.pct_risk_reduction}% Risk Reduction (${data.effectiveness})`;
    verdictBox.style.display = 'block';
    verdictBox.innerHTML = `<strong>AI Tactical Assessment:</strong> ${data.tactical_verdict}`;

    const xVals = data.timeline.map(t => t.minute);
    const baseRisk = data.timeline.map(t => t.baseline_risk * 100);
    const simRisk = data.timeline.map(t => t.simulated_risk * 100);

    const traceBase = {
      x: xVals,
      y: baseRisk,
      name: 'Baseline (No Action)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#F25623', width: 2.5, dash: 'dot' },
      marker: { size: 6, color: '#F25623' }
    };

    const traceSim = {
      x: xVals,
      y: simRisk,
      name: `Action: ${data.action_name}`,
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#10b981', width: 2.5 },
      marker: { size: 7, symbol: 'diamond', color: '#10b981' }
    };

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#DEDEDE', family: 'JetBrains Mono, monospace', size: 11 },
      margin: { t: 20, r: 25, b: 35, l: 45 },
      xaxis: { gridcolor: '#333333', zeroline: false },
      yaxis: { gridcolor: '#333333', zeroline: false, title: 'Infiltration Risk (%)', range: [0, 100] },
      legend: { orientation: 'h', y: 1.15, x: 0.05, font: { color: '#FFFFFF' } }
    };

    Plotly.newPlot('plotlySimChart', [traceBase, traceSim], layout, { responsive: true, displayModeBar: false });
    showToast(`Simulation completed: ${data.pct_risk_reduction}% risk reduction`);
  } catch (err) {
    alert("Failed to execute simulation: " + err);
    summaryBadge.innerText = "";
  }
}

/* Real Dataset Testing Controller */
async function runSelectedDatasetTest() {
  const select = document.getElementById('datasetSelect');
  const dataset = select ? select.value : 'host-becomes-infected';
  const badge = document.getElementById('inferenceStatusBadge');
  const btn = document.getElementById('btnExecuteInference');

  if (btn) btn.disabled = true;
  if (badge) {
    badge.innerHTML = '<span class="beacon-dot" style="background:#F25623;"></span><span>RUNNING INFERENCE...</span>';
    badge.className = 'status-pill status-pill-crimson';
  }

  try {
    const res = await fetch(`/api/demo?dataset=${dataset}`);
    const data = await res.json();
    if (!res.ok) {
      alert("Inference error: " + (data.message || `HTTP ${res.status}`));
      return;
    }
    renderDashboard(data);
    if (badge) {
      badge.innerHTML = '<span class="beacon-dot"></span><span>INFERENCE COMPLETE</span>';
      badge.className = 'status-pill status-pill-green';
    }
    showToast(`Dataset '${dataset}.csv' processed. 10-minute forward trajectory unrolled.`);
  } catch (err) {
    alert("Error running inference on dataset: " + err);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function runDemoScenario() {
  return runSelectedDatasetTest();
}

function handleDatasetChange(val) {
  const badge = document.getElementById('inferenceStatusBadge');
  if (badge) {
    badge.innerHTML = '<span class="beacon-dot"></span><span>READY TO TEST</span>';
    badge.className = 'status-pill status-pill-cyan';
  }
}

/* PCAP or Flow File Upload */
async function handleFileUpload() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  if (!file) return;

  // Cloud deployments have a 200MB upload limit to prevent OOM on free tier
  const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200 MB
  if (file.size > MAX_FILE_SIZE) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    showToast(`File too large (${sizeMB} MB). Max upload is 200 MB on cloud.`, 'error');
    fileInput.value = '';
    return;
  }

  const badge = document.getElementById('inferenceStatusBadge');
  if (badge) {
    badge.innerHTML = '<span class="beacon-dot" style="background:#F25623;"></span><span>ANALYZING UPLOAD...</span>';
    badge.className = 'status-pill status-pill-crimson';
  }
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });

    // Safely parse JSON — server may return empty body on crash or HTML on redirect
    let data = null;
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      try { data = await res.json(); } catch (_) { data = null; }
    } else {
      // Non-JSON response (HTML redirect, empty body, nginx error page, etc.)
      const rawText = await res.text().catch(() => '');
      if (res.status === 401 || res.redirected) {
        window.location.href = '/login';
        return;
      }
      const badge2 = document.getElementById('inferenceStatusBadge');
      if (badge2) {
        badge2.innerHTML = '<span class="beacon-dot" style="background:#F25623;"></span><span>SERVER ERROR</span>';
        badge2.className = 'status-pill status-pill-crimson';
      }
      showToast(`Server error (HTTP ${res.status}). The engine may still be loading — please retry in 30s.`, 'error');
      fileInput.value = '';
      return;
    }

    if (res.status === 401) {
      window.location.href = '/login';
      return;
    }
    if (res.status === 403) {
      showAccessDenied('can_upload', (data && data.message) || 'Elevated clearance required.');
      if (badge) {
        badge.innerHTML = '<span class="beacon-dot" style="background:#F25623;"></span><span>ACCESS DENIED</span>';
        badge.className = 'status-pill status-pill-crimson';
      }
      return;
    }
    if (res.status === 503) {
      showToast('ML engines are still initializing — please retry in ~30 seconds.', 'error');
      if (badge) {
        badge.innerHTML = '<span class="beacon-dot" style="background:#F25623;"></span><span>ENGINE LOADING</span>';
        badge.className = 'status-pill status-pill-crimson';
      }
      fileInput.value = '';
      return;
    }
    if (!res.ok) {
      const msg = (data && data.message) || `HTTP ${res.status} — server error`;
      showToast('Upload failed: ' + msg, 'error');
      if (badge) {
        badge.innerHTML = '<span class="beacon-dot" style="background:#F25623;"></span><span>UPLOAD FAILED</span>';
        badge.className = 'status-pill status-pill-crimson';
      }
      return;
    }

    const fnEl = document.getElementById('fileName');
    if (fnEl) fnEl.innerText = file.name + " (analyzed)";
    if (badge) {
      badge.innerHTML = '<span class="beacon-dot"></span><span>UPLOAD ANALYZED</span>';
      badge.className = 'status-pill status-pill-green';
    }
    renderDashboard(data);
    showToast(`Capture ${file.name} successfully analyzed.`);
  } catch (err) {
    // Network-level failure (fetch itself threw — offline, CORS, etc.)
    const msg = err && err.message ? err.message : String(err);
    showToast('Upload failed: ' + msg + '. Check your connection and retry.', 'error');
    if (badge) {
      badge.innerHTML = '<span class="beacon-dot" style="background:#F25623;"></span><span>NETWORK ERROR</span>';
      badge.className = 'status-pill status-pill-crimson';
    }
    fileInput.value = '';
  }
}

/* Render Dashboard Data */
function renderDashboard(data) {
  if (!data.hosts || data.hosts.length === 0) {
    alert(data.message || "No valid traffic parsed.");
    return;
  }

  const host = data.hosts[0]; // Lead flagged host

  // 1. Update KPI Metric Tiles & Unified Banner
  const riskPct = (host.current_risk_score * 100).toFixed(1);
  const valRisk = document.getElementById('valRisk');
  if (valRisk) {
    valRisk.innerText = riskPct + "%";
    valRisk.style.color = host.is_anomalous ? "var(--neon-crimson)" : "var(--neon-emerald)";
  }

  const valStage = document.getElementById('valStage');
  if (valStage) {
    valStage.innerText = host.current_stage.name;
    valStage.style.color = host.is_anomalous ? "var(--neon-amber)" : "var(--neon-emerald)";
  }

  const targetIpVal = document.getElementById('kpiTargetIp');
  if (targetIpVal) {
    targetIpVal.innerText = host.host_ip;
  }

  // 2. Redraw Forward Rollout Timeline Chart
  drawTimelineChart(host.forecast_timeline);

  // 3. Update Live Network Stats
  if (host.live_stats) {
    const elFlows = document.getElementById('statFlows');
    if (elFlows) elFlows.innerText = host.live_stats.n_flows;
    
    const elPktSize = document.getElementById('statPktSize');
    if (elPktSize) elPktSize.innerText = Math.round(host.live_stats.avg_pkt_size) + " B";
    
    const elOutbound = document.getElementById('statOutbound');
    if (elOutbound) elOutbound.innerText = (host.live_stats.frac_outbound * 100).toFixed(1) + "%";
    
    const elProto = document.getElementById('statProto');
    if (elProto) {
      const tcp = (host.live_stats.frac_tcp * 100).toFixed(0);
      const udp = (host.live_stats.frac_udp * 100).toFixed(0);
      elProto.innerText = `${tcp}% / ${udp}%`;
    }
  }

  // 4. Update Incident & Mitigation Playbook Panel
  const socBox = document.getElementById('socIncidentBox');
  if (host.is_anomalous) {
    socBox.style.display = 'block';
    currentActiveHostIp = host.host_ip;

    const pb = (data.playbooks || []).find(p => p.target_ip === host.host_ip);
    if (pb) {
      currentPlaybookUid = pb.playbook_uid;
      document.getElementById('damageAssessmentText').innerText = pb.damage_assessment || host.current_stage.metadata.description;
      document.getElementById('socActionText').innerText = "Strategy: " + pb.containment_strategy + "\n\nZero-Trust Rules:\n" + (pb.containment_commands || []).join('\n');
      document.getElementById('playbookBadge').innerText = pb.status || "PLAYBOOK ACTIVE";
    }

    const btnIsolate = document.getElementById('btnIsolate');
    btnIsolate.disabled = false;
    btnIsolate.innerText = `🛡️ Execute 1-Click Containment (${host.host_ip})`;
  } else {
    socBox.style.display = 'none';
  }

  // 5. Update Compact Explainability List
  const explainDiv = document.getElementById('explainBars');
  if (explainDiv) {
    explainDiv.innerHTML = '';
    const topFeats = host.explainability.top_features || {};
    for (const [feat, score] of Object.entries(topFeats)) {
      explainDiv.innerHTML += `
        <div class="explain-item">
          <span>${feat}</span>
          <span class="mono">${score.toFixed(1)}%</span>
        </div>
      `;
    }
  }

  // 6. Update Feature Deltas List
  const deltasDiv = document.getElementById('stateDeltasList');
  deltasDiv.innerHTML = '';
  if (host.explainability.state_deltas && host.explainability.state_deltas.length > 0) {
    host.explainability.state_deltas.slice(0, 3).forEach(d => {
      deltasDiv.innerHTML += `<div>• <strong>${d.feature}</strong>: shift ${d.current_value} → ${d.forecast_value} (${d.pct_change > 0 ? '+' : ''}${d.pct_change}%)</div>`;
    });
  } else {
    deltasDiv.innerHTML = `<div>Expected steady state across operational baseline.</div>`;
  }

  // 7. Update Monitored Endpoints Table
  const tb = document.getElementById('hostsTableBody');
  if (tb) {
    tb.innerHTML = '';
    data.hosts.forEach(h => {
      tb.innerHTML += `
        <tr>
          <td><strong class="mono">${h.host_ip}</strong></td>
          <td>${h.window_count}</td>
          <td><strong style="color:${h.is_anomalous ? 'var(--neon-crimson)' : 'var(--neon-emerald)'};">${(h.current_risk_score * 100).toFixed(1)}%</strong></td>
          <td><span class="badge-${h.is_anomalous ? 'threat' : 'benign'}">${h.current_stage.name}</span></td>
          <td>
            <button class="btn-cyber btn-cyber-danger" style="padding:4px 10px; font-size:0.72rem;" onclick="triggerDirectIsolation('${h.host_ip}')">
              Isolate
            </button>
          </td>
        </tr>
      `;
    });
  }

  // 8. Refresh Topology if available
  if (topologyInstance) {
    topologyInstance.loadTopology();
  }
}

/* 1-Click Zero-Trust Quarantine / Host Isolation */
async function triggerMitigation() {
  if (!currentActiveHostIp) return;
  triggerDirectIsolation(currentActiveHostIp, currentPlaybookUid);
}

async function triggerDirectIsolation(targetIp, playbookUid) {
  const btn = document.getElementById('btnIsolate');
  if (btn) {
    btn.disabled = true;
    btn.innerText = "Applying Zero-Trust Rules...";
  }

  try {
    const res = await fetch('/api/v1/mitigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_ip: targetIp,
        playbook_uid: playbookUid || currentPlaybookUid,
        action: 'isolate'
      })
    });

    const data = await res.json();
    if (res.status === 403) {
      showAccessDenied('can_mitigate', data.message);
      return;
    }
    if (data.status === 'success' || data.status === 'warning') {
      showToast(`Host ${targetIp} quarantined successfully.`);
      if (btn) btn.innerText = "✔ Host Isolated";
      if (topologyInstance) topologyInstance.loadTopology();
    } else {
      alert("Mitigation failed: " + data.message);
      if (btn) btn.disabled = false;
    }
  } catch (err) {
    alert("Network error executing containment: " + err);
    if (btn) btn.disabled = false;
  }
}
window.triggerDirectIsolation = triggerDirectIsolation;

/* Access Denied Modal */
function showAccessDenied(perm, actionName) {
  const modal = document.getElementById('accessDeniedModal');
  if (modal) {
    document.getElementById('deniedDetailMsg').innerText = 
      `The action '${actionName || perm}' was blocked by Zero-Trust policy. Elevated clearance is required.`;
    modal.classList.add('show');
  }
}

function closeAccessDeniedModal() {
  const modal = document.getElementById('accessDeniedModal');
  if (modal) modal.classList.remove('show');
}

/* Role Switching */
async function handleRoleSwitch(newRole) {
  try {
    const res = await fetch('/api/v1/auth/switch-role', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: newRole })
    });
    const data = await res.json();
    if (data.status === 'success') {
      showToast(`Clearance elevated to ${data.role_info.title}`);
      setTimeout(() => window.location.reload(), 600);
    } else {
      alert("Failed to switch clearance: " + data.message);
    }
  } catch (err) {
    alert("Error communicating with auth service: " + err);
  }
}

/* Health Poller */
async function pollHealthStatus() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      const statusPill = document.getElementById('engineStatusPill');
      if (statusPill) {
        statusPill.innerHTML = `<span class="beacon-dot"></span>ENGINE ONLINE (${data.device.toUpperCase()})`;
      }
    }
  } catch (e) {}
}

/* Floating Toast Notifications */
function showToast(msg, type) {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = 'toast-msg' + (type === 'error' ? ' toast-msg-error' : '');
  const icon = type === 'error' ? '⚠️' : '🛡️';
  toast.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'all 0.3s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}
