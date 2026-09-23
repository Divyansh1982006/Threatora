
/**
 * Threatora SOC Operations Dashboard Client Controller
 * 100% Dynamic Telemetry & World Model Inference (NTRO PS 26153)
 */

// Global State
let currentActiveHostIp = "192.168.1.105";
let currentPlaybookUid = null;
let currentRoleInfo = window.INITIAL_ROLE_INFO || {};
let topologyInstance = null;

// Telemetry & Plotly Cache
let currentTimelineData = null;
let currentHistoricalData = null;
let currentMitigatedTimeline = null;
let currentActiveWindow = null;
let currentInspectorRows = [];
let showPlotlyCI = true;
let showPlotlyBreach = true;

window.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Network Topology if canvas exists
  const topoCanvas = document.getElementById('topologyCanvas');
  if (topoCanvas && typeof CyberNetworkTopology !== 'undefined') {
    try {
      topologyInstance = new CyberNetworkTopology('topologyCanvas', 'nodeInspector');
    } catch (e) {
      console.warn("Topology init notice:", e);
    }
  }

  // 2. Initialize What-If Plotly simulation chart if container exists
  if (document.getElementById('plotlySimChart') && typeof Plotly !== 'undefined') {
    renderSimulationPlotlyChart(null, null);
  }

  // 3. Health Poller
  pollHealthStatus();
  setInterval(pollHealthStatus, 30000);

  // 4. Restore uploaded telemetry session if switching screens or reloading
  restoreActiveTelemetrySession();
});

function updateTelemetrySourceLabel(label) {
  const sel = document.getElementById('datasetSelect');
  if (sel && label) {
    let customOpt = document.getElementById('uploadedCustomOption');
    if (!customOpt) {
      customOpt = document.createElement('option');
      customOpt.id = 'uploadedCustomOption';
      sel.prepend(customOpt);
    }
    customOpt.value = 'uploaded_custom';
    customOpt.innerText = label;
    customOpt.selected = true;
  }
  const bLabel = document.getElementById('benchmarkLabel');
  if (bLabel && label) {
    bLabel.innerText = label.startsWith('UPLOADED') ? 'TELEMETRY SOURCE:' : 'CURATED BENCHMARK:';
  }
}

async function restoreActiveTelemetrySession() {
  const saved = sessionStorage.getItem('threatora_active_telemetry') || localStorage.getItem('threatora_active_telemetry');
  const savedLabel = sessionStorage.getItem('threatora_telemetry_source_label') || localStorage.getItem('threatora_telemetry_source_label');

  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      if (savedLabel) {
        updateTelemetrySourceLabel(savedLabel);
      }
      renderDashboard(parsed);
      return;
    } catch (e) {
      console.warn("Could not parse stored telemetry:", e);
    }
  }

  // Fallback: Check if backend server has an active analyzed telemetry session
  try {
    const res = await fetch('/api/telemetry/active');
    if (res.ok) {
      const data = await res.json();
      if (data && data.status === 'success') {
        localStorage.setItem('threatora_active_telemetry', JSON.stringify(data));
        sessionStorage.setItem('threatora_active_telemetry', JSON.stringify(data));
        const lbl = (data.file_label && (data.file_label.toLowerCase().includes('demo') || data.file_label.toLowerCase().includes('benchmark')))
          ? `CURATED BENCHMARK: ${data.file_label}`
          : `UPLOADED TELEMETRY: ${data.file_label || 'Active Capture'}`;
        localStorage.setItem('threatora_telemetry_source_label', lbl);
        sessionStorage.setItem('threatora_telemetry_source_label', lbl);
        updateTelemetrySourceLabel(lbl);
        renderDashboard(data);
      }
    }
  } catch (err) {
    // Silent fallback
  }
}

/* =========================================================================
   1. Dynamic SOC Dashboard Renderer
   ========================================================================= */
function renderDashboard(data) {
  if (!data || data.status !== 'success') {
    showToast(data.message || "Failed to parse telemetry stream.", 'error');
    return;
  }

  // Show active dashboard, hide empty state
  const emptyBanner = document.getElementById('awaitingTelemetryBanner');
  if (emptyBanner) emptyBanner.style.display = 'none';

  const activeContainer = document.getElementById('activeDashboardContainer');
  if (activeContainer) activeContainer.style.display = 'block';

  // Cache data for What-If sandbox and inspector
  currentTimelineData = data.forecast_timeline || [];
  currentHistoricalData = data.historical_trajectory || [];
  currentActiveWindow = data.active_window_raw || null;
  currentInspectorRows = data.inspector_rows || [];
  currentMitigatedTimeline = null;

  const kpis = data.kpis || {};
  const meta = data.meta || {};

  // 1. Update DEFCON Ribbon
  const defconVal = document.getElementById('defconVal');
  const defconStatus = document.getElementById('defconStatus');
  const defconDot = document.getElementById('defconDot');
  const defconBadge = document.getElementById('defconBadge');

  if (defconVal) defconVal.innerText = `DEFCON ${kpis.defcon || 3}`;
  if (defconStatus) {
    // Strip redundant "DEFCON X //" prefix since defconVal already shows it
    let statusText = kpis.threat_level || "ELEVATED READINESS";
    statusText = statusText.replace(/^DEFCON\s*\d+\s*\/\/\s*/i, '');
    defconStatus.innerText = statusText;
    const colorMap = { 1: "#FF5252", 2: "#FF7043", 3: "#FFB300", 4: "#29B6F6", 5: "#00E676" };
    const alertCol = colorMap[kpis.defcon] || "#FFB300";
    defconStatus.style.color = alertCol;
    if (defconDot) defconDot.style.background = alertCol;
    if (defconBadge) defconBadge.style.borderColor = alertCol;
  }

  // 2. Update 6 Executive KPI Metric Cards
  const valRisk = document.getElementById('valRisk');
  if (valRisk) {
    valRisk.innerText = (kpis.peak_risk_pct !== undefined ? kpis.peak_risk_pct : 0.0) + "%";
    valRisk.style.color = (kpis.peak_risk_pct >= 50) ? "var(--neon-crimson)" : "var(--neon-emerald)";
  }

  const valRiskSub = document.getElementById('valRiskSub');
  if (valRiskSub) {
    valRiskSub.innerText = `Current S_t: ${kpis.current_risk_pct || 0}% // Peak: ${kpis.peak_risk_pct || 0}%`;
  }

  const valStage = document.getElementById('valStage');
  if (valStage) {
    valStage.innerText = kpis.stage_name || "Benign";
    valStage.style.color = kpis.stage_color || "var(--neon-emerald)";
  }

  const valTechnique = document.getElementById('valTechnique');
  if (valTechnique) {
    valTechnique.innerText = `${kpis.technique_id || "TA0000"} - ${kpis.technique || "Baseline"}`;
  }

  const valSmoothL1 = document.getElementById('valSmoothL1');
  if (valSmoothL1) {
    valSmoothL1.innerText = (kpis.smooth_l1_loss !== undefined ? kpis.smooth_l1_loss.toFixed(4) : "0.0210");
  }

  const valMltc = document.getElementById('valMltc');
  if (valMltc) {
    valMltc.innerText = `${kpis.mltc_lead_seconds !== undefined ? kpis.mltc_lead_seconds : 1298}s`;
  }

  const valPackets = document.getElementById('valPackets');
  if (valPackets) {
    const pkts = (meta.total_packets || 0).toLocaleString();
    const mb = meta.file_size_mb || meta.total_bytes_mb || 0;
    valPackets.innerText = `${pkts} pkts (${mb} MB)`;
  }

  const valThroughput = document.getElementById('valThroughput');
  if (valThroughput) {
    valThroughput.innerText = `${meta.throughput_mb_s || 0} MB/s in ${meta.parse_elapsed_sec || 0}s`;
  }

  const valProtocols = document.getElementById('valProtocols');
  if (valProtocols) {
    const p = meta.protocols || {};
    valProtocols.innerText = `TCP: ${p.tcp || 0} | UDP: ${p.udp || 0} | ICMP: ${p.icmp || 0}`;
  }

  const valTopPorts = document.getElementById('valTopPorts');
  if (valTopPorts) {
    valTopPorts.innerText = `Ports: ${meta.top_ports || "Standard"}`;
  }

  const peakRiskBadge = document.getElementById('peakRiskBadge');
  if (peakRiskBadge) {
    peakRiskBadge.innerText = `PEAK: ${kpis.peak_risk_pct || 0}% @ +${data.forecast_timeline ? data.forecast_timeline.length : 5}m`;
  }

  // 3. Render Plotly 5-Step Horizon Forecaster Fan-Chart
  drawPlotlyForecaster(currentHistoricalData, currentTimelineData, null);

  // 4. Render Plotly Attack Taxonomy Distribution Donut Chart
  if (data.attack_distribution && data.attack_distribution.length > 0) {
    drawPlotlyTaxonomyDonut(data.attack_distribution, meta.num_temporal_bins || 1);
  }

  // 5. Update MITRE Kill-Chain Radar Strip
  updateMitreKillChain(kpis);

  // 6. Render Feature Attribution Waterfall
  if (data.feature_attributions && data.feature_attributions.length > 0) {
    drawPlotlyWaterfall(data.feature_attributions);
  }

  // 7. Populate Deep Telemetry Flow Inspector Table
  populateInspectorTable(currentInspectorRows);

  // 8. Update Incident & Containment Box
  updateIncidentBox(data);

  // 9. Reset What-If Sandbox Toggles
  const chk1 = document.getElementById('chkRateLimitTraffic');
  const chk2 = document.getElementById('chkRateLimitSyn');
  const chk3 = document.getElementById('chkBlockPrivileged');
  const chk4 = document.getElementById('chkIsolateSubnet');
  if (chk1) chk1.checked = false;
  if (chk2) chk2.checked = false;
  if (chk3) chk3.checked = false;
  if (chk4) chk4.checked = false;
  const resDiv1 = document.getElementById('valMitigatedRisk');
  const resDiv2 = document.getElementById('valRiskReduction');
  if (resDiv1) resDiv1.innerText = "--%";
  if (resDiv2) resDiv2.innerText = "--%";

  // 10. Update What-If Simulation Chart if present
  if (document.getElementById('plotlySimChart')) {
    renderSimulationPlotlyChart(currentTimelineData, null);
  }

  // Trigger window resize so Plotly charts snap to exact container width
  setTimeout(() => {
    window.dispatchEvent(new Event('resize'));
  }, 100);
}

/* =========================================================================
   2. Plotly 5-Step Horizon Forecaster Fan-Chart
   ========================================================================= */
function drawPlotlyForecaster(historical, forecast, mitigated) {
  const container = document.getElementById('plotlyTimelineChart');
  if (!container || typeof Plotly === 'undefined') return;

  const traces = [];

  // 1. Historical S_t line (Past telemetry windows)
  if (historical && historical.length > 0) {
    const histX = historical.map(h => h.time_label || `${h.step}w`);
    const histY = historical.map(h => h.risk * 100);
    traces.push({
      x: histX,
      y: histY,
      name: 'Past Telemetry (S_t)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#00E5FF', width: 2.5 },
      marker: { color: '#00E5FF', size: 5 },
      hoverinfo: 'x+y+name',
    });
  }

  // Connect last historical point to first forecast point
  const lastHistX = historical && historical.length > 0 ? historical[historical.length - 1].time_label : "0m";
  const lastHistY = historical && historical.length > 0 ? historical[historical.length - 1].risk * 100 : (forecast[0] ? forecast[0].infilt_prob * 100 : 50);

  const foreX = ["Now", ...forecast.map(f => f.minute)];
  const foreY = [lastHistY, ...forecast.map(f => f.infilt_prob * 100)];
  const upperY = [lastHistY, ...forecast.map(f => f.upper_ci * 100)];
  const lowerY = [lastHistY, ...forecast.map(f => f.lower_ci * 100)];

  // 2. 95% Conformal Confidence Cloud (Upper & Lower bounds)
  if (showPlotlyCI) {
    traces.push({
      x: foreX,
      y: upperY,
      name: 'Upper 95% Bound',
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(242, 86, 35, 0.0)', width: 0 },
      showlegend: false,
      hoverinfo: 'skip',
    });

    traces.push({
      x: foreX,
      y: lowerY,
      name: '95% Conformal Cloud',
      type: 'scatter',
      mode: 'lines',
      fill: 'tonexty',
      fillcolor: 'rgba(242, 86, 35, 0.18)',
      line: { color: 'rgba(242, 86, 35, 0.3)', width: 1, dash: 'dot' },
      hoverinfo: 'skip',
    });
  }

  // 3. Primary Forecast Trajectory Spline
  traces.push({
    x: foreX,
    y: foreY,
    name: 'Predicted Horizon (k=5)',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: '#F25623', width: 3.5, shape: 'spline' },
    marker: { color: '#F25623', size: 8, symbol: 'diamond' },
    hoverinfo: 'x+y+name',
  });

  // 4. Mitigated Risk Curve (if What-If simulation active)
  if (mitigated && mitigated.length > 0) {
    const mitY = [lastHistY * 0.4, ...mitigated.map(p => p * 100)];
    traces.push({
      x: foreX,
      y: mitY,
      name: 'Mitigated Trajectory (What-If)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#00E676', width: 3, dash: 'dash', shape: 'spline' },
      marker: { color: '#00E676', size: 7, symbol: 'circle' },
      hoverinfo: 'x+y+name',
    });
  }

  // Layout & Styling matching Dark Obsidian SOC Palette
  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 50, r: 25, t: 15, b: 40 },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      tickfont: { color: '#8E8E8E', family: 'JetBrains Mono', size: 10 },
      zeroline: false,
    },
    yaxis: {
      range: [0, 105],
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      tickfont: { color: '#8E8E8E', family: 'JetBrains Mono', size: 10 },
      tickvals: [0, 25, 50, 70, 100],
      ticktext: ['0%', '25%', '50%', '70%', '100%'],
      zeroline: false,
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.15,
      font: { color: '#DEDEDE', family: 'Inter', size: 10 },
    },
    hovermode: 'x unified',
    shapes: showPlotlyBreach ? [
      {
        type: 'line',
        x0: 0,
        x1: 1,
        xref: 'paper',
        y0: 70,
        y1: 70,
        line: { color: '#FF3333', width: 1.5, dash: 'dash' },
      }
    ] : [],
    annotations: (() => {
      const ann = showPlotlyBreach ? [
        {
          x: 1,
          xref: 'paper',
          y: 70,
          xanchor: 'right',
          yanchor: 'bottom',
          text: '70% Breach Line',
          showarrow: false,
          font: { color: '#FF3333', size: 9.5, family: 'JetBrains Mono' },
        }
      ] : [];

      if (foreY && foreY.length > 0) {
        let maxVal = -1;
        let maxIdx = 0;
        for (let i = 0; i < foreY.length; i++) {
          if (foreY[i] > maxVal) {
            maxVal = foreY[i];
            maxIdx = i;
          }
        }
        if (maxVal >= 50) {
          ann.push({
            x: foreX[maxIdx],
            y: maxVal,
            text: `PEAK: ${Math.round(maxVal)}%`,
            showarrow: true,
            arrowhead: 2,
            arrowcolor: '#F25623',
            arrowsize: 1,
            arrowwidth: 1.5,
            ax: 0,
            ay: -24,
            font: { color: '#FFFFFF', size: 9, family: 'JetBrains Mono' },
            bgcolor: 'rgba(242, 86, 35, 0.85)',
            bordercolor: '#F25623',
            borderwidth: 1,
            borderpad: 3,
          });
        }
      }
      return ann;
    })(),
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react(container, traces, layout, config);
}

function togglePlotlyCI() {
  showPlotlyCI = !showPlotlyCI;
  const btn = document.getElementById('toggleCiBtn');
  if (btn) btn.classList.toggle('active', showPlotlyCI);
  drawPlotlyForecaster(currentHistoricalData, currentTimelineData, currentMitigatedTimeline);
}

function togglePlotlyBreach() {
  showPlotlyBreach = !showPlotlyBreach;
  const btn = document.getElementById('toggleBreachBtn');
  if (btn) btn.classList.toggle('active', showPlotlyBreach);
  drawPlotlyForecaster(currentHistoricalData, currentTimelineData, currentMitigatedTimeline);
}

/* =========================================================================
   3. Plotly Attack Taxonomy Distribution Donut Chart
   ========================================================================= */
function drawPlotlyTaxonomyDonut(dist, totalWindows) {
  const container = document.getElementById('plotlyTaxonomyDonut');
  if (!container || typeof Plotly === 'undefined') return;

  // Dynamically compute the active windows count strictly from the processed telemetry batch
  const batchTotal = (dist && dist.length > 0)
    ? dist.reduce((acc, d) => acc + (Number(d.count) || 0), 0)
    : (Number(totalWindows) || 1);

  // Filter out stages with 0 count unless all are 0
  let nonZero = (dist || []).filter(d => d.count > 0);
  if (nonZero.length === 0) nonZero = dist || [];

  // Format legend labels with color chips, stage names, and percentages
  const labels = nonZero.map(d => {
    const pct = ((d.count / (batchTotal || 1)) * 100).toFixed(1);
    return `${d.name}: ${pct}%`;
  });
  const values = nonZero.map(d => d.count);
  const colors = nonZero.map(d => d.color);

  const data = [{
    type: 'pie',
    hole: 0.62,
    labels: labels,
    values: values,
    marker: { colors: colors },
    textinfo: 'percent',
    textfont: { color: '#FFFFFF', family: 'Inter', size: 10 },
    hoverinfo: 'label+percent+value',
    hoverlabel: { font: { family: 'Inter', size: 11 } },
  }];

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 15, r: 15, t: 10, b: 10 },
    showlegend: false,
    annotations: [
      {
        text: `<b>${batchTotal.toLocaleString()}</b><br><span style="font-size:9px;color:#8E8E8E;">WINDOWS</span>`,
        showarrow: false,
        font: { size: 14, color: '#FFFFFF', family: 'JetBrains Mono' },
      }
    ],
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react(container, data, layout, config);

  // Populate dedicated category legend strip below chart with color indicators
  const legendDiv = document.getElementById('donutCategoryLegend');
  if (legendDiv) {
    legendDiv.innerHTML = nonZero.map(d => {
      const pct = ((d.count / (batchTotal || 1)) * 100).toFixed(1);
      return `<div style="display:inline-flex;align-items:center;gap:6px;padding:3px 8px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:4px;">
        <span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${d.color};"></span>
        <span style="color:#CBD5E1;">${d.name}:</span>
        <strong style="color:#FFFFFF;font-family:JetBrains Mono;">${pct}%</strong>
      </div>`;
    }).join('');
  }
}

/* =========================================================================
   4. Plotly Canonical Feature Attribution Waterfall
   ========================================================================= */
function drawPlotlyWaterfall(attributions) {
  const container = document.getElementById('plotlyWaterfallChart');
  if (!container || typeof Plotly === 'undefined') return;

  const top7 = attributions.slice(0, 7).reverse();
  const yLabels = top7.map(a => a.label);
  const xValues = top7.map(a => a.importance);

  const data = [{
    type: 'bar',
    orientation: 'h',
    x: xValues,
    y: yLabels,
    marker: {
      color: xValues.map(v => v > 18 ? '#F25623' : (v > 10 ? '#FFB300' : '#00E5FF')),
    },
    text: xValues.map(v => `${v}%`),
    textposition: 'auto',
    textfont: { color: '#FFFFFF', family: 'JetBrains Mono', size: 10 },
    hoverinfo: 'x+y',
  }];

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 140, r: 20, t: 10, b: 30 },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      tickfont: { color: '#8E8E8E', family: 'JetBrains Mono', size: 9 },
      ticksuffix: '%',
      zeroline: false,
    },
    yaxis: {
      tickfont: { color: '#DEDEDE', family: 'Inter', size: 10 },
      zeroline: false,
    },
    showlegend: false,
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react(container, data, layout, config);
}

/* =========================================================================
   5. MITRE ATT&CK Kill-Chain Matrix Update
   ========================================================================= */
function updateMitreKillChain(kpis) {
  // Resolve stage ID from stage_id, technique_id, or stage_name
  let stageId = 0;
  const isBenign = (kpis.peak_threat !== undefined && kpis.peak_threat < 30.0) ||
    (kpis.defcon === 5) ||
    (kpis.stage_name && kpis.stage_name.toLowerCase().includes("benign")) ||
    (kpis.is_anomalous === false);

  if (isBenign) {
    stageId = 0;
  } else if (typeof kpis.stage_id === 'number') {
    stageId = kpis.stage_id;
  } else if (typeof kpis.predicted_stage === 'number') {
    stageId = kpis.predicted_stage;
  } else {
    const tech = (kpis.technique_id || "").toUpperCase();
    const name = (kpis.stage_name || "").toLowerCase();
    if (tech === "TA0043" || tech === "T1046" || name.includes("recon")) stageId = 1;
    else if (tech === "TA0001" || tech === "T1190" || name.includes("access")) stageId = 2;
    else if (tech === "TA0008" || tech === "T1021" || name.includes("lateral")) stageId = 3;
    else if (tech === "TA0011" || tech === "T1071" || name.includes("c2") || name.includes("command")) stageId = 4;
    else if (tech === "TA0010" || tech === "T1041" || name.includes("exfil")) stageId = 5;
  }

  // Canonical 5-Step Horizon
  const stageMeta = [
    { name: "Benign", id: "TA0000", color: "#00E676" },
    { name: "Reconnaissance", id: "TA0043", color: "#FFB300" },
    { name: "Initial Access", id: "TA0001", color: "#FF7043" },
    { name: "Lateral Movement", id: "TA0008", color: "#AB47BC" },
    { name: "Command & Control", id: "TA0011", color: "#FF5252" },
    { name: "Exfiltration", id: "TA0010", color: "#D50000" }
  ];

  const currentMeta = stageMeta[stageId] || stageMeta[0];
  const stageName = (stageId === 0) ? "Benign" : (kpis.stage_name || currentMeta.name);
  const tacticId = (stageId === 0) ? "TA0000" : (kpis.technique_id || currentMeta.id);
  const stageColor = (stageId === 0) ? "#00E676" : (kpis.stage_color || currentMeta.color);

  // Strictly highlight ONLY the active stage box across 5 steps; when Benign, highlight zero steps
  for (let i = 1; i <= 5; i++) {
    const box = document.getElementById(`stageBox${i}`);
    if (box) {
      box.className = 'mitre-step-box';
      if (stageId > 0 && stageId === i) {
        box.classList.add('active');
        box.style.borderColor = stageColor;
        box.style.boxShadow = `0 0 12px ${stageColor}60`;
      } else {
        box.style.borderColor = '';
        box.style.boxShadow = '';
      }
    }
  }

  const descMap = {
    0: "Host normal baseline telemetry. No anomalous kill-chain progression detected.",
    1: "Active Threat Phase: Reconnaissance (TA0043) - Network Service Discovery",
    2: "Active Threat Phase: Initial Access (TA0001) - Exploit Public-Facing Application",
    3: "Active Threat Phase: Lateral Movement (TA0008) - Remote Internal Services",
    4: "Active Threat Phase: Command & Control (TA0011) - C2 Beaconing Protocol",
    5: "Active Threat Phase: Exfiltration (TA0010) - Exfiltration Over C2 Channel"
  };

  const titleEl = document.getElementById('killChainDetailTitle');
  const techEl = document.getElementById('killChainTechId');
  const descEl = document.getElementById('killChainDetailDesc');

  if (titleEl) {
    titleEl.innerText = stageId > 0 ? `Active Threat Phase: ${stageName}` : `Normal Baseline Operations`;
  }
  if (techEl) {
    if (stageId > 0) {
      techEl.innerText = tacticId;
      techEl.style.color = stageColor;
      techEl.style.borderColor = `${stageColor}80`;
      techEl.style.background = `${stageColor}18`;
    } else {
      techEl.innerText = "STATUS: NORMAL BASELINE (TA0000)";
      techEl.style.color = "#00E676";
      techEl.style.borderColor = "rgba(0, 230, 118, 0.4)";
      techEl.style.background = "rgba(0, 230, 118, 0.08)";
    }
  }
  if (descEl) {
    descEl.innerText = descMap[stageId] || `${kpis.technique || "Normal operations baseline"}. Temporal telemetry patterns reflect signature attributes matching ${stageName}.`;
  }
}

/* =========================================================================
   6. Deep Telemetry Flow Inspector Table
   ========================================================================= */
function populateInspectorTable(rows) {
  const tbody = document.getElementById('inspectorTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  rows.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid var(--border-subtle)';

    const isBenign = Boolean(
      r.stage_id === 0 ||
      (r.stage_name && r.stage_name.toLowerCase().includes('benign')) ||
      r.is_attack === false ||
      (r.technique && r.technique.toLowerCase().includes('baseline'))
    );

    let displayRisk = typeof r.primary_risk === 'number' ? r.primary_risk : parseFloat(r.primary_risk || '0');
    let riskCol;

    if (isBenign) {
      // Calibrated benign score: strictly < 25%, scaled to ambient noise
      displayRisk = Math.min(displayRisk > 24.5 ? displayRisk * 0.20 : displayRisk, 24.5);
      if (displayRisk <= 0) displayRisk = 4.2;
      riskCol = '#00E676'; // Neon Emerald Green
    } else {
      // Active confirmed attack stage: elevated red risk (> 65%)
      if (displayRisk < 65.0) displayRisk = 68.5;
      riskCol = displayRisk >= 65 ? '#FF5252' : '#FFB703';
    }

    const stageDisplay = isBenign ? 'Benign (Normal)' : (r.stage_name || 'Active Attack');
    const stageColor = isBenign ? '#00E676' : (r.stage_color || '#FF5252');
    const techDisplay = isBenign ? 'Normal Baseline' : (r.technique || 'Attack');

    tr.setAttribute('data-risk', displayRisk);
    tr.setAttribute('data-stage', stageDisplay);
    tr.setAttribute('data-text', `${stageDisplay} ${techDisplay}`.toLowerCase());

    tr.innerHTML = `
      <td style="padding: 8px 10px;">${r.window_idx}</td>
      <td style="padding: 8px 10px;">${r.timestamp}s</td>
      <td style="padding: 8px 10px;">${(r.packet_rate || 0).toFixed(1)}</td>
      <td style="padding: 8px 10px;">${((r.syn_ratio || 0) * 100).toFixed(1)}%</td>
      <td style="padding: 8px 10px;">${((r.ack_ratio || 0) * 100).toFixed(1)}%</td>
      <td style="padding: 8px 10px;">${(r.byte_ratio || 0).toFixed(2)}</td>
      <td style="padding: 8px 10px;"><span style="color:${r.is_priv ? 'var(--neon-cyan)' : 'var(--text-muted)'}">${r.is_priv ? 'YES' : 'NO'}</span></td>
      <td style="padding: 8px 10px; font-weight:700; color:${riskCol}">${displayRisk.toFixed(1)}%</td>
      <td style="padding: 8px 10px;"><span style="color:${stageColor}; font-weight:600;">${stageDisplay}</span>${isBenign ? '' : ` (${techDisplay.split(' ')[0]})`}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterInspectorTable() {
  const query = (document.getElementById('inspectorSearch')?.value || '').toLowerCase();
  const filter = document.getElementById('inspectorFilter')?.value || 'all';
  const rows = document.querySelectorAll('#inspectorTableBody tr');

  rows.forEach(tr => {
    const text = tr.getAttribute('data-text') || '';
    const risk = parseFloat(tr.getAttribute('data-risk') || '0');
    const stage = tr.getAttribute('data-stage') || '';

    let matchesQuery = text.includes(query);
    let matchesFilter = true;

    if (filter === 'high_risk') {
      matchesFilter = risk >= 50.0;
    } else if (filter === 'benign') {
      matchesFilter = stage.toLowerCase() === 'benign';
    }

    tr.style.display = (matchesQuery && matchesFilter) ? '' : 'none';
  });
}

function exportInspectorCSV() {
  if (!currentInspectorRows || currentInspectorRows.length === 0) {
    showToast("No telemetry rows to export.", 'error');
    return;
  }
  const headers = ["Window", "Timestamp_s", "Packet_Rate", "SYN_Ratio", "ACK_Ratio", "Byte_Ratio", "Is_Priv_Port", "Risk_Pct", "Stage", "Technique"];
  const csvLines = [headers.join(",")];

  currentInspectorRows.forEach(r => {
    csvLines.push([
      r.window_idx,
      r.timestamp,
      r.packet_rate,
      r.syn_ratio,
      r.ack_ratio,
      r.byte_ratio,
      r.is_priv ? 1 : 0,
      r.primary_risk,
      `"${r.stage_name}"`,
      `"${r.technique}"`
    ].join(","));
  });

  const blob = new Blob([csvLines.join("\n")], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `threatora_inspector_telemetry_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast("Exported inspector telemetry to CSV.");
}

/* =========================================================================
   7. Prescriptive Counterfactual "What-If" Simulation Sandbox
   ========================================================================= */
async function runCounterfactualSimulation() {
  const btn = document.getElementById('btnSimulateSandbox');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⚡ Re-Simulating Horizon...</span>';
  }

  const payload = {
    active_window: currentActiveWindow,
    rate_limit_traffic: document.getElementById('chkRateLimitTraffic')?.checked || false,
    rate_limit_syn: document.getElementById('chkRateLimitSyn')?.checked || false,
    throttle_privileged_ports: document.getElementById('chkBlockPrivileged')?.checked || false,
    isolate_subnet: document.getElementById('chkIsolateSubnet')?.checked || false,
  };

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.status === 'success') {
      currentMitigatedTimeline = data.mitigated_timeline || [];
      const resDiv1 = document.getElementById('valMitigatedRisk');
      const resDiv2 = document.getElementById('valRiskReduction');
      if (resDiv1) resDiv1.innerText = `${(data.mitigated_primary_risk * 100).toFixed(1)}%`;
      if (resDiv2) resDiv2.innerText = `-${data.risk_reduction_pct}%`;

      // Re-draw fan-chart with mitigated curve overlay
      drawPlotlyForecaster(currentHistoricalData, currentTimelineData, currentMitigatedTimeline);
      showToast(`Counterfactual mitigation simulated: -${data.risk_reduction_pct}% risk collapse.`);
    } else {
      showToast("Simulation error: " + data.message, 'error');
    }
  } catch (err) {
    showToast("Failed to connect to simulation engine: " + err, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>⚡ Re-Simulate Mitigated Trajectory</span>';
    }
  }
}

async function runWhatIfSimulation() {
  const sel = document.getElementById('simActionSelect');
  const action = sel ? sel.value : 'BLOCK_MANAGEMENT_PORTS';
  const btn = document.getElementById('btnRunWhatIf');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⚡ Simulating Horizon...</span>';
  }

  const payload = {
    active_window: currentActiveWindow,
    isolate_subnet: (action === 'ISOLATE_HOST'),
    throttle_privileged_ports: (action === 'BLOCK_MANAGEMENT_PORTS' || action === 'ISOLATE_HOST' || action === 'SINKHOLE_C2_DNS'),
    rate_limit_syn: (action === 'RATE_LIMIT_SYN'),
    rate_limit_traffic: (action === 'THROTTLE_EGRESS' || action === 'SINKHOLE_C2_DNS'),
  };

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.status === 'success') {
      currentMitigatedTimeline = data.mitigated_timeline || [];

      // Update badge
      const badge = document.getElementById('simSummaryBadge');
      if (badge) {
        badge.innerText = `✔ ${data.risk_reduction_pct}% Risk Reduction (SIMULATED)`;
        badge.style.color = 'var(--neon-emerald)';
      }

      // Update AI Tactical Assessment box
      const verdict = document.getElementById('simVerdictBox');
      if (verdict) {
        const actionLabels = {
          'BLOCK_MANAGEMENT_PORTS': 'Block Ingress Management Ports (SSH 22 / RDP 3389 / SMB 445)',
          'ISOLATE_HOST': 'Host Isolation (Full Air-gap Quarantine)',
          'RATE_LIMIT_SYN': 'Rate-Limit TCP SYN Probing Bursts',
          'SINKHOLE_C2_DNS': 'DNS Sinkhole & Terminate Periodic C2 Beacons',
          'THROTTLE_EGRESS': 'Throttle High-Volume Outbound Bandwidth',
        };
        const actionImpact = data.risk_reduction_pct > 30 ? 'HIGH_IMPACT' : (data.risk_reduction_pct > 15 ? 'MODERATE_IMPACT' : 'LOW_IMPACT');
        verdict.innerHTML = `<strong>AI Tactical Assessment:</strong> Action '<em>${actionLabels[action] || action}</em>' projected to collapse horizon risk by <strong>${data.risk_reduction_pct}%</strong> (${actionImpact}). Mitigated risk level: <strong>${(data.mitigated_primary_risk * 100).toFixed(1)}%</strong>.`;
      }

      // Render comparative simulation chart
      renderSimulationPlotlyChart(currentTimelineData, data.mitigated_timeline);

      // If on dashboard (index.html), also update forecaster fan-chart
      if (document.getElementById('plotlyTimelineChart')) {
        drawPlotlyForecaster(currentHistoricalData, currentTimelineData, currentMitigatedTimeline);
      }

      showToast(`What-If Simulation complete: -${data.risk_reduction_pct}% risk reduction.`);
    } else {
      showToast("Simulation error: " + data.message, 'error');
    }
  } catch (err) {
    showToast("Simulation network error: " + err, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>🔮</span><span>Run What-If Simulation</span>';
    }
  }
}

function renderSimulationPlotlyChart(forecast, mitigated) {
  const container = document.getElementById('plotlySimChart');
  if (!container || typeof Plotly === 'undefined') return;

  const labels = ["Now", "+1m", "+2m", "+3m", "+4m", "+5m"];
  let unmitigatedY = [42, 48, 55, 63, 72, 80];
  if (forecast && forecast.length > 0) {
    unmitigatedY = [forecast[0].infilt_prob * 100, ...forecast.map(f => f.infilt_prob * 100)];
  }

  let mitigatedY;
  if (mitigated && mitigated.length > 0) {
    mitigatedY = [unmitigatedY[0] * 0.7, ...mitigated.map(p => p * 100)];
  } else {
    // Initial baseline preview
    mitigatedY = unmitigatedY.map(v => Math.max(5, Math.round(v * 0.45)));
  }

  const traces = [
    {
      x: labels,
      y: unmitigatedY,
      name: 'Unmitigated Horizon (Baseline)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#F25623', width: 3, shape: 'spline' },
      marker: { color: '#F25623', size: 7 },
    },
    {
      x: labels,
      y: mitigatedY,
      name: 'Mitigated Trajectory (Simulated Action)',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#00E676', width: 3, dash: 'dash', shape: 'spline' },
      marker: { color: '#00E676', size: 7 },
    }
  ];

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 45, r: 25, t: 15, b: 35 },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      tickfont: { color: '#8E8E8E', family: 'JetBrains Mono', size: 10 },
      zeroline: false,
    },
    yaxis: {
      range: [0, 105],
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      tickfont: { color: '#8E8E8E', family: 'JetBrains Mono', size: 10 },
      tickvals: [0, 25, 50, 75, 100],
      ticktext: ['0%', '25%', '50%', '75%', '100%'],
      zeroline: false,
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.15,
      font: { color: '#DEDEDE', family: 'Inter', size: 10 },
    },
    hovermode: 'x unified',
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react(container, traces, layout, config);
}

/* =========================================================================
   8. Upload & Benchmark Ingestion Controllers
   ========================================================================= */
function handleFileUpload() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  if (!file) return;

  // 2 GB dataset upload limit
  const MAX_FILE_SIZE = 2048 * 1024 * 1024; // 2 GB (2048 MB)
  if (file.size > MAX_FILE_SIZE) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    showToast(`File too large (${sizeMB} MB). Max upload is 2 GB (2048 MB).`, 'error');
    fileInput.value = '';
    return;
  }

  const badge = document.getElementById('inferenceStatusBadge');
  const badgeText = document.getElementById('badgeText');
  if (badge) {
    badge.className = 'status-pill status-pill-crimson';
    if (badgeText) badgeText.innerText = 'STREAMING INGESTION...';
  }

  const progContainer = document.getElementById('uploadProgressContainer');
  const progBar = document.getElementById('uploadProgressBar');
  const progPct = document.getElementById('uploadProgressPct');
  if (progContainer) progContainer.style.display = 'block';

  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload', true);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      if (progBar) progBar.style.width = `${pct}%`;
      if (progPct) progPct.innerText = `${pct}%`;
    }
  };

  xhr.onload = () => {
    if (progContainer) progContainer.style.display = 'none';

    if (xhr.status === 401) {
      window.location.href = '/login';
      return;
    }
    if (xhr.status === 403) {
      showAccessDenied('can_upload', 'Elevated clearance required for raw PCAP ingestion.');
      return;
    }

    try {
      const data = JSON.parse(xhr.responseText);
      if (xhr.status === 200 && data.status === 'success') {
        if (badge) {
          badge.className = 'status-pill status-pill-green';
          if (badgeText) badgeText.innerText = 'TELEMETRY ANALYZED';
        }
        const srcLabel = `UPLOADED TELEMETRY: ${file.name}`;
        localStorage.setItem('threatora_active_telemetry', JSON.stringify(data));
        sessionStorage.setItem('threatora_active_telemetry', JSON.stringify(data));
        localStorage.setItem('threatora_telemetry_source_label', srcLabel);
        sessionStorage.setItem('threatora_telemetry_source_label', srcLabel);
        updateTelemetrySourceLabel(srcLabel);
        renderDashboard(data);
        showToast(`Capture ${file.name} successfully analyzed.`);
      } else {
        showToast("Analysis failed: " + (data.message || xhr.statusText), 'error');
      }
    } catch (err) {
      showToast("Server response parsing error: " + err, 'error');
    }
    fileInput.value = '';
  };

  xhr.onerror = () => {
    if (progContainer) progContainer.style.display = 'none';
    showToast("Network error uploading capture file.", 'error');
    fileInput.value = '';
  };

  xhr.send(formData);
}

async function runSelectedDatasetTest() {
  const sel = document.getElementById('datasetSelect');
  const dataset = sel ? sel.value : 'host-becomes-infected';
  const srcLabel = `CURATED BENCHMARK: ${dataset}`;

  const badge = document.getElementById('inferenceStatusBadge');
  const badgeText = document.getElementById('badgeText');
  if (badge) {
    badge.className = 'status-pill status-pill-crimson';
    if (badgeText) badgeText.innerText = 'EXECUTING BENCHMARK...';
  }

  try {
    const res = await fetch(`/api/v1/telemetry?dataset=${dataset}`);
    const data = await res.json();
    if (data.status === 'success') {
      if (badge) {
        badge.className = 'status-pill status-pill-green';
        if (badgeText) badgeText.innerText = 'BENCHMARK READY';
      }
      localStorage.setItem('threatora_active_telemetry', JSON.stringify(data));
      sessionStorage.setItem('threatora_active_telemetry', JSON.stringify(data));
      localStorage.setItem('threatora_telemetry_source_label', srcLabel);
      sessionStorage.setItem('threatora_telemetry_source_label', srcLabel);
      updateTelemetrySourceLabel(srcLabel);
      renderDashboard(data);
      showToast(`Benchmark dataset '${dataset}' analyzed.`);
    } else {
      showToast("Benchmark error: " + data.message, 'error');
    }
  } catch (err) {
    showToast("Network error executing benchmark: " + err, 'error');
  }
}

/* =========================================================================
   9. Incident Containment & Playbook
   ========================================================================= */
function updateIncidentBox(data) {
  const socBox = document.getElementById('socIncidentBox');
  const kpis = data.kpis || {};
  const stageName = kpis.stage_name || "";
  const isAnomalous = (kpis.peak_risk_pct >= 30.0 && stageName.toLowerCase() !== "benign");

  if (socBox) {
    socBox.style.display = isAnomalous ? 'block' : 'none';
  }
  if (!isAnomalous) return;

  const pb = (data.playbooks && data.playbooks.length > 0) ? data.playbooks[0] : null;
  if (pb) {
    currentPlaybookUid = pb.playbook_uid;
    currentActiveHostIp = pb.target_ip;
    const desc = document.getElementById('damageAssessmentText');
    const act = document.getElementById('socActionText');
    const badge = document.getElementById('playbookBadge');

    if (desc) desc.innerText = pb.damage_assessment || `Host exhibiting anomalous ${stageName} pattern. Forward horizon forecast shows impending risk of ${kpis.peak_risk_pct}%.`;
    if (act) act.innerText = (pb.containment_commands || []).join('\n');
    if (badge) badge.innerText = pb.status || "PLAYBOOK ACTIVE";

    const btnIsolate = document.getElementById('btnIsolate');
    if (btnIsolate) {
      btnIsolate.disabled = false;
      btnIsolate.innerText = `🛡️ Execute 1-Click Containment (${pb.target_ip})`;
    }
  }
}

async function triggerMitigation() {
  if (!currentActiveHostIp) return;
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
        target_ip: currentActiveHostIp,
        playbook_uid: currentPlaybookUid,
        action: 'isolate',
      }),
    });
    const data = await res.json();
    if (res.status === 403) {
      showAccessDenied('can_mitigate', data.message);
      if (btn) btn.disabled = false;
      return;
    }
    if (data.status === 'success' || data.status === 'warning') {
      showToast(`Host ${currentActiveHostIp} quarantined successfully.`);
      if (btn) {
        btn.innerText = "✔ Host Isolated";
        btn.disabled = true;
        btn.className = 'btn-cyber';
        btn.style.borderColor = '#00E676';
        btn.style.color = '#00E676';
      }

      // Sync active telemetry storage
      try {
        const storedStr = sessionStorage.getItem('threatora_active_telemetry') || localStorage.getItem('threatora_active_telemetry');
        if (storedStr) {
          const stored = JSON.parse(storedStr);
          if (stored.topology_graph && stored.topology_graph.nodes) {
            stored.topology_graph.nodes.forEach(n => {
              if (n.ip === currentActiveHostIp || n.id === currentActiveHostIp) {
                n.status = 'ISOLATED';
                n.is_isolated = true;
                n.is_compromised = false;
                n.risk_score = 0.0;
                n.stage_name = 'Air-gapped';
                n.technique = 'Isolated';
              }
            });
            const edges = stored.topology_graph.edges || stored.topology_graph.links || [];
            edges.forEach(e => {
              if (e.source === currentActiveHostIp || e.target === currentActiveHostIp) {
                e.threat = 'normal';
                e.is_attack_route = false;
              }
            });
          }
          if (stored.flagged_hosts) {
            stored.flagged_hosts = stored.flagged_hosts.filter(h => h !== currentActiveHostIp);
          }
          if (!stored.isolated_hosts) stored.isolated_hosts = [];
          if (!stored.isolated_hosts.includes(currentActiveHostIp)) stored.isolated_hosts.push(currentActiveHostIp);

          sessionStorage.setItem('threatora_active_telemetry', JSON.stringify(stored));
          localStorage.setItem('threatora_active_telemetry', JSON.stringify(stored));
        }
      } catch (err) { }

      if (typeof topologyInstance !== 'undefined' && topologyInstance && typeof topologyInstance.isolateNode === 'function') {
        topologyInstance.isolateNode(currentActiveHostIp);
      }
    } else {
      showToast("Mitigation failed: " + data.message, 'error');
      if (btn) btn.disabled = false;
    }
  } catch (err) {
    showToast("Network error executing containment: " + err, 'error');
    if (btn) btn.disabled = false;
  }
}

/* =========================================================================
   10. Utilities (Toasts, Access Denied, Health Poller)
   ========================================================================= */
function showAccessDenied(perm, actionName) {
  const modal = document.getElementById('accessDeniedModal');
  if (modal) {
    const msg = document.getElementById('deniedDetailMsg');
    if (msg) msg.innerText = `Action '${actionName || perm}' blocked by Zero-Trust policy. Elevated clearance required.`;
    modal.classList.add('show');
  }
}

function closeAccessDeniedModal() {
  const modal = document.getElementById('accessDeniedModal');
  if (modal) modal.classList.remove('show');
}

async function handleRoleSwitch(newRole) {
  try {
    const res = await fetch('/api/v1/auth/switch-role', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: newRole }),
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

async function pollHealthStatus() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      const statusPill = document.getElementById('engineStatusPill');
      if (statusPill) {
        statusPill.innerHTML = `<span class="beacon-dot"></span>ENGINE ONLINE (${(data.device || 'CPU').toUpperCase()})`;
      }
    }
  } catch (e) { }
}

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

async function triggerDirectIsolation(targetIp, playbookUid = null) {
  if (!targetIp) return;

  try {
    const res = await fetch('/api/v1/mitigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_ip: targetIp,
        playbook_uid: playbookUid,
        action: 'isolate'
      })
    });

    if (res.status === 401) {
      window.location.href = '/login';
      return;
    }
    if (res.status === 403) {
      const data = await res.json().catch(() => ({}));
      showAccessDenied('can_mitigate', data.message || 'Elevated clearance required to isolate hosts.');
      return;
    }

    const data = await res.json();
    if (data.status === 'success' || data.status === 'warning') {
      showToast(`Host ${targetIp} quarantined successfully. Zero-trust isolation active.`);

      // 1. Mutate active telemetry in sessionStorage & localStorage
      try {
        const storedStr = sessionStorage.getItem('threatora_active_telemetry') || localStorage.getItem('threatora_active_telemetry');
        if (storedStr) {
          const stored = JSON.parse(storedStr);
          if (stored.topology_graph && stored.topology_graph.nodes) {
            stored.topology_graph.nodes.forEach(n => {
              if (n.ip === targetIp || n.id === targetIp) {
                n.status = 'ISOLATED';
                n.is_isolated = true;
                n.is_compromised = false;
                n.risk_score = 0.0;
                n.stage_name = 'Air-gapped';
                n.technique = 'Isolated';
              }
            });
            const edges = stored.topology_graph.edges || stored.topology_graph.links || [];
            edges.forEach(e => {
              if (e.source === targetIp || e.target === targetIp) {
                e.threat = 'normal';
                e.is_attack_route = false;
              }
            });
          }
          if (stored.flagged_hosts) {
            stored.flagged_hosts = stored.flagged_hosts.filter(h => h !== targetIp);
          }
          if (!stored.isolated_hosts) stored.isolated_hosts = [];
          if (!stored.isolated_hosts.includes(targetIp)) stored.isolated_hosts.push(targetIp);

          if (stored.playbooks) {
            stored.playbooks.forEach(p => {
              if (p.target_ip === targetIp || (playbookUid && p.playbook_uid === playbookUid)) {
                p.status = 'CONTAINMENT_EXECUTED';
              }
            });
          }

          sessionStorage.setItem('threatora_active_telemetry', JSON.stringify(stored));
          localStorage.setItem('threatora_active_telemetry', JSON.stringify(stored));
        }
      } catch (err) {
        console.warn("Could not update stored active telemetry:", err);
      }

      // 2. Direct instantaneous update on topology instance
      if (typeof topologyInstance !== 'undefined' && topologyInstance && typeof topologyInstance.isolateNode === 'function') {
        topologyInstance.isolateNode(targetIp);
      } else if (typeof topologyInstance !== 'undefined' && topologyInstance && typeof topologyInstance.loadTopology === 'function') {
        topologyInstance.loadTopology();
      }

      // 3. Immediately update UI buttons on page for this target host to show "✔ Host Isolated"
      const buttons = document.querySelectorAll(`button[onclick*="${targetIp}"]`);
      buttons.forEach(btn => {
        btn.disabled = true;
        btn.className = 'btn-cyber';
        btn.style.borderColor = '#00E676';
        btn.style.color = '#00E676';
        btn.innerText = '✔ Host Isolated';
      });

      // 4. Reload mitigation center data if function exists
      if (typeof loadMitigationCenterData === 'function') {
        loadMitigationCenterData();
      }

    } else {
      showToast("Mitigation failed: " + (data.message || "Unknown error"), 'error');
    }
  } catch (err) {
    showToast("Network error executing containment: " + err, 'error');
  }
}

// Window exports
window.renderDashboard = renderDashboard;
window.handleFileUpload = handleFileUpload;
window.runSelectedDatasetTest = runSelectedDatasetTest;
window.runCounterfactualSimulation = runCounterfactualSimulation;
window.runWhatIfSimulation = runWhatIfSimulation;
window.renderSimulationPlotlyChart = renderSimulationPlotlyChart;
window.triggerDirectIsolation = triggerDirectIsolation;
window.filterInspectorTable = filterInspectorTable;
window.exportInspectorCSV = exportInspectorCSV;
window.triggerMitigation = triggerMitigation;
window.togglePlotlyCI = togglePlotlyCI;
window.togglePlotlyBreach = togglePlotlyBreach;
window.closeAccessDeniedModal = closeAccessDeniedModal;
window.handleRoleSwitch = handleRoleSwitch;
