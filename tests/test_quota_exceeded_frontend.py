"""Frontend and integration verification for QuotaExceededError fix.

Verifies:
1. Short (13 windows) and Large (350+ windows) telemetry payloads:
   - Successfully uploaded and parsed via /api/upload.
2. Zero Browser Storage Errors:
   - Full unconstrained telemetry data is preserved in-memory on `window.threatoraTelemetry`.
   - Caching to persistent storage (`localStorage`/`sessionStorage`) stores only compact metadata (< 50 KB).
   - When localStorage throws QuotaExceededError, cacheActiveTelemetry safely catches it.
   - Telemetry upload handler passes parsed JSON directly to rendering functions
     (renderHUD, renderInspectorTable, renderForecastingCurve, renderTopology) from memory.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
import pandas as pd
import pytest

from server import create_app


def create_synthetic_tabular_df(num_windows: int, is_attack: bool = False) -> pd.DataFrame:
    """Generates synthetic tabular flow telemetry with exact row/window count."""
    rows = []
    base_t = 1628596800.0
    for i in range(num_windows):
        is_att_row = is_attack and (i >= num_windows // 2)
        rows.append({
            "window_id": i + 1,
            "timestamp": base_t + (i * 10.0),
            "byte_rate": 85000.0 if is_att_row else 5200.0,
            "packet_rate": 450.0 if is_att_row else 28.0,
            "mean_iat": 0.002 if is_att_row else 0.045,
            "syn_flag_ratio": 0.65 if is_att_row else 0.08,
            "ack_flag_ratio": 0.35 if is_att_row else 0.92,
            "fin_flag_ratio": 0.0,
            "rst_flag_ratio": 0.15 if is_att_row else 0.0,
            "psh_flag_ratio": 0.45 if is_att_row else 0.10,
            "bwd_to_fwd_ratio": 0.25 if is_att_row else 1.20,
            "is_privileged_port": 1 if is_att_row else 0,
            "label": 1 if is_att_row else 0,
            "mitre_stage": "Exfiltration" if is_att_row else "Benign",
            "technique_id": "T1048" if is_att_row else "TA0000",
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def test_client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_01_backend_short_and_large_telemetry_ingestion(test_client):
    """Verify backend ingestion handles both short (13 windows) and large (350+ windows) CSVs."""
    headers = {"X-API-Key": "threatora-zero-trust"}

    # 1. Short telemetry capture: exactly 13 windows
    df_short = create_synthetic_tabular_df(13, is_attack=False)
    csv_bytes_short = df_short.to_csv(index=False).encode("utf-8")
    res_short = test_client.post(
        "/api/upload",
        data={"file": (io.BytesIO(csv_bytes_short), "short_13_windows.csv")},
        content_type="multipart/form-data",
        headers=headers,
    )
    assert res_short.status_code == 200
    data_short = res_short.get_json()
    assert data_short["status"] == "success"
    assert data_short["meta"]["num_temporal_bins"] == 13
    assert len(data_short["inspector_rows"]) == 13

    # 2. Large telemetry capture: exactly 350 windows
    df_large = create_synthetic_tabular_df(350, is_attack=True)
    csv_bytes_large = df_large.to_csv(index=False).encode("utf-8")
    res_large = test_client.post(
        "/api/upload",
        data={"file": (io.BytesIO(csv_bytes_large), "large_350_windows.csv")},
        content_type="multipart/form-data",
        headers=headers,
    )
    assert res_large.status_code == 200
    data_large = res_large.get_json()
    assert data_large["status"] == "success"
    assert data_large["meta"]["num_temporal_bins"] == 350
    assert len(data_large["inspector_rows"]) > 0


def test_02_frontend_storage_quota_and_in_memory_rendering(test_client):
    """Test frontend JavaScript under QuotaExceededError condition using headless browser runner."""
    headers = {"X-API-Key": "threatora-zero-trust"}

    # Retrieve short and large payloads
    df_13 = create_synthetic_tabular_df(13, is_attack=False)
    res_13 = test_client.post(
        "/api/upload",
        data={"file": (io.BytesIO(df_13.to_csv(index=False).encode()), "short_13_windows.csv")},
        content_type="multipart/form-data",
        headers=headers,
    ).get_json()

    df_350 = create_synthetic_tabular_df(350, is_attack=True)
    res_350 = test_client.post(
        "/api/upload",
        data={"file": (io.BytesIO(df_350.to_csv(index=False).encode()), "large_350_windows.csv")},
        content_type="multipart/form-data",
        headers=headers,
    ).get_json()

    repo_root = Path(__file__).resolve().parent.parent
    js_path = repo_root / "static" / "js" / "cyber_dashboard.js"
    assert js_path.exists()
    js_content = js_path.read_text(encoding="utf-8")

    # Construct headless test HTML file
    test_html_path = repo_root / "data" / "processed" / "test_storage_runner.html"
    test_html_path.parent.mkdir(parents=True, exist_ok=True)

    json_13_str = json.dumps(res_13)
    json_350_str = json.dumps(res_350)

    html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Storage Quota Test</title></head>
<body>
  <div id="testOutput">RUNNING</div>
  <div id="awaitingTelemetryBanner" style="display:block;"></div>
  <div id="activeDashboardContainer" style="display:none;"></div>
  <div id="defconVal"></div>
  <div id="defconStatus"></div>
  <div id="defconDot"></div>
  <div id="defconBadge"></div>
  <div id="valRisk"></div>
  <div id="valRiskSub"></div>
  <div id="valStage"></div>
  <div id="valTechnique"></div>
  <div id="valSmoothL1"></div>
  <div id="valMltc"></div>
  <div id="valPackets"></div>
  <div id="valThroughput"></div>
  <div id="valProtocols"></div>
  <div id="valTopPorts"></div>
  <div id="peakRiskBadge"></div>
  <div id="socIncidentBox"></div>
  <div id="incidentStagePill"></div>
  <div id="incidentTechniqueId"></div>
  <div id="incidentHostTarget"></div>
  <div id="incidentAttackHorizon"></div>
  <div id="incidentLeadTime"></div>
  <div id="playbooksList"></div>
  <div id="samplePacketsList"></div>
  <div id="toastContainer"></div>
  <div id="plotlyTimelineChart"></div>
  <div id="plotlyTaxonomyChart"></div>
  <div id="plotlyWaterfallChart"></div>
  <div id="plotlySimChart"></div>
  <div id="radarStage0"></div>
  <div id="radarStage1"></div>
  <div id="radarStage2"></div>
  <div id="radarStage3"></div>
  <div id="radarStage4"></div>
  <div id="radarStage5"></div>
  <table id="inspectorTable"><tbody id="inspectorTableBody"></tbody></table>

  <script>
    // Stub Plotly
    window.Plotly = {{
      react: function() {{}},
      newPlot: function() {{}},
      relayout: function() {{}}
    }};

    // Stub CyberNetworkTopology
    window.CyberNetworkTopology = function() {{
      return {{
        loadTopology: function() {{ window.topologyLoaded = true; }},
        setData: function(d) {{ window.topologyDataSet = d; window.topologyLoaded = true; }},
        isolateNode: function() {{}}
      }};
    }};
  </script>

  <script>
    {js_content}
  </script>

  <script>
    window.testResults = [];
    function record(name, pass, msg) {{
      window.testResults.push({{ name: name, pass: pass, msg: msg || '' }});
    }}

    try {{
      const shortData = {json_13_str};
      const largeData = {json_350_str};

      // 1. Verify in-memory storage of short data (13 windows)
      cacheActiveTelemetry(shortData);
      record("short_in_memory", window.threatoraTelemetry === shortData, "Short data set to window.threatoraTelemetry");
      
      // Check compact storage payload size
      const shortStored = sessionStorage.getItem('threatora_active_telemetry');
      record("short_stored_present", !!shortStored, "Compact short data cached in storage");
      if (shortStored) {{
        const shortSizeKB = shortStored.length / 1024;
        record("short_size_bounded", shortSizeKB < 50.0, "Short storage size: " + shortSizeKB.toFixed(2) + " KB (< 50 KB)");
      }}

      // 2. Render short data from memory
      renderDashboard(shortData);
      const inspectorTbody = document.getElementById('inspectorTableBody');
      record("short_inspector_rendered", inspectorTbody.children.length === 13, "Rendered 13 inspector rows from memory");
      record("short_hud_rendered", document.getElementById('valPackets').innerText.includes("pkts"), "Rendered short HUD");

      // 3. SIMULATE QuotaExceededError in localStorage & sessionStorage
      const originalSetItem = Storage.prototype.setItem;
      Storage.prototype.setItem = function(key, val) {{
        throw new DOMException("The quota has been exceeded.", "QuotaExceededError");
      }};

      let quotaThrown = false;
      try {{
        // Caching under QuotaExceededError must NOT throw!
        cacheActiveTelemetry(largeData);
      }} catch (e) {{
        quotaThrown = true;
      }}
      record("quota_exceeded_handled", !quotaThrown, "cacheActiveTelemetry safely caught QuotaExceededError without throwing");
      record("large_in_memory_preserved", window.threatoraTelemetry === largeData, "Large data (350 windows) preserved in memory");

      // 4. Render large data directly from memory under QuotaExceededError
      let renderThrown = false;
      try {{
        renderDashboard(largeData);
      }} catch (e) {{
        renderThrown = true;
      }}
      record("large_render_success", !renderThrown, "renderDashboard succeeded without depending on localStorage");
      record("large_inspector_rendered", inspectorTbody.children.length > 0, "Rendered large inspector rows from memory");
      record("large_hud_defcon", document.getElementById('defconVal').innerText.includes("DEFCON"), "DEFCON HUD rendered");

      // Restore Storage
      Storage.prototype.setItem = originalSetItem;

      // Check compact storage payload for large dataset is strictly bounded (< 50 KB)
      cacheActiveTelemetry(largeData);
      const largeStored = sessionStorage.getItem('threatora_active_telemetry');
      if (largeStored) {{
        const largeSizeKB = largeStored.length / 1024;
        record("large_size_bounded", largeSizeKB < 50.0, "Large storage size: " + largeSizeKB.toFixed(2) + " KB (< 50 KB)");
      }}

      // Check all tests passed
      const allPassed = window.testResults.every(t => t.pass);
      document.getElementById('testOutput').innerText = allPassed ? "ALL_TESTS_PASSED" : "TESTS_FAILED: " + JSON.stringify(window.testResults);
    }} catch (globalErr) {{
      document.getElementById('testOutput').innerText = "FATAL_ERROR: " + globalErr;
    }}
  </script>
</body>
</html>
"""
    test_html_path.write_text(html_content, encoding="utf-8")

    # Execute headless Edge browser
    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    file_url = test_html_path.as_uri()

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--dump-dom",
        file_url,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=15)
        output = proc.stdout or ""

        assert "ALL_TESTS_PASSED" in output, f"Headless browser test failed. DOM Output:\n{output[:1500]}"
    finally:
        if test_html_path.exists():
            try:
                test_html_path.unlink()
            except Exception:
                pass
