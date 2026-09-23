/**
 * Threatora Defense Terminal Client Controller
 * NTRO Problem Statement 26153 // Zero-Trust RBAC & 2GB Chunked Ingestion
 */

// State Management
const STATE = {
  currentRole: 'Analyst (Level 3)',
  roleLevel: 3,
  username: 'Alex Kelly (@admin)',
  isUploading: false,
  sseSource: null,
  pendingAction: null,
};

// DOM Elements
let dom = {};

window.addEventListener('DOMContentLoaded', () => {
  dom = {
    userChipText: document.getElementById('userChipText'),
    btnDatasetBurst: document.getElementById('btnDatasetBurst'),
    chunkFileInput: document.getElementById('chunkFileInput'),
    btnIngestDataset: document.getElementById('btnIngestDataset'),
    uploadProgressContainer: document.getElementById('uploadProgressContainer'),
    uploadProgressFill: document.getElementById('uploadProgressFill'),
    uploadProgressPct: document.getElementById('uploadProgressPct'),
    uploadProgressBytes: document.getElementById('uploadProgressBytes'),
    uploadProgressChunks: document.getElementById('uploadProgressChunks'),
    accessDeniedModal: document.getElementById('accessDeniedModal'),
    modalDeniedReason: document.getElementById('modalDeniedReason'),
    btnElevateRole: document.getElementById('btnElevateRole'),
    targetNodeRisk: document.getElementById('targetNodeRisk'),
    toastContainer: document.getElementById('toastContainer'),
  };

  initEventListeners();
  initSSETelemetry();
});

function initEventListeners() {
  // Trigger Dataset Ingestion file browser
  if (dom.btnIngestDataset && dom.chunkFileInput) {
    dom.btnIngestDataset.addEventListener('click', () => {
      dom.chunkFileInput.click();
    });
  }

  // Handle selected file for chunked upload
  if (dom.chunkFileInput) {
    dom.chunkFileInput.addEventListener('change', handleFileSelected);
  }

  // Test Dataset Burst action (Requires clearance Level 5)
  if (dom.btnDatasetBurst) {
    dom.btnDatasetBurst.addEventListener('click', () => {
      requireClearance(5, 'Test Dataset Burst (Live Latent Injection)', () => {
        executeDatasetBurst();
      });
    });
  }

  // Elevate Role Modal Button
  if (dom.btnElevateRole) {
    dom.btnElevateRole.addEventListener('click', elevateToCISO);
  }
}

/* =========================================================================
   1. Zero-Trust RBAC & 403 Elevation Modal
   ========================================================================= */
function requireClearance(requiredLevel, actionName, onApproved) {
  if (STATE.roleLevel >= requiredLevel) {
    onApproved();
    return;
  }

  // Block execution & trigger 403 Access Denied modal
  STATE.pendingAction = onApproved;
  if (dom.modalDeniedReason) {
    dom.modalDeniedReason.innerText = 
      `Operational action '${actionName}' requires Level ${requiredLevel} Clearance. ` +
      `Your current active identity '${STATE.username}' holds '${STATE.currentRole}'.`;
  }
  showModal();
}

function showModal() {
  if (dom.accessDeniedModal) {
    dom.accessDeniedModal.classList.add('show');
  }
}

function closeModal() {
  if (dom.accessDeniedModal) {
    dom.accessDeniedModal.classList.remove('show');
  }
}

function elevateToCISO() {
  STATE.currentRole = 'Chief CISO (Level 5)';
  STATE.roleLevel = 5;

  if (dom.userChipText) {
    dom.userChipText.innerText = `${STATE.username} - Chief CISO L5`;
  }

  showToast('Clearance elevated to Chief CISO (Level 5). Zero-Trust restriction unlocked.');
  closeModal();

  // Re-run pending action if present
  if (STATE.pendingAction) {
    const action = STATE.pendingAction;
    STATE.pendingAction = null;
    action();
  }
}

function executeDatasetBurst() {
  showToast('⚡ Injecting dataset burst into Threatora Temporal Transformer World Model...');
  if (window.triggerRadarBurst) {
    window.triggerRadarBurst();
  }

  // Post burst trigger to backend if active
  fetch('/api/test-burst', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.status === 'success') {
        showToast(`Burst processed: ${data.message || 'Horizon trajectory updated.'}`);
      }
    })
    .catch(() => {
      // Offline / simulation feedback
      showToast('Burst simulated across active topology nodes.');
    });
}

/* =========================================================================
   2. 10MB Chunked File Uploader (Supporting up to 2.5 GB)
   ========================================================================= */
const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB per chunk

async function handleFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;

  const MAX_LIMIT = 2.5 * 1024 * 1024 * 1024; // 2.5 GB
  if (file.size > MAX_LIMIT) {
    showToast(`File exceeds 2.5 GB ceiling (${(file.size / (1024*1024)).toFixed(1)} MB).`, true);
    dom.chunkFileInput.value = '';
    return;
  }

  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  showToast(`Initiating chunked ingestion for '${file.name}' (${(file.size / (1024*1024)).toFixed(1)} MB, ${totalChunks} chunks)...`);

  if (dom.uploadProgressContainer) {
    dom.uploadProgressContainer.style.display = 'block';
  }

  STATE.isUploading = true;
  let uploadedBytes = 0;

  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    const start = chunkIndex * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunkBlob = file.slice(start, end);

    const formData = new FormData();
    formData.append('file', chunkBlob, file.name);
    formData.append('filename', file.name);
    formData.append('chunk_index', chunkIndex);
    formData.append('total_chunks', totalChunks);

    try {
      const response = await fetch('/api/upload-chunk', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload error on chunk ${chunkIndex + 1}/${totalChunks} (HTTP ${response.status})`);
      }

      const resData = await response.json();
      uploadedBytes += (end - start);

      // Update UI Progress Bar
      const pct = Math.round((uploadedBytes / file.size) * 100);
      if (dom.uploadProgressFill) dom.uploadProgressFill.style.width = `${pct}%`;
      if (dom.uploadProgressPct) dom.uploadProgressPct.innerText = `${pct}%`;
      if (dom.uploadProgressBytes) dom.uploadProgressBytes.innerText = `${(uploadedBytes / (1024*1024)).toFixed(1)} MB / ${(file.size / (1024*1024)).toFixed(1)} MB`;
      if (dom.uploadProgressChunks) dom.uploadProgressChunks.innerText = `Chunk ${chunkIndex + 1}/${totalChunks}`;

      if (chunkIndex === totalChunks - 1) {
        showToast(`Dataset '${file.name}' fully assembled and ingested into World Model pipeline.`);
        setTimeout(() => {
          if (dom.uploadProgressContainer) dom.uploadProgressContainer.style.display = 'none';
        }, 2500);
      }
    } catch (err) {
      showToast(`Chunk ingestion failed: ${err.message}`, true);
      STATE.isUploading = false;
      dom.chunkFileInput.value = '';
      return;
    }
  }

  STATE.isUploading = false;
  dom.chunkFileInput.value = '';
}

/* =========================================================================
   3. Server-Sent Events (SSE) Telemetry Stream Listener
   ========================================================================= */
function initSSETelemetry() {
  if (STATE.sseSource) {
    STATE.sseSource.close();
  }

  STATE.sseSource = new EventSource('/api/stream-telemetry');

  STATE.sseSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateTelemetryHUD(data);
    } catch (e) {
      console.warn("SSE telemetry parse note:", e);
    }
  };

  STATE.sseSource.onerror = () => {
    // Reconnects automatically via browser EventSource standard
  };
}

function updateTelemetryHUD(data) {
  if (!data) return;

  // Update target node risk readout
  if (dom.targetNodeRisk && data.risk_score !== undefined) {
    dom.targetNodeRisk.innerText = `74.8% INFILTRATION RISK (STREAM: ${(data.risk_score * 100).toFixed(1)}%)`;
  }
}

/* =========================================================================
   4. UI Utilities
   ========================================================================= */
function showToast(msg, isError = false) {
  if (!dom.toastContainer) return;
  const t = document.createElement('div');
  t.className = `toast ${isError ? 'toast-error' : ''}`;
  t.innerHTML = `<span>${isError ? '⛔' : '🛡️'}</span><span>${msg}</span>`;
  dom.toastContainer.appendChild(t);

  setTimeout(() => {
    t.style.transition = 'all 0.3s ease';
    t.style.opacity = '0';
    t.style.transform = 'translateY(10px)';
    setTimeout(() => t.remove(), 300);
  }, 4000);
}

// Window exports
window.closeModal = closeModal;
window.elevateToCISO = elevateToCISO;
