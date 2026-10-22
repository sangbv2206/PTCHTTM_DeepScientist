const stages = [
  '0. Tiền kiểm dữ liệu',
  '1. Khảo sát tri thức',
  '2. Xác định bài toán',
  '3. Thiết kế phương pháp',
  '4. Thiết kế thực nghiệm',
  '5. Huấn luyện PyTorch',
  '6. Biên soạn bài báo',
  '7. Phản biện học thuật'
];
const $ = (id) => document.getElementById(id);

let allRuns = [];
let selectedRun = null;          // ID đề tài đang được hiển thị trong #detailPanel
let activeBackgroundRun = null;  // Thông tin đề tài đang chạy ngầm { run_id, topic, status, stage, progress, message }
let bgEventSource = null;        // EventSource theo dõi tiến trình chạy ngầm
let currentPage = 1;
const PAGE_SIZE = 5;
let searchQuery = '';

async function initWorkspace() {
  try {
    const health = await fetch('/api/health').then((r) => r.json());
    if (health.default_model) {
      $('model').value = health.default_model;
    }
  } catch {}
  try {
    const data = await fetch('/api/templates').then((response) => response.json());
    data.templates.forEach((name) => $('template').insertAdjacentHTML('beforeend', `<option value="${name}">${name.replaceAll('_', ' ')}</option>`));
  } catch { 
    $('formMessage').textContent = 'Không thể nạp danh sách template. Máy chủ API có thể chưa bật.'; 
  }
}

function statusLabel(status) {
  const map = {
    'queued': 'CHỜ XỬ LÝ',
    'running': 'ĐANG THỰC HIỆN',
    'completed': 'HOÀN THÀNH',
    'failed': 'THẤT BẠI',
    'interrupted': 'GIÁN ĐOẠN'
  };
  return map[status] || (status || 'CHỜ XỬ LÝ').toUpperCase();
}

function escapeHtml(value) {
  return String(value || '').replace(/[&<>'"]/g, (char) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[char]));
}

function updateSelectedCardHighlight() {
  document.querySelectorAll('.run-item').forEach((item) => {
    item.classList.toggle('active', item.dataset.run === selectedRun);
  });
}

function updateCardInDOM(runId, status) {
  const card = document.querySelector(`.run-item[data-run="${runId}"]`);
  if (!card) return;
  const pill = card.querySelector('.status-pill');
  if (pill) {
    pill.textContent = statusLabel(status);
    pill.className = `status-pill ${status === 'failed' || status === 'interrupted' ? 'failed' : ''}`;
  }
}

function updateActiveBanner() {
  const banner = $('activeRunBanner');
  if (!banner) return;
  
  // Chỉ hiển thị banner khi có tiến trình đang chạy VÀ người dùng đang KHÔNG xem trực tiếp nó
  if (activeBackgroundRun && selectedRun !== activeBackgroundRun.run_id) {
    banner.classList.remove('hidden');
    const topicEl = $('activeRunTopic');
    if (topicEl) topicEl.textContent = activeBackgroundRun.topic || activeBackgroundRun.run_id;
    const stageEl = $('activeRunStage');
    if (stageEl) {
      const stageIdx = activeBackgroundRun.stage || 0;
      stageEl.textContent = `Giai đoạn: ${stages[stageIdx] || stageIdx}`;
    }
    const progEl = $('activeRunProgress');
    if (progEl) {
      const pct = Math.round((activeBackgroundRun.progress || 0) * 100);
      progEl.textContent = `${pct}%`;
    }
    const msgEl = $('activeRunMessage');
    if (msgEl) {
      msgEl.textContent = activeBackgroundRun.message ? `(${activeBackgroundRun.message})` : '';
    }
  } else {
    banner.classList.add('hidden');
  }
}

function trackActiveRun(runId, topic) {
  if (bgEventSource) {
    bgEventSource.close();
    bgEventSource = null;
  }
  
  activeBackgroundRun = {
    run_id: runId,
    topic: topic || runId,
    status: 'running',
    stage: 0,
    progress: 0.0,
    message: 'Đang khởi chạy…'
  };
  updateActiveBanner();
  
  const source = new EventSource(`/api/runs/${runId}/events`);
  bgEventSource = source;

  source.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data);
      activeBackgroundRun = {
        ...activeBackgroundRun,
        ...event,
        run_id: runId,
        topic: activeBackgroundRun.topic || event.topic || runId
      };

      // Cập nhật trạng thái trong bộ nhớ allRuns
      const existing = allRuns.find((r) => r.run_id === runId);
      if (existing) {
        Object.assign(existing, event);
        updateCardInDOM(runId, event.status);
      }

      updateActiveBanner();

      // QUAN TRỌNG: Chỉ cập nhật giao diện chi tiết nếu người dùng đang CHỦ ĐÍCH XEM đề tài này!
      // Nếu người dùng đang bấm xem bài báo cũ khác, không được ghi đè màn hình của họ!
      if (selectedRun === runId) {
        renderDetail({
          ...event,
          topic: activeBackgroundRun.topic,
          run_id: runId
        });
        loadArtifacts(runId);
      }

      // Khi tiến trình kết thúc
      if (event.status === 'completed' || event.status === 'failed' || event.status === 'interrupted') {
        source.close();
        bgEventSource = null;
        const finishedTopic = activeBackgroundRun.topic;
        activeBackgroundRun = null;
        updateActiveBanner();
        // Cập nhật lại toàn bộ danh sách để có đầy đủ trạng thái và PDF
        loadRuns();
      }
    } catch (e) {
      console.warn('Lỗi xử lý SSE event:', e);
    }
  };

  source.onerror = () => {
    // Nếu mất kết nối tạm thời hoặc hoàn tất
    if (activeBackgroundRun && (activeBackgroundRun.status === 'completed' || activeBackgroundRun.status === 'failed')) {
      source.close();
      bgEventSource = null;
      activeBackgroundRun = null;
      updateActiveBanner();
    }
  };
}

function renderRuns() {
  const filtered = allRuns.filter((r) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (r.topic || '').toLowerCase().includes(q) || (r.run_id || '').toLowerCase().includes(q);
  });

  const total = filtered.length;
  if ($('runsCountBadge')) {
    $('runsCountBadge').textContent = allRuns.length;
  }

  const totalPages = Math.ceil(total / PAGE_SIZE) || 1;
  if (currentPage > totalPages) currentPage = totalPages;
  if (currentPage < 1) currentPage = 1;

  const start = (currentPage - 1) * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, total);
  const paged = filtered.slice(start, end);

  if (!paged.length) {
    $('runsList').innerHTML = searchQuery 
      ? '<div class="empty-state">Không tìm thấy đề tài nào khớp với từ khóa tìm kiếm.</div>'
      : '<div class="empty-state">Chưa có lượt chạy nào. Lượt nghiên cứu mới sẽ xuất hiện tại đây.</div>';
    $('runsPagination').innerHTML = '';
    return;
  }

  $('runsList').innerHTML = paged.map((run) => `
    <article class="run-item ${selectedRun === run.run_id ? 'active' : ''}" data-run="${run.run_id}">
      <div class="run-topic">${escapeHtml(run.topic)}</div>
      <div class="run-meta">
        <span>Mã: ${run.run_id}</span>
        <span class="status-pill ${run.status === 'failed' || run.status === 'interrupted' ? 'failed' : ''}">${statusLabel(run.status)}</span>
      </div>
    </article>
  `).join('');

  // Thanh điều khiển phân trang
  if (total > PAGE_SIZE) {
    $('runsPagination').innerHTML = `
      <span>Hiển thị <strong>${start + 1}–${end}</strong> / <strong>${total}</strong> đề tài</span>
      <div class="pagination-controls">
        <button class="page-btn" id="prevPageBtn" ${currentPage === 1 ? 'disabled' : ''}>← Trước</button>
        <span class="page-info">${currentPage}/${totalPages}</span>
        <button class="page-btn" id="nextPageBtn" ${currentPage === totalPages ? 'disabled' : ''}>Sau →</button>
      </div>
    `;
    const prevBtn = $('prevPageBtn');
    const nextBtn = $('nextPageBtn');
    if (prevBtn) prevBtn.addEventListener('click', (e) => { e.stopPropagation(); if (currentPage > 1) { currentPage--; renderRuns(); } });
    if (nextBtn) nextBtn.addEventListener('click', (e) => { e.stopPropagation(); if (currentPage < totalPages) { currentPage++; renderRuns(); } });
  } else {
    $('runsPagination').innerHTML = `<span>Tổng cộng: <strong>${total}</strong> đề tài</span>`;
  }
}

async function loadRuns() {
  try {
    const data = await fetch('/api/runs').then((response) => response.json());
    allRuns = data.runs || [];
    renderRuns();

    // Tự động phát hiện nếu có đề tài đang chạy ngầm
    const runningRun = allRuns.find((r) => r.status === 'running' || r.status === 'queued');
    if (runningRun) {
      if (!bgEventSource || !activeBackgroundRun || activeBackgroundRun.run_id !== runningRun.run_id) {
        trackActiveRun(runningRun.run_id, runningRun.topic);
      }
    }

    // Nếu chưa chọn đề tài nào, mặc định mở đề tài đang chạy hoặc bài báo gần nhất
    if (!selectedRun && allRuns.length > 0) {
      const defaultToOpen = runningRun ? runningRun.run_id : allRuns[0].run_id;
      openRun(defaultToOpen);
    }
  } catch (err) {
    console.error('Lỗi nạp danh sách đề tài:', err);
  }
}

function renderDetail(run) {
  $('detailPanel').classList.remove('hidden');
  $('detailTitle').textContent = run.topic || 'Đề tài nghiên cứu';
  $('detailMessage').textContent = run.message || '';
  $('detailStatus').textContent = statusLabel(run.status);
  $('detailStatus').className = `status-pill ${run.status === 'failed' || run.status === 'interrupted' ? 'failed' : ''}`;
  const pct = Math.round((run.progress || 0) * 100);
  $('progressBar').style.width = `${pct}%`;
  $('progressValue').textContent = `${pct}%`;
  $('stageValue').textContent = stages[run.stage || 0] || `Giai đoạn ${run.stage}`;
  $('runValue').textContent = run.run_id;
  updatePaperViewer(run.run_id, run.status);
  $('stageRail').innerHTML = stages.map((stage, index) => `
    <div class="stage-item ${index <= (run.stage || 0) ? 'active' : ''}">
      <span class="stage-index">0${index}</span>
      <span>${stage}</span>
    </div>
  `).join('');
}

function artifactLabel(path) {
  if (path.endsWith('paper.pdf')) return 'Toàn văn bài báo khoa học (PDF)';
  if (path.endsWith('.png')) return `Biểu đồ đối sánh (${path.split('/').pop()})`;
  if (path.endsWith('peer_review.md')) return 'Báo cáo phản biện hội nghị (Peer Review)';
  if (path.endsWith('claim_ledger.md')) return 'Sổ cái tuyên bố số liệu (Claim Ledger)';
  if (path.endsWith('preflight_assessment.json')) return 'Báo cáo tiền kiểm đề tài (Preflight)';
  if (path.endsWith('.bib')) return 'Tệp tài liệu tham khảo (BibTeX)';
  if (path.endsWith('.tex')) return 'Mã nguồn bản thảo (LaTeX)';
  if (path.endsWith('experiment_upgraded.py')) return 'Mã nguồn thuật toán đề xuất (PyTorch)';
  if (path.endsWith('experiment.py')) return 'Mã nguồn mô hình đối chứng cơ sở (Baseline)';
  return `Tệp: ${path.split('/').pop()}`;
}

async function loadArtifacts(runId) {
  try {
    const data = await fetch(`/api/runs/${runId}/artifacts`).then((response) => response.json());
    if (selectedRun !== runId) return; // Nếu đã chuyển sang xem đề tài khác thì bỏ qua
    $('artifactList').innerHTML = data.artifacts && data.artifacts.length ? data.artifacts.slice(-16).map((item) => `
      <a class="artifact-link" href="/api/runs/${runId}/artifacts/${item.path}" target="_blank" rel="noreferrer">
        ↗ ${artifactLabel(item.path)}
      </a>
    `).join('') : 'Các tệp kết quả sẽ xuất hiện khi tiến trình chạy.';
  } catch {}
}

function updatePaperViewer(runId, status) {
  const ready = status === 'completed';
  $('paperViewer').classList.toggle('hidden', !ready);
  if (ready) {
    const pdfUrl = `/api/runs/${runId}/pdf`;
    if ($('paperFrame').src !== window.location.origin + pdfUrl) {
      $('paperFrame').src = pdfUrl;
    }
    $('paperLink').href = pdfUrl;
  }
}

async function openRun(runId) {
  selectedRun = runId;
  updateSelectedCardHighlight();
  updateActiveBanner();

  try {
    const run = await fetch(`/api/runs/${runId}`).then((response) => response.json());
    if (selectedRun !== runId) return; // Nếu người dùng đã click nhanh sang đề tài khác
    renderDetail(run);
    await loadArtifacts(runId);
  } catch (err) {
    console.error('Không thể tải chi tiết lượt chạy:', err);
  }

  $('detailPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function submitRun(event) {
  event.preventDefault();
  $('formMessage').textContent = 'Đang khởi chạy tiến trình nghiên cứu…';
  const payload = {
    topic: $('topic').value.trim(),
    template: $('template').value || null,
    iterations: Number($('iterations').value),
    model: $('model').value.trim() || null,
    offline: $('offline').checked
  };
  try {
    const run = await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then((response) => response.json());
    
    $('formMessage').textContent = 'Nghiên cứu đã bắt đầu thành công!';
    $('topic').value = '';
    
    // Bắt đầu theo dõi tiến trình chạy ngầm
    trackActiveRun(run.run_id, payload.topic);
    
    // Nạp lại danh sách và mở trực tiếp đề tài vừa tạo
    await loadRuns();
    openRun(run.run_id);
  } catch (err) {
    $('formMessage').textContent = 'Không thể bắt đầu lượt chạy. Hãy kiểm tra máy chủ web có đang hoạt động không!';
  }
}

async function rerunRun() {
  if (!selectedRun) return;
  const btn = $('rerunButton');
  btn.disabled = true;
  btn.textContent = 'Đang khởi chạy…';
  try {
    const res = await fetch(`/api/runs/${selectedRun}/rerun`, { method: 'POST' });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Máy chủ chưa được cập nhật API mới. Hãy khởi động lại python run_web.py!');
    }
    const run = await res.json();
    if (run && run.run_id) {
      trackActiveRun(run.run_id, run.topic);
      await loadRuns();
      openRun(run.run_id);
    }
  } catch (err) {
    alert('Không thể chạy lại: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Chạy lại';
  }
}

async function deleteRun() {
  if (!selectedRun) return;
  const targetRun = selectedRun;
  const ok = confirm(`Bạn có chắc chắn muốn xóa lượt nghiên cứu [${targetRun}] này không? Toàn bộ file và thư mục sẽ được dọn dẹp sạch sẽ.`);
  if (!ok) return;

  const btn = $('deleteButton');
  btn.disabled = true;
  btn.textContent = 'Đang xóa…';

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);

  try {
    const res = await fetch(`/api/runs/${targetRun}`, { 
      method: 'DELETE',
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!res.ok && res.status !== 404) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Không thể xóa lượt nghiên cứu này.');
    }

    // Nếu đang theo dõi tiến trình của chính bài bị xóa
    if (activeBackgroundRun && activeBackgroundRun.run_id === targetRun) {
      if (bgEventSource) bgEventSource.close();
      bgEventSource = null;
      activeBackgroundRun = null;
      updateActiveBanner();
    }

    selectedRun = null;
    $('detailPanel').classList.add('hidden');
    await loadRuns();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      alert('Yêu cầu xóa quá thời gian chờ (Timeout). Có thể máy chủ backend đang bị dừng hoặc bận.');
    } else if (err.message && err.message.includes('Failed to fetch')) {
      alert('Không thể kết nối đến máy chủ backend (Cổng 8000). Vui lòng kiểm tra lại terminal xem server có đang chạy không.');
    } else {
      alert('Lỗi khi xóa: ' + err.message);
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Xóa';
  }
}

// Bắt sự kiện Click bằng Event Delegation cho toàn bộ danh sách bài báo:
// Đảm bảo không bao giờ bị mất click hay xung đột khi dữ liệu cập nhật
$('runsList').addEventListener('click', (e) => {
  const item = e.target.closest('[data-run]');
  if (item && item.dataset.run) {
    openRun(item.dataset.run);
  }
});

// Nút chuyển nhanh đến đề tài đang chạy ngầm trên banner
const jumpBtn = $('viewActiveRunBtn');
if (jumpBtn) {
  jumpBtn.addEventListener('click', () => {
    if (activeBackgroundRun && activeBackgroundRun.run_id) {
      openRun(activeBackgroundRun.run_id);
    }
  });
}

$('runForm').addEventListener('submit', submitRun);
$('refreshRuns').addEventListener('click', loadRuns);
$('rerunButton').addEventListener('click', rerunRun);
$('deleteButton').addEventListener('click', deleteRun);

const searchInput = $('runsSearchInput');
if (searchInput) {
  searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value.trim();
    currentPage = 1;
    renderRuns();
  });
}

initWorkspace();
loadRuns();
