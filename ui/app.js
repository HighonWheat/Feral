/* ==========================================================================
   FERAL DESKTOP COMMAND CENTER — ODYSSEUS TACTICAL FRONTEND CONTROLLER
   ========================================================================== */
const state = {
  snapshot: null,
  ws: null,
  streamingCardEl: null,
  streamingThinkText: "",
  streamingAnswerText: "",
  draggedAgentId: null,
};

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlightCodeSyntax(escapedCode) {
  return escapedCode
    .replace(/(#.*$)/gm, '<span class="tok-comment">$1</span>')
    .replace(/\b(def|class|return|import|from|if|elif|else|for|while|try|except|finally|with|as|assert|True|False|None|async|await)\b/g, '<span class="tok-kw">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="tok-num">$1</span>');
}

function renderMarkdownWithCodeBlocks(rawText) {
  if (!rawText) return "";
  const parts = [];
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(rawText)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: "text", content: rawText.slice(lastIndex, match.index) });
    }
    parts.push({
      type: "code",
      lang: match[1] || "code",
      content: match[2].replace(/\n$/, ""),
    });
    lastIndex = codeBlockRegex.lastIndex;
  }
  if (lastIndex < rawText.length) {
    parts.push({ type: "text", content: rawText.slice(lastIndex) });
  }

  return parts
    .map((p) => {
      if (p.type === "code") {
        const encoded = encodeURIComponent(p.content);
        const highlighted = highlightCodeSyntax(escapeHtml(p.content));
        return `
          <div class="code-block-wrap">
            <div class="code-block-header">
              <span>${escapeHtml(p.lang.toUpperCase())}</span>
              <button class="copy-code-btn" onclick="copyCodeBlock(this, '${encoded}')">Copy Code</button>
            </div>
            <pre class="code-block-pre"><code>${highlighted}</code></pre>
          </div>
        `;
      } else {
        const inlineFormatted = escapeHtml(p.content)
          .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
          .replace(/`([^`]+)`/g, '<code style="background:#090d16;padding:1px 5px;border-radius:4px;color:#38bdf8;font-family:var(--font-mono);">$1</code>')
          .replace(/\n\n+/g, "</p><p>")
          .replace(/\n/g, "<br/>");
        return `<p>${inlineFormatted}</p>`;
      }
    })
    .join("");
}

window.copyCodeBlock = function (btnEl, encodedCode) {
  const decoded = decodeURIComponent(encodedCode);
  navigator.clipboard.writeText(decoded).then(() => {
    const prev = btnEl.textContent;
    btnEl.textContent = "Copied ✓";
    setTimeout(() => {
      btnEl.textContent = prev;
    }, 1400);
  });
};

/* ==========================================================================
   1. SNAPSHOT RENDERING ACROSS ALL 5 MODULES
   ========================================================================== */
function applySnapshot(snap) {
  if (!snap) return;
  state.snapshot = snap;

  // Topbar GPU & Watchdog Telemetry
  const hw = snap.integrations_panel?.hardware_cookbook || {};
  const wd = snap.workbenches?.loop_watchdog || {};
  document.getElementById("top-gpu-name").textContent = (hw.device_name || "RTX 5070 Ti").replace("NVIDIA GeForce ", "");
  document.getElementById("top-vram-bar").style.width = `${hw.vram_utilization_pct || 47}%`;
  document.getElementById("top-vram-text").textContent = `${Math.round(hw.vram_allocated_mib || 0)} / ${Math.round(hw.total_vram_mib || 16276)} MiB`;
  document.getElementById("top-watchdog-text").textContent = `TEMP: ${(wd.current_temperature ?? 0).toFixed(2)}`;

  // Module 1: Chat Hub & Collapsible Context Panel
  renderChatHub(snap.chat_hub);

  // Module 2: Autonomous Agents Board & Task DAG
  renderAgentsBoard(snap.agents_board);

  // Module 3: Knowledge Base & Vector Storage UI
  renderKnowledgeBase(snap.knowledge_base);

  // Module 4: Integrations Panel & Hardware Cookbook
  renderIntegrationsPanel(snap.integrations_panel);

  // Module 5: Developer Workbenches
  renderWorkbenches(snap.workbenches);
}

function renderChatHub(chatHub) {
  if (!chatHub) return;
  const ctx = chatHub.context_panel || {};
  const models = chatHub.available_models || [];
  const comp = chatHub.compaction || {};

  // Populate local model dropdown
  const modelSelect = document.getElementById("ctx-model-select");
  if (modelSelect && models.length > 0 && modelSelect.options.length !== models.length) {
    modelSelect.innerHTML = models
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.id)} (${m.backend})</option>`)
      .join("");
  }
  if (ctx.model_id) {
    modelSelect.value = ctx.model_id;
    document.getElementById("active-model-banner").textContent = ctx.model_id;
  }

  const sysTextarea = document.getElementById("ctx-system-prompt");
  if (sysTextarea && document.activeElement !== sysTextarea && ctx.system_prompt) {
    sysTextarea.value = ctx.system_prompt;
  }

  if (typeof ctx.temperature === "number") {
    document.getElementById("ctx-temp-slider").value = ctx.temperature;
    document.getElementById("ctx-temp-val").textContent = ctx.temperature.toFixed(2);
  }
  if (typeof ctx.max_new_tokens === "number") {
    document.getElementById("ctx-tokens-slider").value = ctx.max_new_tokens;
    document.getElementById("ctx-tokens-val").textContent = ctx.max_new_tokens;
  }

  // Compaction gauge & AGENT_STATE.md
  const activeTok = comp.active_tokens || 0;
  const maxTok = comp.max_context_tokens || 4096;
  const pct = Math.min(100, Math.round((activeTok / maxTok) * 100));
  document.getElementById("chat-token-gauge").textContent = `${activeTok} / ${maxTok} tok`;
  document.getElementById("chat-token-bar").style.width = `${Math.max(4, pct)}%`;
  document.getElementById("compaction-pass-label").textContent = `Compaction Passes: #${comp.compaction_count || 0} (Trigger: ${Math.round((comp.trigger_ratio || 0.75) * 100)}%)`;
  document.getElementById("ctx-agent-state-md").textContent = comp.agent_state_markdown || "# AGENT_STATE.md";

  // Render Chat Messages if not currently mid-stream
  if (!state.streamingCardEl && Array.isArray(chatHub.messages) && chatHub.messages.length > 0) {
    const streamEl = document.getElementById("chat-stream");
    streamEl.innerHTML = chatHub.messages.map(renderChatMessageCard).join("");
    streamEl.scrollTop = streamEl.scrollHeight;
  }
}

function renderChatMessageCard(entry) {
  if (entry.role === "user") {
    return `
      <div class="msg-card user">
        <div class="msg-header">
          <span>OPERATOR // MISSION DIRECTIVE</span>
          <span>#${entry.id}</span>
        </div>
        <div class="md-content">${renderMarkdownWithCodeBlocks(entry.content)}</div>
      </div>
    `;
  }

  const thinkBlock = entry.think
    ? `
      <details class="think-drawer" open>
        <summary class="think-summary">
          <span>&lt;think&gt; Autonomous Reasoning Stream</span>
          <span>click to collapse</span>
        </summary>
        <div class="think-body">${escapeHtml(entry.think)}</div>
      </details>
    `
    : "";

  const toolBlock = entry.tool_call
    ? `
      <div class="tool-call-box">
        <div style="color:var(--accent-cyan);font-weight:700;margin-bottom:4px;">
          &lt;tool_call&gt; ${escapeHtml(entry.tool_call.name)}
        </div>
        <div>${escapeHtml(JSON.stringify(entry.tool_call.arguments, null, 2))}</div>
      </div>
    `
    : "";

  const obsBlock = entry.tool_observation
    ? `
      <div class="tool-obs-box">${escapeHtml(entry.tool_observation)}</div>
    `
    : "";

  const ansBlock = entry.content
    ? `<div class="md-content">${renderMarkdownWithCodeBlocks(entry.content)}</div>`
    : "";

  const speedLabel = entry.tokens_per_sec ? `${entry.tokens_per_sec} tok/s | ${entry.vram_allocated_mib || 7643} MiB` : "FERAL";

  return `
    <div class="msg-card assistant">
      <div class="msg-header">
        <span>FERAL AGENT // TURN #${entry.id}</span>
        <span style="color:var(--accent-emerald);">${escapeHtml(speedLabel)}</span>
      </div>
      ${thinkBlock}
      ${toolBlock}
      ${obsBlock}
      ${ansBlock}
    </div>
  `;
}

/* ==========================================================================
   2. MODULE 2: AUTONOMOUS AGENTS BOARD & SQLITE TASK DAG
   ========================================================================== */
function renderAgentsBoard(board) {
  if (!board) return;
  const cards = board.cards || [];
  document.getElementById("badge-agents-count").textContent = cards.length;

  const gridEl = document.getElementById("agents-grid");
  gridEl.innerHTML = cards
    .map(
      (c, idx) => `
      <div class="agent-card"
           draggable="true"
           data-agent-id="${escapeHtml(c.agent_id)}"
           data-col="${c.grid_col}"
           data-row="${c.grid_row}"
           ondragstart="onAgentDragStart(event, '${escapeHtml(c.agent_id)}')"
           ondragover="onAgentDragOver(event)"
           ondrop="onAgentDrop(event, ${idx % 3}, ${Math.floor(idx / 3)})">
        <div class="agent-card-top">
          <div>
            <div class="agent-title">${escapeHtml(c.name)}</div>
            <div style="font-family:var(--font-mono);font-size:10.5px;color:var(--text-muted);">
              ID: ${escapeHtml(c.agent_id)} // GRID (${c.grid_col}, ${c.grid_row}) // STEPS: ${c.steps_completed}/${c.max_steps}
            </div>
          </div>
          <span class="status-badge status-${escapeHtml(c.status)}">${escapeHtml(c.status)}</span>
        </div>

        <div class="agent-goal">${escapeHtml(c.goal)}</div>

        <pre class="agent-console">${escapeHtml((c.console_log || []).join("\n"))}</pre>

        <div class="agent-actions">
          <span style="font-family:var(--font-mono);font-size:10.5px;color:var(--accent-cyan);">
            Context: ${c.child_context_tokens || 0} &rarr; ${c.parent_summary_tokens || 0} tok
          </span>
          <div style="display:flex;gap:6px;">
            <button class="btn-primary" style="padding:5px 10px;font-size:11px;" onclick="deployAgentCard('${escapeHtml(c.agent_id)}')">
              Deploy
            </button>
            <button class="btn-secondary" style="padding:5px 9px;font-size:11px;" onclick="controlAgentCard('${escapeHtml(c.agent_id)}', 'pause')">
              Pause
            </button>
            <button class="btn-secondary" style="padding:5px 9px;font-size:11px;" onclick="controlAgentCard('${escapeHtml(c.agent_id)}', 'resume')">
              Resume
            </button>
          </div>
        </div>
      </div>
    `
    )
    .join("");

  // Render Task DAG
  const dag = board.dag || {};
  document.getElementById("dag-mission-title").textContent = `Mission: ${dag.goal || "Autonomous Workflow"}`;
  document.getElementById("dag-progress-label").textContent = `${dag.progress_pct || 0}% (${dag.completed_nodes || 0}/${dag.total_nodes || 0})`;
  document.getElementById("dag-progress-bar").style.width = `${dag.progress_pct || 0}%`;

  const dagList = document.getElementById("dag-nodes-list");
  dagList.innerHTML = (dag.nodes || [])
    .map(
      (n) => `
      <div class="dag-node-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <strong style="font-family:var(--font-mono);font-size:11.5px;color:var(--accent-cyan);">${escapeHtml(n.node_id)}</strong>
          <span class="status-badge status-${escapeHtml(n.status)}">${escapeHtml(n.status)}</span>
        </div>
        <div style="font-weight:600;font-size:12.5px;">${escapeHtml(n.title)}</div>
        <div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">
          Agent: ${escapeHtml(n.assigned_agent)} | Deps: ${(n.dependencies || []).join(", ") || "none"}
        </div>
      </div>
    `
    )
    .join("");
}

window.onAgentDragStart = function (ev, agentId) {
  state.draggedAgentId = agentId;
  ev.currentTarget.classList.add("dragging");
};

window.onAgentDragOver = function (ev) {
  ev.preventDefault();
};

window.onAgentDrop = async function (ev, targetCol, targetRow) {
  ev.preventDefault();
  const agentId = state.draggedAgentId;
  if (!agentId) return;
  await fetch(`/api/agents/${encodeURIComponent(agentId)}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ grid_col: targetCol, grid_row: targetRow }),
  });
  await refreshSnapshot();
};

window.deployAgentCard = async function (agentId) {
  await fetch(`/api/agents/${encodeURIComponent(agentId)}/deploy`, { method: "POST" });
  await refreshSnapshot();
};

window.controlAgentCard = async function (agentId, action) {
  await fetch(`/api/agents/${encodeURIComponent(agentId)}/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  await refreshSnapshot();
};

/* ==========================================================================
   3. MODULE 3: KNOWLEDGE BASE & VECTOR STORAGE UI
   ========================================================================== */
function renderKnowledgeBase(kb) {
  if (!kb) return;
  const cols = kb.collections || [];
  document.getElementById("badge-kb-count").textContent = cols.length;

  const sel = document.getElementById("kb-target-collection");
  if (sel && cols.length > 0) {
    const prevVal = sel.value;
    sel.innerHTML = cols
      .map((c) => `<option value="${escapeHtml(c.collection_id)}">${escapeHtml(c.name)} (${c.collection_id})</option>`)
      .join("");
    if (prevVal && cols.some((c) => c.collection_id === prevVal)) {
      sel.value = prevVal;
    }
  }

  const grid = document.getElementById("kb-collections-grid");
  grid.innerHTML = cols
    .map(
      (c) => `
      <div class="collection-card" id="col-card-${escapeHtml(c.collection_id)}">
        <div class="agent-card-top">
          <div>
            <div class="agent-title">${escapeHtml(c.name)}</div>
            <div style="font-family:var(--font-mono);font-size:10.5px;color:var(--accent-cyan);">
              CLUSTER ID: ${escapeHtml(c.collection_id)} // ${c.embedding_dim}-DIM L2
            </div>
          </div>
          <span class="status-badge status-${escapeHtml(c.status)}">${escapeHtml(c.status)}</span>
        </div>
        <div style="font-size:12px;color:var(--text-secondary);">${escapeHtml(c.description || "Local ChromaDB / SQLite vector cluster")}</div>
        <div>
          <div class="field-label" style="margin-bottom:4px;">
            <span>Embedding Progress</span>
            <strong id="prog-text-${escapeHtml(c.collection_id)}" style="color:var(--accent-emerald);">${(c.embedding_progress_pct ?? 100).toFixed(1)}%</strong>
          </div>
          <div class="progress-track">
            <div class="progress-fill" id="prog-bar-${escapeHtml(c.collection_id)}" style="width:${c.embedding_progress_pct ?? 100}%;"></div>
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);">
          <span>Documents: <strong>${c.document_count}</strong></span>
          <span>Vector Chunks: <strong>${c.chunk_count}</strong></span>
        </div>
        <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);">
          Files: ${(c.documents || []).map(escapeHtml).join(", ") || "none"}
        </div>
      </div>
    `
    )
    .join("");
}

/* ==========================================================================
   4. MODULE 4: INTEGRATIONS PANEL & HARDWARE COOKBOOK
   ========================================================================== */
function renderIntegrationsPanel(panel) {
  if (!panel) return;
  const cfg = panel.config || {};
  const hw = panel.hardware_cookbook || {};

  document.getElementById("int-searxng-enabled").checked = !!cfg.searxng_enabled;
  document.getElementById("int-searxng-endpoint").value = cfg.searxng_endpoint || "http://localhost:8080/search";
  document.getElementById("int-headless-enabled").checked = !!cfg.headless_browser_enabled;
  document.getElementById("int-vision-enabled").checked = !!cfg.vision_screenshots_enabled;

  document.getElementById("int-email-enabled").checked = !!cfg.email_enabled;
  document.getElementById("int-imap-host").value = cfg.imap_host || "127.0.0.1";
  document.getElementById("int-imap-port").value = cfg.imap_port || 1143;
  document.getElementById("int-smtp-host").value = cfg.smtp_host || "127.0.0.1";
  document.getElementById("int-smtp-port").value = cfg.smtp_port || 1025;
  document.getElementById("int-email-account").value = cfg.email_account || "feral-agent@localhost.local";

  document.getElementById("int-calendar-enabled").checked = !!cfg.calendar_sync_enabled;
  document.getElementById("int-caldav-endpoint").value = cfg.caldav_endpoint || "http://localhost:5232/feral/calendar.ics";
  document.getElementById("int-watchdog-enabled").checked = !!cfg.loop_watchdog_enabled;
  document.getElementById("int-timetravel-enabled").checked = !!cfg.git_time_travel_enabled;

  // Hardware Cookbook Card
  document.getElementById("hw-arch-badge").textContent = `${hw.architecture || "NVIDIA Blackwell"} (${hw.compute_capability || "sm_120"})`;
  document.getElementById("hw-vram-detail").textContent = `${hw.vram_allocated_mib || 0} / ${hw.total_vram_mib || 16276} MiB (${hw.vram_utilization_pct || 47}%)`;
  document.getElementById("hw-vram-bar").style.width = `${hw.vram_utilization_pct || 47}%`;

  const rec = hw.cookbook_recommendations || {};
  const gguf = rec.gguf_artifact || {};
  document.getElementById("hw-cookbook-box").textContent = [
    `GPU Device           : ${hw.device_name || "NVIDIA GeForce RTX 5070 Ti"}`,
    `VRAM Reserved / Free : ${hw.vram_reserved_mib || 0} MiB reserved | ${hw.vram_free_mib || 0} MiB free`,
    `GPU Core / Thermal   : ${hw.gpu_core_utilization_pct || 0}% util | ${hw.gpu_temp_celsius || 0} °C | ${hw.gpu_power_watts || 0} W`,
    `Active Recipe        : ${rec.active_precision_recipe || "4-bit NF4 + DoRA Norm Fusion"}`,
    `Safe Context Window  : ${rec.max_safe_context_tokens || 8192} tokens (Auto-Compactor @ ${rec.compactor_trigger_tokens || 3072} tok)`,
    `GGUF Export Artifact : ${gguf.filename || "feral-qwen3.5-9b-grpo-q4_k_m.gguf"} (${gguf.size_mib || 3964} MiB, ready=${!!gguf.ready})`,
  ].join("\n");
}

/* ==========================================================================
   5. MODULE 5: DEVELOPER WORKBENCHES (DIFFS, GIT TIME-TRAVEL, TERMINAL, VISION)
   ========================================================================== */
function renderWorkbenches(wb) {
  if (!wb) return;
  const diffs = wb.diff_ledger || [];
  document.getElementById("diff-count-badge").textContent = `${diffs.length} PATCHES`;
  const diffList = document.getElementById("workbench-diff-list");
  diffList.innerHTML = diffs.length
    ? diffs
        .slice()
        .reverse()
        .map(
          (d) => `
        <div class="dag-node-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <strong style="font-family:var(--font-mono);color:var(--accent-cyan);">Patch #${d.patch_id} // ${escapeHtml(d.path)}</strong>
            <button class="btn-secondary" style="padding:3px 8px;font-size:10.5px;" onclick="rollbackPatch(${d.patch_id})">
              Rollback Patch
            </button>
          </div>
          <pre class="diff-pre">${escapeHtml(d.unified_diff)}</pre>
        </div>
      `
        )
        .join("")
    : `<div style="color:var(--text-muted);font-family:var(--font-mono);">No diff patches applied yet in this session.</div>`;

  // Git Time-Travel Checkpoints
  const tt = wb.git_time_travel || {};
  document.getElementById("git-head-badge").textContent = `HEAD: ${tt.head_short_sha || "init"} (Strikes: ${tt.consecutive_strikes || 0}/${tt.max_strikes || 3})`;
  const gitList = document.getElementById("workbench-git-list");
  gitList.innerHTML = (tt.checkpoints || [])
    .slice()
    .reverse()
    .map(
      (cp) => `
      <div class="dag-node-card" style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-family:var(--font-mono);font-size:11.5px;color:var(--accent-emerald);">
            #${cp.checkpoint_id} [${escapeHtml(cp.short_sha)}] ${cp.verified_green ? "✓ GREEN" : "SNAPSHOT"}
          </div>
          <div style="font-size:12px;">${escapeHtml(cp.label)}</div>
        </div>
        <button class="btn-secondary" style="padding:4px 9px;font-size:10.5px;" onclick="rollbackGit(${cp.checkpoint_id})">
          Time-Travel Here
        </button>
      </div>
    `
    )
    .join("");

  // Terminal History
  const termHist = wb.terminal_history || [];
  const termLog = document.getElementById("workbench-term-log");
  if (termHist.length > 0) {
    termLog.textContent = termHist
      .map((h) => `PS ${h.cwd}> ${h.command}\n${h.output}\n[exit_code=${h.exit_code}]`)
      .join("\n\n");
    termLog.scrollTop = termLog.scrollHeight;
  }

  // Browser Vision Feed
  const frames = wb.vision_frames || [];
  const vBox = document.getElementById("workbench-vision-box");
  if (frames.length > 0) {
    const latest = frames[frames.length - 1];
    vBox.innerHTML = `
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--accent-cyan);">
        Frame #${latest.frame_id} // ${escapeHtml(latest.title)} (${latest.width}x${latest.height}, ${latest.file_size_kb} KB)
      </div>
      ${
        latest.base64_png
          ? `<img src="data:image/png;base64,${latest.base64_png}" style="width:100%;max-height:190px;object-fit:contain;border:1px solid var(--border-subtle);border-radius:6px;" />`
          : ""
      }
    `;
  } else {
    vBox.innerHTML = `<div style="color:var(--text-muted);font-family:var(--font-mono);">Click "Capture Local UI Frame" to render a 1280x800 headless Chromium PNG.</div>`;
  }
}

window.rollbackPatch = async function (patchId) {
  await fetch("/api/workbenches/rollback-patch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patch_id: patchId }),
  });
  await refreshSnapshot();
};

window.rollbackGit = async function (cpId) {
  await fetch("/api/workbenches/rollback-git", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint_id_or_sha: cpId }),
  });
  await refreshSnapshot();
};

/* ==========================================================================
   6. NETWORK, WEBSOCKET & INTERACTIVE EVENT BINDINGS
   ========================================================================== */
async function refreshSnapshot() {
  const res = await fetch("/api/snapshot");
  if (res.ok) {
    const snap = await res.json();
    applySnapshot(snap);
  }
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => {
    document.getElementById("ws-status-dot").style.background = "var(--accent-emerald)";
    document.getElementById("ws-status-label").textContent = "WS: ONLINE";
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "snapshot") {
      applySnapshot(msg.snapshot);
    } else if (msg.type === "chat_stream_token") {
      handleStreamingToken(msg);
    } else if (msg.type === "chat_assistant") {
      state.streamingCardEl = null;
      state.streamingThinkText = "";
      state.streamingAnswerText = "";
      refreshSnapshot();
    } else if (msg.type === "agent_card_updated") {
      refreshSnapshot();
    } else if (msg.type === "kb_embedding_progress") {
      const bar = document.getElementById(`prog-bar-${msg.collection_id}`);
      const txt = document.getElementById(`prog-text-${msg.collection_id}`);
      if (bar) bar.style.width = `${msg.progress_pct}%`;
      if (txt) txt.textContent = `${msg.progress_pct.toFixed(1)}%`;
    }
  };

  ws.onclose = () => {
    document.getElementById("ws-status-dot").style.background = "var(--accent-amber)";
    document.getElementById("ws-status-label").textContent = "WS: RECONNECTING";
    setTimeout(connectWebSocket, 2500);
  };
}

function handleStreamingToken(msg) {
  const streamEl = document.getElementById("chat-stream");
  if (!state.streamingCardEl) {
    state.streamingThinkText = "";
    state.streamingAnswerText = "";
    const card = document.createElement("div");
    card.className = "msg-card assistant";
    card.innerHTML = `
      <div class="msg-header">
        <span>FERAL AGENT // LIVE STREAMING (STEP #${msg.step || 1})</span>
        <span style="color:var(--accent-cyan);">GENERATING...</span>
      </div>
      <details class="think-drawer" open>
        <summary class="think-summary"><span>&lt;think&gt; Live Reasoning Stream</span></summary>
        <div class="think-body" id="live-think-box"></div>
      </details>
      <div class="md-content" id="live-ans-box" style="font-family:var(--font-mono);white-space:pre-wrap;"></div>
    `;
    streamEl.appendChild(card);
    state.streamingCardEl = card;
  }
  if (msg.section === "think") {
    state.streamingThinkText += msg.text;
    const tb = state.streamingCardEl.querySelector("#live-think-box");
    if (tb) tb.textContent = state.streamingThinkText;
  } else {
    state.streamingAnswerText += msg.text;
    const ab = state.streamingCardEl.querySelector("#live-ans-box");
    if (ab) ab.textContent = state.streamingAnswerText;
  }
  streamEl.scrollTop = streamEl.scrollHeight;
}

async function sendChatMessage(promptText) {
  const text = (promptText ?? document.getElementById("chat-input").value).trim();
  if (!text) return;
  document.getElementById("chat-input").value = "";

  const streamEl = document.getElementById("chat-stream");
  streamEl.insertAdjacentHTML(
    "beforeend",
    renderChatMessageCard({ id: "NOW", role: "user", content: text })
  );
  streamEl.scrollTop = streamEl.scrollHeight;

  const btn = document.getElementById("btn-send-chat");
  btn.disabled = true;
  btn.textContent = "RUNNING...";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, max_steps: 5 }),
    });
    const data = await res.json();
    state.streamingCardEl = null;
    if (data.steps && data.steps.length > 0) {
      const lastStep = data.steps[data.steps.length - 1];
      document.getElementById("last-speed-badge").textContent = `${lastStep.tokens_per_sec} tok/s (${lastStep.vram_allocated_mib} MiB VRAM)`;
    }
    if (data.snapshot) {
      applySnapshot(data.snapshot);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "EXECUTE →";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".module-view").forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.target);
      if (target) target.classList.add("active");
    });
  });

  // Toggle Collapsible Right Context Panel
  document.getElementById("btn-toggle-context").addEventListener("click", () => {
    document.getElementById("context-sidebar").classList.toggle("collapsed");
  });

  // Context Panel Sliders & Save
  document.getElementById("ctx-temp-slider").addEventListener("input", (e) => {
    document.getElementById("ctx-temp-val").textContent = parseFloat(e.target.value).toFixed(2);
  });
  document.getElementById("ctx-tokens-slider").addEventListener("input", (e) => {
    document.getElementById("ctx-tokens-val").textContent = e.target.value;
  });
  document.getElementById("btn-save-context").addEventListener("click", async () => {
    await fetch("/api/context", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: document.getElementById("ctx-model-select").value,
        system_prompt: document.getElementById("ctx-system-prompt").value,
        temperature: parseFloat(document.getElementById("ctx-temp-slider").value),
        max_new_tokens: parseInt(document.getElementById("ctx-tokens-slider").value, 10),
      }),
    });
    await refreshSnapshot();
  });

  // Preset Mission Buttons & Chat Composer
  document.querySelectorAll(".preset-btn").forEach((b) => {
    b.addEventListener("click", () => sendChatMessage(b.dataset.prompt));
  });
  document.getElementById("btn-send-chat").addEventListener("click", () => sendChatMessage());
  document.getElementById("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
  document.getElementById("btn-clear-chat").addEventListener("click", async () => {
    await fetch("/api/chat/clear", { method: "POST" });
    await refreshSnapshot();
  });

  // Module 2: Create Agent Card & Add DAG Node
  document.getElementById("btn-create-agent").addEventListener("click", async () => {
    const name = document.getElementById("new-agent-name").value.trim() || "Autonomous Worker";
    const goal = document.getElementById("new-agent-goal").value.trim() || "Run `ast_repo_map` on `.` and summarize.";
    await fetch("/api/agents/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, goal }),
    });
    document.getElementById("new-agent-name").value = "";
    document.getElementById("new-agent-goal").value = "";
    await refreshSnapshot();
  });

  document.getElementById("btn-add-dag-node").addEventListener("click", async () => {
    const node_id = document.getElementById("dag-node-id").value.trim();
    const title = document.getElementById("dag-node-title").value.trim();
    const depsRaw = document.getElementById("dag-node-deps").value.trim();
    if (!node_id || !title) return;
    const depends_on = depsRaw ? depsRaw.split(",").map((s) => s.trim()).filter(Boolean) : [];
    await fetch("/api/dag/node", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_id,
        title,
        prompt: title,
        depends_on,
      }),
    });
    await refreshSnapshot();
  });

  // Module 3: Knowledge Base Drag-and-Drop, Snippet Upload, Collection Create & Vector Search
  const dropzone = document.getElementById("kb-dropzone");
  const fileInput = document.getElementById("kb-file-input");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      await uploadFileToKB(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", async (e) => {
    if (e.target.files.length > 0) {
      await uploadFileToKB(e.target.files[0]);
    }
  });

  async function uploadFileToKB(file) {
    const collection_id = document.getElementById("kb-target-collection").value || "feral-core-specs";
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    const content_base64 = btoa(binary);

    await fetch("/api/kb/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection_id, filename: file.name, content_base64 }),
    });
    await refreshSnapshot();
  }

  document.getElementById("btn-kb-upload-snippet").addEventListener("click", async () => {
    const collection_id = document.getElementById("kb-target-collection").value || "feral-core-specs";
    const filename = document.getElementById("kb-quick-filename").value.trim() || "note.md";
    const content_text = document.getElementById("kb-quick-content").value.trim();
    if (!content_text) return;
    await fetch("/api/kb/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection_id, filename, content_text }),
    });
    document.getElementById("kb-quick-content").value = "";
    await refreshSnapshot();
  });

  document.getElementById("btn-kb-create-col").addEventListener("click", async () => {
    const collection_id = document.getElementById("kb-new-col-id").value.trim();
    const name = document.getElementById("kb-new-col-name").value.trim();
    const description = document.getElementById("kb-new-col-desc").value.trim();
    if (!collection_id || !name) return;
    await fetch("/api/kb/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection_id, name, description }),
    });
    await refreshSnapshot();
  });

  document.getElementById("btn-kb-search").addEventListener("click", async () => {
    const collection_id = document.getElementById("kb-target-collection").value || "feral-core-specs";
    const query = document.getElementById("kb-query-input").value.trim();
    if (!query) return;
    const res = await fetch("/api/kb/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection_id, query, top_k: 3 }),
    });
    const data = await res.json();
    const listEl = document.getElementById("kb-results-list");
    listEl.innerHTML = (data.results || [])
      .map(
        (r) => `
        <div class="dag-node-card">
          <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:11px;color:var(--accent-emerald);">
            <span>${escapeHtml(r.doc_name)} (chunk #${r.chunk_index})</span>
            <span>Cosine Similarity: ${r.similarity.toFixed(4)}</span>
          </div>
          <div style="font-size:12px;margin-top:4px;color:var(--text-primary);">${escapeHtml(r.content)}</div>
        </div>
      `
      )
      .join("");
  });

  // Module 4: Save Integrations Panel
  document.getElementById("btn-save-integrations").addEventListener("click", async () => {
    await fetch("/api/integrations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        searxng_enabled: document.getElementById("int-searxng-enabled").checked,
        searxng_endpoint: document.getElementById("int-searxng-endpoint").value.trim(),
        headless_browser_enabled: document.getElementById("int-headless-enabled").checked,
        vision_screenshots_enabled: document.getElementById("int-vision-enabled").checked,
        email_enabled: document.getElementById("int-email-enabled").checked,
        imap_host: document.getElementById("int-imap-host").value.trim(),
        imap_port: parseInt(document.getElementById("int-imap-port").value, 10),
        smtp_host: document.getElementById("int-smtp-host").value.trim(),
        smtp_port: parseInt(document.getElementById("int-smtp-port").value, 10),
        email_account: document.getElementById("int-email-account").value.trim(),
        calendar_sync_enabled: document.getElementById("int-calendar-enabled").checked,
        caldav_endpoint: document.getElementById("int-caldav-endpoint").value.trim(),
        loop_watchdog_enabled: document.getElementById("int-watchdog-enabled").checked,
        git_time_travel_enabled: document.getElementById("int-timetravel-enabled").checked,
      }),
    });
    await refreshSnapshot();
  });

  // Module 5: Workbenches (Terminal & Browser Vision)
  document.getElementById("btn-run-term").addEventListener("click", async () => {
    const command = document.getElementById("term-cmd-input").value.trim();
    if (!command) return;
    await fetch("/api/workbenches/terminal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });
    await refreshSnapshot();
  });

  document.getElementById("btn-capture-vision").addEventListener("click", async () => {
    await fetch("/api/workbenches/vision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "Feral Command Center UI", url: location.origin }),
    });
    await refreshSnapshot();
  });

  // Support ?tab=agents|kb|integrations|workbenches deep-linking
  const params = new URLSearchParams(location.search);
  const tabParam = params.get("tab");
  if (tabParam) {
    const targetId = tabParam.startsWith("view-") ? tabParam : `view-${tabParam}`;
    const tabBtn = document.querySelector(`.nav-tab[data-target="${targetId}"]`);
    if (tabBtn) tabBtn.click();
  }

  refreshSnapshot();
  connectWebSocket();
});
