const state = {
  threadId: null,
  threads: [],
  paperIds: [],
  selectedPaperIds: new Set(),
  pendingDiscoveredPapers: [],
  isStreaming: false,
};

const CHAT_CLIENT_TIMEOUT_MS = 135000;

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const statusPill = document.querySelector("#status-pill");
const threadList = document.querySelector("#thread-list");
const paperList = document.querySelector("#paper-list");
const inputPlaceholder = inputEl.getAttribute("placeholder") || "";

document.querySelector("#new-chat").addEventListener("click", () => {
  startNewChat();
});

document.querySelector("#refresh-threads").addEventListener("click", loadThreads);
document.querySelector("#cleanup-unsaved-papers").addEventListener("click", cleanupUnsavedPapers);
document.querySelector("#refresh-papers").addEventListener("click", loadPapers);

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message || state.isStreaming) return;

  inputEl.value = "";
  resizeInput();
  addMessage("user", message);
  const assistantBubble = addMessage("assistant", "");
  await streamChat(message, assistantBubble);
});

inputEl.addEventListener("input", resizeInput);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

async function streamChat(message, assistantBubble) {
  setBusy(true, "Planning");
  setLoadingBubble(assistantBubble, "Planning and retrieving evidence...");
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => {
    controller.abort();
  }, CHAT_CLIENT_TIMEOUT_MS);
  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        thread_id: state.threadId,
        message,
        title: "Research chat",
        user_id: "local-user",
        active_paper_ids: activePaperIds(),
        max_steps: 8,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamedText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const event = parseSseEvent(part);
        if (!event) continue;
        if (event.name === "status") {
          const label = streamStatusLabel(event.data);
          setBusy(true, label);
          updateLoadingBubble(assistantBubble, label);
        } else if (event.name === "token") {
          streamedText += event.data.text || "";
          clearLoadingBubble(assistantBubble);
          assistantBubble.textContent = streamedText;
          scrollToBottom();
          setBusy(true, "Writing");
        } else if (event.name === "final") {
          state.threadId = event.data.thread?.thread_id || state.threadId;
          const finalText = finalDisplayText(event.data);
          clearLoadingBubble(assistantBubble);
          assistantBubble.textContent = finalText;
          renderDiscoveredPaperPrompt(event.data.discovered_papers || []);
          loadThreads();
          loadPapers();
        } else if (event.name === "error") {
          clearLoadingBubble(assistantBubble);
          assistantBubble.textContent = event.data.message || "The request failed.";
        }
      }
    }
  } catch (error) {
    clearLoadingBubble(assistantBubble);
    assistantBubble.textContent = (
      error.name === "AbortError"
        ? "The request took too long to answer. Please narrow the question, select fewer papers, or try again."
        : error.message || "The request failed."
    );
  } finally {
    window.clearTimeout(timeoutId);
    setBusy(false, "Ready");
    scrollToBottom();
  }
}

function startNewChat() {
  state.threadId = null;
  messagesEl.innerHTML = "";
  renderThreads(state.threads);
  addMessage("assistant", "Started a new chat. Ask a question when you are ready.");
  inputEl.focus();
}

function parseSseEvent(raw) {
  const lines = raw.split("\n");
  const nameLine = lines.find((line) => line.startsWith("event:"));
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!nameLine || !dataLine) return null;
  try {
    return {
      name: nameLine.slice(6).trim(),
      data: JSON.parse(dataLine.slice(5).trim()),
    };
  } catch {
    return null;
  }
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "A";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  article.append(avatar, bubble);
  messagesEl.append(article);
  scrollToBottom();
  return bubble;
}

function addLoadedMessage(message) {
  if (!message || !["user", "assistant", "system"].includes(message.role)) return;
  if (message.role === "system" && message.metadata_json?.hidden_from_ui) return;
  addMessage(
    message.role === "system" ? "assistant" : message.role,
    message.content || ""
  );
}

function setLoadingBubble(bubble, label) {
  bubble.classList.add("loading-bubble");
  bubble.innerHTML = "";

  const spinner = document.createElement("span");
  spinner.className = "loading-spinner";
  spinner.setAttribute("aria-hidden", "true");

  const text = document.createElement("span");
  text.className = "loading-text";
  text.textContent = label;

  bubble.append(spinner, text);
}

function updateLoadingBubble(bubble, label) {
  if (!bubble.classList.contains("loading-bubble")) return;
  const text = bubble.querySelector(".loading-text");
  if (text) {
    text.textContent = label;
  }
}

function clearLoadingBubble(bubble) {
  bubble.classList.remove("loading-bubble");
  bubble.innerHTML = "";
}

function streamStatusLabel(data) {
  const rawMessage = data.message || "Working";
  if (rawMessage === "started") {
    return "Planning and searching papers...";
  }
  if (rawMessage === "Still working" && data.elapsed_seconds !== undefined) {
    return `Still working... ${data.elapsed_seconds}s`;
  }
  return rawMessage;
}

function answerText(finalAnswer) {
  if (!finalAnswer) return "";
  const answer = finalAnswer.answer;
  if (typeof answer === "string") return answer;
  return JSON.stringify(answer ?? finalAnswer, null, 2);
}

function finalDisplayText(payload) {
  const answer = answerText(payload?.final_answer);
  if (answer) return answer;

  const toolSummary = lastToolSummary(payload?.tool_history || []);
  if (toolSummary && payload?.last_error) {
    return `${toolSummary}\n\n${payload.last_error}`;
  }
  if (toolSummary) return toolSummary;
  if (payload?.last_error) {
    return `I could not complete this request: ${payload.last_error}`;
  }
  if (payload?.status && payload.status !== "success") {
    return `The request finished with status ${payload.status}, but no answer was produced.`;
  }
  return "No answer was produced.";
}

function lastToolSummary(toolHistory) {
  for (let index = toolHistory.length - 1; index >= 0; index -= 1) {
    const summary = toolHistory[index]?.summary;
    if (typeof summary === "string" && summary.trim()) {
      return summary.trim();
    }
  }
  return "";
}

function renderDiscoveredPaperPrompt(papers) {
  const discovered = dedupePapers(papers);
  state.pendingDiscoveredPapers = discovered;
  if (!discovered.length) return;

  const article = document.createElement("article");
  article.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "A";

  const panel = document.createElement("div");
  panel.className = "bubble discovery-panel";

  const heading = document.createElement("div");
  heading.className = "discovery-heading";
  const title = document.createElement("h3");
  title.textContent = "Save these papers?";
  const subtitle = document.createElement("p");
  subtitle.textContent = "Keep selected papers in your workspace, or prepare them for RAG.";
  heading.append(title, subtitle);

  const list = document.createElement("div");
  list.className = "discovery-list";
  for (const paper of discovered) {
    list.append(discoveryPaperRow(paper));
  }

  const actions = document.createElement("div");
  actions.className = "discovery-actions";

  const saveButton = document.createElement("button");
  saveButton.type = "button";
  saveButton.className = "secondary-button";
  saveButton.textContent = "Save metadata";
  saveButton.addEventListener(
    "click",
    () => saveDiscoveredPapers(panel, discovered, false)
  );

  const prepareButton = document.createElement("button");
  prepareButton.type = "button";
  prepareButton.className = "primary-small-button";
  prepareButton.textContent = "Save + prepare RAG";
  prepareButton.addEventListener(
    "click",
    () => saveDiscoveredPapers(panel, discovered, true)
  );

  const status = document.createElement("div");
  status.className = "discovery-status";
  status.dataset.role = "save-status";

  actions.append(saveButton, prepareButton);
  panel.append(heading, list, actions, status);
  article.append(avatar, panel);
  messagesEl.append(article);
  scrollToBottom();
}

function discoveryPaperRow(paper) {
  const label = document.createElement("label");
  label.className = "discovery-paper";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = true;
  checkbox.value = paper.paper_id || "";
  checkbox.dataset.paperId = paper.paper_id || "";

  const body = document.createElement("span");
  body.className = "discovery-paper-body";

  const title = document.createElement("span");
  title.className = "discovery-paper-title";
  title.textContent = paper.title || paper.paper_id || "Untitled paper";

  const meta = document.createElement("span");
  meta.className = "discovery-paper-meta";
  meta.textContent = [
    paper.paper_id,
    paper.published_date,
    (paper.authors || []).slice(0, 2).join(", "),
  ].filter(Boolean).join(" | ");

  body.append(title, meta);
  label.append(checkbox, body);
  return label;
}

async function saveDiscoveredPapers(panel, papers, prepareForRag) {
  const selectedIds = Array.from(
    panel.querySelectorAll("input[type='checkbox']:checked")
  ).map((input) => input.dataset.paperId).filter(Boolean);
  const status = panel.querySelector("[data-role='save-status']");
  if (!selectedIds.length) {
    status.textContent = "Select at least one paper.";
    return;
  }

  const buttons = panel.querySelectorAll("button");
  buttons.forEach((button) => { button.disabled = true; });
  status.textContent = prepareForRag ? "Saving and preparing..." : "Saving...";
  setBusy(true, prepareForRag ? "Preparing RAG" : "Saving");

  try {
    const response = await fetch("/papers/save-discovered", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: state.threadId,
        papers,
        paper_ids: selectedIds,
        knowledge_base_id: "default",
        prepare_for_rag: prepareForRag,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not save papers.");
    }
    status.textContent = payload.summary || "Saved selected papers.";
    if (prepareForRag && payload.prepare_job?.job_id) {
      const job = await pollIngestionJob(payload.prepare_job.job_id, status);
      const readyIds = readyPaperIds(job?.result);
      replaceActivePaperSelection(readyIds.length ? readyIds : selectedIds);
    }
    await loadPapers();
  } catch (error) {
    status.textContent = error.message || "Could not save papers.";
    buttons.forEach((button) => { button.disabled = false; });
  } finally {
    setBusy(false, "Ready");
  }
}

async function pollIngestionJob(jobId, statusEl) {
  const terminalStatuses = new Set(["success", "partial_success", "failed"]);
  while (true) {
    await sleep(2000);
    try {
      const response = await fetch(`/ingestion-jobs/${encodeURIComponent(jobId)}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Could not load preparation status.");
      }
      const job = payload.job || {};
      if (job.status === "queued" || job.status === "running") {
        statusEl.textContent = `RAG preparation ${job.status}...`;
        continue;
      }
      if (terminalStatuses.has(job.status)) {
        const resultSummary = job.result?.summary;
        statusEl.textContent = resultSummary || (
          job.status === "failed"
            ? `RAG preparation failed: ${job.error || "unknown error"}`
            : `RAG preparation finished with status ${job.status}.`
        );
        await loadPapers();
        return job;
      }
      statusEl.textContent = `RAG preparation status: ${job.status || "unknown"}.`;
    } catch (error) {
      statusEl.textContent = error.message || "Could not load preparation status.";
      return null;
    }
  }
}

function readyPaperIds(result) {
  const ids = [];
  for (const key of ["ready_paper_ids", "already_ready_paper_ids"]) {
    for (const paperId of result?.[key] || []) {
      if (paperId && !ids.includes(paperId)) ids.push(paperId);
    }
  }
  return ids;
}

function replaceActivePaperSelection(paperIds) {
  state.selectedPaperIds = new Set((paperIds || []).filter(Boolean));
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function dedupePapers(papers) {
  const byId = new Map();
  for (const paper of papers) {
    if (!paper || !paper.paper_id || byId.has(paper.paper_id)) continue;
    byId.set(paper.paper_id, paper);
  }
  return Array.from(byId.values());
}

async function loadPapers() {
  try {
    const response = await fetch("/papers?limit=50");
    if (!response.ok) throw new Error("Could not load papers.");
    const payload = await response.json();
    renderPapers(payload.papers || []);
  } catch (error) {
    paperList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

async function loadThreads() {
  try {
    const response = await fetch("/threads?user_id=local-user&limit=30");
    if (!response.ok) throw new Error("Could not load chats.");
    const payload = await response.json();
    state.threads = payload.threads || [];
    renderThreads(state.threads);
  } catch (error) {
    threadList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderThreads(threads) {
  threadList.innerHTML = "";
  if (!threads.length) {
    threadList.innerHTML = '<div class="empty-state">No saved chats yet.</div>';
    return;
  }
  for (const thread of threads) {
    const item = document.createElement("div");
    item.className = "thread-item";
    if (thread.thread_id === state.threadId) {
      item.classList.add("selected");
    }

    const openButton = document.createElement("button");
    openButton.className = "thread-open";
    openButton.type = "button";
    openButton.addEventListener("click", () => loadThread(thread.thread_id));

    const title = document.createElement("span");
    title.className = "thread-title";
    title.textContent = thread.title || "Untitled chat";

    const meta = document.createElement("span");
    meta.className = "thread-meta";
    meta.textContent = formatThreadTime(thread.updated_at || thread.created_at);

    const deleteButton = document.createElement("button");
    deleteButton.className = "row-delete-button";
    deleteButton.type = "button";
    deleteButton.title = "Delete chat";
    deleteButton.setAttribute("aria-label", `Delete chat ${thread.title || thread.thread_id}`);
    deleteButton.innerHTML = trashIconSvg();
    deleteButton.addEventListener("click", () => deleteThread(thread.thread_id));

    openButton.append(title, meta);
    item.append(openButton, deleteButton);
    threadList.append(item);
  }
}

async function loadThread(threadId) {
  if (!threadId || state.isStreaming) return;
  setBusy(true, "Loading chat");
  try {
    const response = await fetch(
      `/threads/${encodeURIComponent(threadId)}/messages?limit=100`
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not load chat.");
    }
    state.threadId = threadId;
    messagesEl.innerHTML = "";
    for (const message of payload.messages || []) {
      addLoadedMessage(message);
    }
    if (!(payload.messages || []).length) {
      addMessage("assistant", "This chat does not have messages yet.");
    }
    renderThreads(state.threads);
  } catch (error) {
    messagesEl.innerHTML = "";
    addMessage("assistant", error.message || "Could not load chat.");
  } finally {
    setBusy(false, "Ready");
    inputEl.focus();
  }
}

function formatThreadTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderPapers(papers) {
  paperList.innerHTML = "";
  state.paperIds = papers.map((paper) => paper.paper_id).filter(Boolean);
  if (state.selectedPaperIds.size === 0 && state.paperIds.length === 1) {
    state.selectedPaperIds.add(state.paperIds[0]);
  }
  if (!papers.length) {
    paperList.innerHTML = '<div class="empty-state">No papers loaded yet.</div>';
    return;
  }
  for (const paper of papers) {
    const item = document.createElement("div");
    item.className = "paper-item";
    if (state.selectedPaperIds.has(paper.paper_id)) {
      item.classList.add("selected");
    }

    const openButton = document.createElement("button");
    openButton.className = "paper-open";
    openButton.type = "button";
    openButton.addEventListener("click", () => togglePaperSelection(paper.paper_id));

    const title = document.createElement("p");
    title.className = "paper-title";
    title.textContent = paper.title || paper.paper_id || "Untitled paper";
    const meta = document.createElement("div");
    meta.className = "paper-meta";
    meta.textContent = [
      paper.paper_id,
      paper.published_date,
      (paper.authors || []).slice(0, 2).join(", "),
    ].filter(Boolean).join(" | ");

    const deleteButton = document.createElement("button");
    deleteButton.className = "row-delete-button";
    deleteButton.type = "button";
    deleteButton.title = "Remove paper";
    deleteButton.setAttribute("aria-label", `Remove paper ${paper.title || paper.paper_id}`);
    deleteButton.innerHTML = trashIconSvg();
    deleteButton.addEventListener("click", () => deletePaper(paper.paper_id));

    openButton.append(title, meta);
    item.append(openButton, deleteButton);
    paperList.append(item);
  }
}

async function deleteThread(threadId) {
  if (!threadId || state.isStreaming) return;
  if (!window.confirm("Delete this chat?")) return;
  setBusy(true, "Deleting chat");
  try {
    const response = await fetch(`/threads/${encodeURIComponent(threadId)}`, {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not delete chat.");
    }
    state.threads = state.threads.filter((thread) => thread.thread_id !== threadId);
    if (state.threadId === threadId) {
      state.threadId = null;
      messagesEl.innerHTML = "";
      addMessage("assistant", "Deleted the selected chat.");
    }
    renderThreads(state.threads);
  } catch (error) {
    addMessage("assistant", error.message || "Could not delete chat.");
  } finally {
    setBusy(false, "Ready");
  }
}

async function deletePaper(paperId) {
  if (!paperId || state.isStreaming) return;
  if (!window.confirm("Remove this paper, fetched files, and vector chunks?")) return;
  setBusy(true, "Removing paper");
  try {
    const response = await fetch(`/papers/${encodeURIComponent(paperId)}`, {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not remove paper.");
    }
    state.selectedPaperIds.delete(paperId);
    await loadPapers();
    addMessage("assistant", payload.summary || "Removed paper from workspace.");
  } catch (error) {
    addMessage("assistant", error.message || "Could not remove paper.");
  } finally {
    setBusy(false, "Ready");
  }
}

async function cleanupUnsavedPapers() {
  if (state.isStreaming) return;
  if (!window.confirm("Remove metadata for papers that were not saved?")) return;
  setBusy(true, "Cleaning papers");
  try {
    const response = await fetch("/papers/unsaved", { method: "DELETE" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not clean paper metadata.");
    }
    const remaining = new Set((payload.detail?.requested_paper_ids || []));
    for (const paperId of remaining) {
      state.selectedPaperIds.delete(paperId);
    }
    await loadPapers();
    addMessage("assistant", payload.summary || "Cleaned unsaved paper metadata.");
  } catch (error) {
    addMessage("assistant", error.message || "Could not clean paper metadata.");
  } finally {
    setBusy(false, "Ready");
  }
}

function togglePaperSelection(paperId) {
  if (!paperId) return;
  if (state.selectedPaperIds.has(paperId)) {
    state.selectedPaperIds.delete(paperId);
  } else {
    state.selectedPaperIds.add(paperId);
  }
  loadPapers();
}

function activePaperIds() {
  if (state.selectedPaperIds.size > 0) {
    return Array.from(state.selectedPaperIds);
  }
  if (state.paperIds.length === 1) {
    return [state.paperIds[0]];
  }
  return [];
}

function trashIconSvg() {
  return `
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M5 6l1 14h12l1-14" />
    </svg>
  `;
}

function setBusy(isBusy, label) {
  state.isStreaming = isBusy;
  sendButton.disabled = isBusy;
  inputEl.disabled = isBusy;
  inputEl.placeholder = isBusy ? "Waiting for the current answer..." : inputPlaceholder;
  statusPill.textContent = label;
}

function resizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(value) {
  const span = document.createElement("span");
  span.textContent = value;
  return span.innerHTML;
}

loadThreads();
loadPapers();
