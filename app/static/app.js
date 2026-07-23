const state = {
  threadId: null,
  paperIds: [],
  selectedPaperIds: new Set(),
  isStreaming: false,
};

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const statusPill = document.querySelector("#status-pill");
const paperList = document.querySelector("#paper-list");

document.querySelector("#new-chat").addEventListener("click", () => {
  state.threadId = null;
  messagesEl.innerHTML = "";
  addMessage("assistant", "Started a new chat. Ask a question when you are ready.");
  inputEl.focus();
});

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
  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
          setBusy(true, event.data.message || "Working");
        } else if (event.name === "token") {
          streamedText += event.data.text || "";
          assistantBubble.textContent = streamedText;
          scrollToBottom();
          setBusy(true, "Writing");
        } else if (event.name === "final") {
          state.threadId = event.data.thread?.thread_id || state.threadId;
          if (!streamedText) {
            assistantBubble.textContent = answerText(event.data.final_answer);
          }
          loadPapers();
        } else if (event.name === "error") {
          assistantBubble.textContent = event.data.message || "The request failed.";
        }
      }
    }
  } catch (error) {
    assistantBubble.textContent = error.message || "The request failed.";
  } finally {
    setBusy(false, "Ready");
    scrollToBottom();
  }
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

function answerText(finalAnswer) {
  if (!finalAnswer) return "";
  const answer = finalAnswer.answer;
  if (typeof answer === "string") return answer;
  return JSON.stringify(answer ?? finalAnswer, null, 2);
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
    const item = document.createElement("button");
    item.className = "paper-item";
    item.type = "button";
    if (state.selectedPaperIds.has(paper.paper_id)) {
      item.classList.add("selected");
    }
    item.addEventListener("click", () => togglePaperSelection(paper.paper_id));
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
    item.append(title, meta);
    paperList.append(item);
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

function setBusy(isBusy, label) {
  state.isStreaming = isBusy;
  sendButton.disabled = isBusy;
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

loadPapers();
