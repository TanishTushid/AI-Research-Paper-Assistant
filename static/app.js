const fileInput = document.querySelector("#file-input");
const uploadZone = document.querySelector("#upload-zone");
const uploadMessage = document.querySelector("#upload-message");
const documentList = document.querySelector("#document-list");
const questionForm = document.querySelector("#question-form");
const questionInput = document.querySelector("#question-input");
const topKSelect = document.querySelector("#top-k");
const answerCard = document.querySelector("#answer-card");
const appStatus = document.querySelector("#app-status");

function setStatus(message, busy = false) {
  appStatus.lastChild.textContent = message;
  appStatus.classList.toggle("busy", busy);
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function createDocumentRow(document) {
  const row = documentCreate("article", "document-row");
  const badge = documentCreate("span", "pdf-badge", "PDF");
  const info = documentCreate("div", "document-info");
  const name = documentCreate("div", "document-name", document.name);
  const details = [];
  if (Number.isInteger(document.pages) && document.pages > 0) {
    details.push(`${document.pages} page${document.pages === 1 ? "" : "s"}`);
  }
  details.push(formatSize(document.size));
  const meta = documentCreate("div", "document-meta", details.join(" · "));
  const actions = documentCreate("div", "document-actions");

  info.append(name, meta);
  actions.append(documentCreate(
    "span",
    document.indexed ? "indexed-check" : "not-indexed",
    document.indexed ? "✓" : "Indexing required",
  ));
  const remove = documentCreate("button", "remove-document");
  remove.type = "button";
  remove.setAttribute("aria-label", `Remove ${document.name}`);
  remove.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7m4 3v6m4-6v6"/></svg>';
  remove.addEventListener("click", () => removeDocument(document.id));
  actions.append(remove);
  row.append(badge, info, actions);
  return row;
}

function documentCreate(tag, className, text) {
  const element = window.document.createElement(tag);
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderDocuments(documents) {
  documentList.replaceChildren(...documents.map(createDocumentRow));
  const count = documents.length;
  document.querySelector("#library-count").textContent =
    `${count} document${count === 1 ? "" : "s"} loaded`;
  document.querySelector("#document-total").textContent = `Total: ${count} file(s)`;
  document.querySelector("#empty-documents").hidden = count > 0;
  document.querySelector("#clear-documents").disabled = count === 0;
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/documents");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load the document library.");
    renderDocuments(data.documents);
  } catch (error) {
    uploadMessage.textContent = error.message;
    uploadMessage.classList.add("error");
  }
}

async function uploadFiles(files) {
  if (!files.length) return;
  const formData = new FormData();
  for (const file of files) formData.append("files", file);
  uploadMessage.textContent = "Uploading and indexing your papers locally…";
  uploadMessage.classList.remove("error");
  setStatus("Indexing", true);
  uploadZone.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await response.json();
    if (data.documents) renderDocuments(data.documents);
    const messages = (data.results || []).map((result) =>
      result.error ? `${result.name}: ${result.error}` : `${result.name} added to your library.`,
    );
    if (!response.ok && !messages.length) throw new Error(data.error || "Upload failed.");
    uploadMessage.textContent = messages.join(" ");
    uploadMessage.classList.toggle("error", (data.results || []).some((result) => result.error));
  } catch (error) {
    uploadMessage.textContent = error.message;
    uploadMessage.classList.add("error");
  } finally {
    setStatus("Ready");
    uploadZone.removeAttribute("aria-busy");
    fileInput.value = "";
  }
}

async function removeDocument(id) {
  if (!window.confirm("Remove this paper from your library?")) return;
  setStatus("Updating library", true);
  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not remove this paper.");
    renderDocuments(data.documents);
  } catch (error) {
    window.alert(error.message);
  } finally {
    setStatus("Ready");
  }
}

async function clearDocuments() {
  if (!window.confirm("Remove all research papers and their indexed passages from this library?")) return;
  setStatus("Clearing library", true);
  try {
    const response = await fetch("/api/documents", { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not clear the library.");
    renderDocuments(data.documents);
    answerCard.hidden = true;
    uploadMessage.textContent = "Your research library has been cleared.";
    uploadMessage.classList.remove("error");
  } catch (error) {
    window.alert(error.message);
  } finally {
    setStatus("Ready");
  }
}

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }

  const askButton = document.querySelector("#ask-button");
  askButton.disabled = true;
  setStatus("Searching papers", true);
  answerCard.hidden = false;
  document.querySelector("#answer-text").textContent = "Searching your research library…";
  document.querySelector("#citation-list").replaceChildren();
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: Number(topKSelect.value) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not answer this question.");
    document.querySelector("#answer-text").textContent = data.answer;
    const citationList = document.querySelector("#citation-list");
    for (const citation of data.citations) {
      const label = citation.page === "Unknown page"
        ? `${citation.source}`
        : `${citation.source} · Page ${citation.page}`;
      const pill = documentCreate("span", "citation-pill", label);
      citationList.append(pill);
    }
    if (!data.citations.length) {
      citationList.append(documentCreate("span", "document-meta", "No source citations were returned."));
    }
  } catch (error) {
    document.querySelector("#answer-text").textContent = error.message;
    document.querySelector("#citation-list").replaceChildren();
  } finally {
    askButton.disabled = false;
    setStatus("Ready");
  }
});

fileInput.addEventListener("change", () => uploadFiles([...fileInput.files]));
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("dragging");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragging"));
uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.classList.remove("dragging");
  uploadFiles([...event.dataTransfer.files]);
});
document.querySelector("#clear-documents").addEventListener("click", clearDocuments);
document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.textContent;
    questionInput.focus();
  });
});

function showPage(name) {
  document.querySelectorAll(".page-panel").forEach((panel) => {
    const active = panel.dataset.page === name;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle("active", link.dataset.pageLink === name);
  });
}
document.querySelectorAll("[data-page-link]").forEach((link) => {
  link.addEventListener("click", () => showPage(link.dataset.pageLink));
});

const defaultTopK = document.querySelector("#default-top-k");
const savedTopK = localStorage.getItem("research-assistant-top-k");
if (savedTopK && [...topKSelect.options].some((option) => option.value === savedTopK)) {
  topKSelect.value = savedTopK;
}
defaultTopK.value = topKSelect.value;
defaultTopK.addEventListener("change", () => {
  topKSelect.value = defaultTopK.value;
  localStorage.setItem("research-assistant-top-k", defaultTopK.value);
});
topKSelect.addEventListener("change", () => {
  defaultTopK.value = topKSelect.value;
  localStorage.setItem("research-assistant-top-k", topKSelect.value);
});

loadDocuments();
