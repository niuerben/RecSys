const form = document.querySelector("#upload-form");
const filesInput = document.querySelector("#files");
const fileList = document.querySelector("#file-list");
const message = document.querySelector("#message");
const resultBody = document.querySelector("#result-body");
const submitButton = form.querySelector("button");

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "";
  }
  return Number(value).toFixed(4);
}

function renderTable(rows) {
  resultBody.innerHTML = "";
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    if (index === rows.length - 1 && row.method !== "XSimGCL") {
      tr.className = "user-row";
    }
    tr.innerHTML = `
      <td>${row.method}</td>
      <td>${fmt(row.recall)}</td>
      <td>${fmt(row.mrr)}</td>
      <td>${fmt(row.ndcg)}</td>
      <td>${fmt(row.hit)}</td>
      <td>${fmt(row.precision)}</td>
    `;
    resultBody.appendChild(tr);
  });
}

function setMessage(text, kind = "") {
  message.textContent = text;
  message.className = `message ${kind}`;
}

async function loadStandard() {
  const response = await fetch("/api/standard");
  const payload = await response.json();
  renderTable(payload.table);
}

filesInput.addEventListener("change", () => {
  const names = Array.from(filesInput.files).map((file) => file.name);
  fileList.textContent = names.length ? names.join(", ") : "尚未选择文件";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!filesInput.files.length) {
    setMessage("请先选择 CSV 或 ZIP 文件。", "error");
    return;
  }

  const data = new FormData(form);
  submitButton.disabled = true;
  setMessage("测评中...");

  try {
    const response = await fetch("/api/evaluate", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "测评失败");
    }
    renderTable(payload.table);
    setMessage("测评完成。", "ok");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});

loadStandard().catch((error) => setMessage(error.message, "error"));
