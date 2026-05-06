const BACKEND_URL = "http://127.0.0.1:5000";

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

let selectedFile = null;

function setStatus(text) {
  statusEl.textContent = text || "";
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  selectedFile = file || null;
  resultEl.textContent = "";

  if (!selectedFile) {
    preview.style.display = "none";
    preview.src = "";
    predictBtn.disabled = true;
    setStatus("");
    return;
  }

  const url = URL.createObjectURL(selectedFile);
  preview.src = url;
  preview.style.display = "block";
  predictBtn.disabled = false;
  setStatus("Ready.");
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  setStatus("Predicting...");
  resultEl.textContent = "";

  try {
    const formData = new FormData();
    formData.append("image", selectedFile, selectedFile.name);

    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Request failed");
    }

    resultEl.textContent = JSON.stringify(data, null, 2);
    setStatus(`Prediction: ${data.display_name_tr} (${data.confidence}%)`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
  }
});

