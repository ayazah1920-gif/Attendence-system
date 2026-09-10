const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let capturedSamples = [];
const REQUIRED_SAMPLES = 5;

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => { video.srcObject = stream; })
  .catch(err => {
    document.getElementById("recognizeResult").innerHTML =
      `<span class="error">Camera access nahi mila: ${err.message}</span>`;
  });

function captureFrame() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.9);
}

// ---------- Mark Attendance ----------
document.getElementById("markBtn").addEventListener("click", async () => {
  const resultBox = document.getElementById("recognizeResult");
  resultBox.innerHTML = "Processing...";
  const image = captureFrame();

  try {
    const res = await fetch("/api/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image })
    });
    const data = await res.json();

    if (!data.success) {
      resultBox.innerHTML = `<span class="error">${data.message}</span>`;
      return;
    }

    resultBox.innerHTML = data.results.map(r => {
      const cls = r.status === "Not recognized" ? "error" : "success";
      return `<div class="${cls}">${r.name} ${r.roll_no ? "(" + r.roll_no + ")" : ""} — ${r.status}</div>`;
    }).join("");

    loadAttendance();
  } catch (e) {
    resultBox.innerHTML = `<span class="error">Error: ${e.message}</span>`;
  }
});

// ---------- Register Student ----------
document.getElementById("captureBtn").addEventListener("click", () => {
  if (capturedSamples.length >= REQUIRED_SAMPLES) {
    capturedSamples = [];
  }
  const img = captureFrame();
  capturedSamples.push(img);
  document.getElementById("sampleCount").innerText =
    `Samples captured: ${capturedSamples.length} / ${REQUIRED_SAMPLES}`;

  if (capturedSamples.length >= REQUIRED_SAMPLES) {
    document.getElementById("registerBtn").disabled = false;
  }
});

document.getElementById("registerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = document.getElementById("studentName").value.trim();
  const roll_no = document.getElementById("rollNo").value.trim();
  const resultBox = document.getElementById("registerResult");

  if (capturedSamples.length < 3) {
    resultBox.innerHTML = `<span class="error">Pehle "Capture Samples" button se face samples lein.</span>`;
    return;
  }

  resultBox.innerHTML = "Registering...";

  try {
    const res = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, roll_no, images: capturedSamples })
    });
    const data = await res.json();
    resultBox.innerHTML = `<span class="${data.success ? 'success' : 'error'}">${data.message}</span>`;

    if (data.success) {
      capturedSamples = [];
      document.getElementById("sampleCount").innerText = `Samples captured: 0 / ${REQUIRED_SAMPLES}`;
      document.getElementById("registerBtn").disabled = true;
      document.getElementById("registerForm").reset();
    }
  } catch (err) {
    resultBox.innerHTML = `<span class="error">Error: ${err.message}</span>`;
  }
});

// ---------- Attendance Table ----------
async function loadAttendance() {
  const dateFilter = document.getElementById("dateFilter");
  const day = dateFilter.value || new Date().toISOString().split("T")[0];
  dateFilter.value = day;

  const res = await fetch(`/api/attendance?date=${day}`);
  const rows = await res.json();
  const tbody = document.querySelector("#attendanceTable tbody");
  tbody.innerHTML = rows.map(r =>
    `<tr><td>${r.name}</td><td>${r.roll_no}</td><td>${r.date}</td><td>${r.time}</td></tr>`
  ).join("") || `<tr><td colspan="4">Koi record nahi mila.</td></tr>`;
}

document.getElementById("refreshBtn").addEventListener("click", loadAttendance);
document.getElementById("exportBtn").addEventListener("click", () => {
  const day = document.getElementById("dateFilter").value || new Date().toISOString().split("T")[0];
  window.location.href = `/api/attendance/export?date=${day}`;
});

// Init
document.getElementById("dateFilter").value = new Date().toISOString().split("T")[0];
loadAttendance();