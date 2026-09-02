const statusEl = document.getElementById("status");
const savedEl = document.getElementById("saved");
const form = document.getElementById("keys");

function render(data) {
  statusEl.classList.remove("status--pending", "status--ok", "status--warn");
  if (data.ready) {
    statusEl.classList.add("status--ok");
    statusEl.textContent = `Ready — ${data.state.toLowerCase()}. Boop an antenna to ask.`;
  } else {
    statusEl.classList.add("status--warn");
    statusEl.textContent = `Needs a key: ${data.missing.join(" and ")}.`;
  }
  // Keys are never sent back, so show placeholders rather than values.
  if (data.anthropic_set) document.getElementById("anthropic").placeholder = "•••••••• (set)";
  if (data.tavily_set) document.getElementById("tavily").placeholder = "•••••••• (set)";
  if (data.voice) document.getElementById("voice").placeholder = data.voice;
}

async function refresh() {
  try {
    render(await (await fetch("api/status")).json());
  } catch {
    statusEl.textContent = "Can't reach the app.";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    anthropic_api_key: document.getElementById("anthropic").value,
    tavily_api_key: document.getElementById("tavily").value,
    voice: document.getElementById("voice").value,
  };
  await fetch("api/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  form.reset();
  savedEl.hidden = false;
  setTimeout(() => (savedEl.hidden = true), 2500);
  refresh();
});

refresh();
setInterval(refresh, 4000);

const askBtn = document.getElementById("ask");
askBtn.addEventListener("click", async () => {
  askBtn.disabled = true;
  try {
    const res = await (await fetch("api/ask", { method: "POST" })).json();
    askBtn.textContent = res.accepted ? "🎤 Listening…" : `Busy (${res.reason})`;
  } catch {
    askBtn.textContent = "Can't reach the app";
  }
  setTimeout(() => { askBtn.textContent = "🎤 Ask now"; askBtn.disabled = false; }, 3000);
});
