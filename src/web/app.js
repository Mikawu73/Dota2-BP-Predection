"use strict";

const $ = (id) => document.getElementById(id);
let metadata, heroes = new Map(), history = [], undoStack = [], redoStack = [];
let revision = 0, timer, controller;
let teamPicker = createTeamPicker([]);
const guard = (action) => (...args) => {
  try { return action(...args); } catch (error) { showError(error.message); }
};
const icons = () => window.lucide?.createIcons();
const showError = (message = "") => { $("error").textContent = message; $("error").hidden = !message; };
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
function portrait(hero) {
  const img = el("img");
  img.alt = hero.name;
  img.loading = "lazy";
  img.referrerPolicy = "no-referrer";
  if (hero.image) img.src = hero.image;
  else img.classList.add("image-failed");
  img.addEventListener("error", () => img.classList.add("image-failed"), {once: true});
  return img;
}
function state() {
  return {
    patch_id: Number($("patch").value),
    radiant_team_id: teamPicker.resolve($("radiant-team").value),
    dire_team_id: teamPicker.resolve($("dire-team").value),
    acting_side: document.querySelector('input[name="side"]:checked').value,
    is_pick: document.querySelector('input[name="action"]:checked').value === "pick",
    history: structuredClone(history), top_k: 5,
  };
}
function applyState(next) {
  $("patch").value = next.patch_id;
  $("radiant-team").value = teamPicker.format(next.radiant_team_id);
  $("dire-team").value = teamPicker.format(next.dire_team_id);
  document.querySelector(`input[name="side"][value="${next.acting_side}"]`).checked = true;
  document.querySelector(`input[name="action"][value="${next.is_pick ? "pick" : "ban"}"]`).checked = true;
  history = structuredClone(next.history);
  changed();
}
function remember() { undoStack.push(state()); redoStack = []; }
function addHero(id) {
  if (!metadata || history.length >= metadata.max_actions) return;
  const next = state();
  if (history.some((action) => action.hero_id === id)) return;
  if (next.is_pick && history.filter((a) => a.is_pick && a.side === next.acting_side).length >= 5) {
    showError(`${next.acting_side === "radiant" ? "Radiant" : "Dire"} already has five picks.`);
    return;
  }
  remember();
  history.push({hero_id: id, side: next.acting_side, is_pick: next.is_pick});
  changed();
}
function renderDraft() {
  for (const side of ["radiant", "dire"]) {
    const picks = history.filter((a) => a.side === side && a.is_pick);
    $(side + "-count").textContent = `${picks.length} / 5`;
    $(side + "-picks").replaceChildren(...Array.from({length: 5}, (_, i) => {
      const slot = el("div", "pick-slot");
      if (picks[i]) {
        const hero = heroes.get(picks[i].hero_id);
        slot.append(portrait(hero), el("span", "slot-name", hero.name));
        slot.title = hero.name;
      } else slot.textContent = String(i + 1);
      return slot;
    }));
  }
  $("history-count").textContent = `${history.length} / ${metadata.max_actions}`;
  $("action-number").textContent = history.length === metadata.max_actions ? "Complete" : `${String(history.length + 1).padStart(2, "0")} / ${metadata.max_actions}`;
  $("history").replaceChildren(...history.map((action, index) => {
    const hero = heroes.get(action.hero_id);
    const button = el("button", `history-item ${action.side} ${action.is_pick ? "pick" : "ban"}`);
    button.title = `${index + 1}. ${action.side} ${action.is_pick ? "pick" : "ban"}: ${hero.name}. Rewind before this action`;
    button.setAttribute("aria-label", button.title);
    button.append(portrait(hero), el("span", "", `${index + 1} ${action.is_pick ? "PICK" : "BAN"}`));
    button.onclick = guard(() => { remember(); history = history.slice(0, index); changed(); });
    return button;
  }));
  if (!history.length) $("history").append(el("span", "empty-history", "No actions recorded"));
  $("undo").disabled = !undoStack.length;
  $("redo").disabled = !redoStack.length;
  $("reset").disabled = !history.length;
  renderPool();
}
function renderPool() {
  if (!metadata) return;
  const used = new Set(history.map((action) => action.hero_id));
  const query = $("search").value.trim().toLowerCase();
  const visible = [...heroes.values()].filter((hero) =>
    (hero.name.toLowerCase().includes(query) || String(hero.id) === query) &&
    (!$("available-only").checked || !used.has(hero.id)));
  $("hero-count").textContent = `${heroes.size - used.size} available`;
  $("hero-grid").replaceChildren(...visible.map((hero) => {
    const button = el("button", "hero-button");
    button.title = `${hero.name} / ID ${hero.id}`;
    button.setAttribute("aria-label", `Select ${hero.name}`);
    button.disabled = used.has(hero.id) || history.length >= metadata.max_actions;
    const img = portrait(hero); img.classList.add("hero-image");
    button.append(img, el("span", "", hero.name));
    button.onclick = guard(() => addHero(hero.id));
    return button;
  }));
  if (!visible.length) $("hero-grid").append(el("p", "no-results", "No matching heroes"));
}
function changed() {
  revision += 1;
  controller?.abort();
  clearTimeout(timer);
  showError();
  updateTeamLabels();
  $("predictions").replaceChildren();
  $("predictions").setAttribute("aria-busy", "true");
  $("warnings").replaceChildren();
  $("warning-count").textContent = "";
  renderDraft();
  if (history.length >= metadata.max_actions) {
    $("prediction-status").textContent = "Draft complete";
    $("predictions").setAttribute("aria-busy", "false");
    return;
  }
  $("prediction-status").textContent = "Predicting...";
  timer = setTimeout(predict, 120);
}
async function requestPrediction(input, signal) {
  const response = await fetch("/api/predict", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(input), signal,
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Prediction failed");
  return result;
}
async function predict() {
  const current = revision;
  controller = new AbortController();
  const started = performance.now();
  try {
    const result = await requestPrediction(state(), controller.signal);
    if (current !== revision) return;
    $("predictions").replaceChildren(...result.predictions.map((prediction, index) => {
      const hero = heroes.get(prediction.hero_id);
      const button = el("button", "prediction");
      button.title = `Select ${hero.name}`;
      button.setAttribute("aria-label", `Prediction ${index + 1}: ${hero.name}`);
      const info = el("div", "prediction-info");
      const probability = el("div", "probability");
      const bar = el("progress"); bar.max = 1; bar.value = prediction.probability;
      bar.setAttribute("aria-label", "Model probability");
      probability.append(el("span", "", `${(100 * prediction.probability).toFixed(1)}%`), bar);
      info.append(el("strong", "", hero.name), probability);
      button.append(el("span", "rank", String(index + 1)), portrait(hero), info);
      button.onclick = guard(() => addHero(hero.id));
      return button;
    }));
    $("warnings").replaceChildren(...result.warnings.map((warning) => el("li", "", warning)));
    $("warning-count").textContent = result.warnings.length ? `${result.warnings.length} notes` : "";
    $("prediction-status").textContent = `${result.available_heroes} candidates / ${Math.round(performance.now() - started)} ms`;
  } catch (error) {
    if (error.name === "AbortError" || current !== revision) return;
    showError(error.message);
    $("prediction-status").textContent = "Prediction unavailable";
  } finally {
    if (current === revision) $("predictions").setAttribute("aria-busy", "false");
  }
}
$("search").oninput = renderPool;
$("available-only").onchange = renderPool;
for (const id of ["patch", "radiant-team", "dire-team"]) $(id).oninput = () => { if (metadata) changed(); };
for (const input of document.querySelectorAll('input[type="radio"]')) input.onchange = () => { if (metadata) changed(); };
$("undo").onclick = guard(() => { if (undoStack.length) { redoStack.push(state()); applyState(undoStack.pop()); } });
$("redo").onclick = guard(() => { if (redoStack.length) { undoStack.push(state()); applyState(redoStack.pop()); } });
$("reset").onclick = guard(() => { remember(); history = []; changed(); });
$("retry").onclick = () => metadata ? changed() : initialize();
$("example").onclick = guard(() => {
  remember();
  const ids = metadata.hero_ids.slice(0, 6);
  applyState({patch_id: metadata.training_patch_ids.at(-1), radiant_team_id: null, dire_team_id: null,
    acting_side: "radiant", is_pick: true,
    history: ids.map((hero_id, index) => ({hero_id, side: index % 2 ? "dire" : "radiant", is_pick: index >= 4})),
  });
});
$("export").onclick = guard(() => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(state(), null, 2)], {type: "application/json"}));
  const link = el("a"); link.href = url; link.download = "draft-state.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("import").onclick = () => $("import-file").click();
$("import-file").onchange = async () => {
  const file = $("import-file").files[0];
  if (!file || !metadata) return;
  const current = revision;
  try {
    if (file.size > 16384) throw new Error("Draft JSON must be no larger than 16 KB");
    const input = JSON.parse(await file.text());
    // Validate on the same inference path before allowing imported state into the UI.
    await requestPrediction(input);
    if (revision !== current) throw new Error("Draft changed during import; please import again");
    remember(); applyState(input);
  } catch (error) { showError(error.message); }
  finally { $("import-file").value = ""; }
};
async function initialize() {
  showError(); icons();
  try {
    const response = await fetch("/api/metadata");
    if (!response.ok) throw new Error("Model metadata is unavailable");
    metadata = await response.json();
    teamPicker = createTeamPicker(metadata.teams || []);
    $("team-options").replaceChildren(...teamPicker.options.map((team) => {
      const option = el("option");
      option.value = team.label;
      return option;
    }));
    heroes = new Map(metadata.heroes.sort((a, b) => a.name.localeCompare(b.name)).map((hero) => [hero.id, hero]));
    const config = metadata.model_config;
    $("model-name").textContent = config.lineup_encoder ? "B+ / DeepSets + MLP" : "Draft MLP";
    $("checkpoint").textContent = metadata.checkpoint_sha256.slice(0, 12);
    $("training-patches").textContent = metadata.training_patch_ids.join(", ");
    $("patch").value = metadata.training_patch_ids.at(-1);
    $("example").disabled = false;
    changed();
  } catch (error) {
    showError(`${error.message}. Check that the local Python server is running.`);
    $("prediction-status").textContent = "Model unavailable";
    $("model-name").textContent = "Disconnected";
    $("hero-grid").replaceChildren();
    $("predictions").setAttribute("aria-busy", "false");
  }
}
function updateTeamLabels() {
  for (const side of ["radiant", "dire"]) {
    const input = $(side + "-team");
    const status = $(side + "-team-status");
    try {
      const id = teamPicker.resolve(input.value);
      const name = teamPicker.byId.get(id)?.name;
      const known = metadata.team_ids.includes(id);
      status.textContent = id == null ? "Unknown team" : `${name || "Unnamed team"} / ID ${id}${known ? "" : " / unseen in training"}`;
      input.setAttribute("aria-invalid", "false");
      input.setCustomValidity("");
    } catch (error) {
      status.textContent = "Select a team or enter its ID";
      input.setAttribute("aria-invalid", "true");
      input.setCustomValidity(error.message);
    }
  }
}
initialize();
