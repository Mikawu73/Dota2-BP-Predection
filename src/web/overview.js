"use strict";

const $ = (id) => document.getElementById(id);
const number = (value) => new Intl.NumberFormat("en-US").format(value);
const percent = (value) => `${(value * 100).toFixed(2)}%`;
const date = (value) => new Date(typeof value === "number" ? value * 1000 : value).toISOString().slice(0, 10);
const icons = () => window.lucide?.createIcons();
const node = (tag, className, text) => {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
};
let report;

function renderDataset() {
  const dataset = report.dataset;
  const reference = report.runs.find((run) => run.run === "decay_current_60");
  $("snapshot").textContent = `${date(dataset.earliest_match_utc)} to ${date(dataset.latest_match_utc)} / UTC`;
  $("stat-matches").textContent = number(dataset.eligible_matches);
  $("stat-filtered").textContent = `${number(dataset.excluded_matches)} excluded after quality checks`;
  $("stat-actions").textContent = number(dataset.samples);
  $("stat-top5").textContent = percent(reference.best_validation_metrics.top5_accuracy);
  $("stat-runs").textContent = number(report.runs.length);
  const behaviors = DraftReport.behaviorRuns(report).length;
  $("stat-run-types").textContent = `${behaviors} behavior / ${report.runs.length - behaviors} outcome`;
  $("normalized-count").textContent = number(dataset.raw_matches);
  $("eligible-count").textContent = number(dataset.eligible_matches);
  $("excluded-count").textContent = `${(100 * dataset.excluded_matches / dataset.raw_matches).toFixed(1)}% excluded`;
  $("training-patches").textContent = reference.training_patches.join(", ");
  const chart = $("split-chart");
  chart.replaceChildren();
  let offset = 0;
  const splitRows = ["train", "validation", "test"].map((split) => {
    const data = dataset.match_splits[split];
    const width = 600 * data.matches / dataset.eligible_matches;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    for (const [key, value] of Object.entries({x: offset, y: 0, width, height: 36, class: split})) rect.setAttribute(key, value);
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${split}: ${number(data.matches)} matches`;
    rect.append(title); chart.append(rect); offset += width;
    const row = node("tr");
    const label = node("td");
    label.append(node("span", `dot ${split}`), node("span", "", split[0].toUpperCase() + split.slice(1)),
      node("small", "", `${date(data.earliest_start_time)} to ${date(data.latest_start_time)}`));
    row.append(label, node("td", "", number(data.matches)), node("td", "", number(dataset.sample_splits[split])));
    return row;
  });
  $("split-rows").replaceChildren(...splitRows);
}

function renderDecay() {
  const metric = document.querySelector('input[name="chart-metric"]:checked').value;
  const display = metric === "mrr" ? (value) => value.toFixed(4) : percent;
  const rows = ["none", "60", "120", "240"].map((suffix) => report.runs.find((run) => run.run === `decay_current_${suffix}`)).filter(Boolean);
  const maximum = Math.ceil(Math.max(metric === "mrr" ? 0.3 : 0.4, ...rows.map((run) => run.best_validation_metrics[metric])) * 10) / 10;
  $("chart-scale-max").textContent = metric === "mrr" ? maximum.toFixed(2) : `${Math.round(maximum * 100)}%`;
  $("decay-bars").replaceChildren(...rows.map((run) => {
    const selected = run.run === "decay_current_60";
    const row = node("div", `decay-row${selected ? " selected" : ""}`);
    const label = node("label", "", run.recency_half_life_days ? `${run.recency_half_life_days} days` : "No decay");
    const bar = node("meter");
    bar.id = `bar-${run.run}`; label.htmlFor = bar.id;
    const value = run.best_validation_metrics[metric];
    bar.min = 0; bar.max = maximum; bar.value = value;
    bar.setAttribute("aria-label", `${label.textContent}: ${display(value)}`);
    row.append(label, bar, node("strong", "", display(value)));
    return row;
  }));
}

function renderExperiments() {
  const runs = DraftReport.behaviorRuns(report, $("experiment-filter").value, $("experiment-query").value, $("experiment-sort").value);
  $("experiment-count").textContent = `${runs.length} runs / validation`;
  $("experiment-rows").replaceChildren(...runs.map((run) => {
    const row = node("tr", run.run === "decay_current_60" ? "reference-run" : "");
    const label = node("td");
    label.append(node("span", "run-label", DraftReport.name(run)), node("span", "run-id", run.run));
    const metrics = run.best_validation_metrics;
    row.append(label, node("td", "", run.seed), node("td", "", run.best_epoch),
      node("td", "", Number.isFinite(metrics.top1_accuracy) ? percent(metrics.top1_accuracy) : "-"),
      node("td", "top5-cell", percent(metrics.top5_accuracy)), node("td", "", metrics.mrr.toFixed(4)));
    const cell = node("td");
    const button = node("button", "icon-button");
    button.title = `Details: ${DraftReport.name(run)}, seed ${run.seed}`;
    button.setAttribute("aria-label", button.title);
    const icon = node("i"); icon.dataset.lucide = "sliders-horizontal";
    button.append(icon); button.onclick = () => showRun(run); cell.append(button); row.append(cell);
    return row;
  }));
  if (!runs.length) {
    const row = node("tr"), cell = node("td", "no-experiments", "No experiments match these filters.");
    cell.colSpan = 7; row.append(cell); $("experiment-rows").append(row);
  }
  icons();
}

function renderSeeds() {
  const comparison = DraftReport.seedComparison(report);
  $("seed-rows").replaceChildren(...comparison.pairs.map((pair) => {
    const row = node("tr");
    row.append(node("td", "", pair.seed), node("td", "", percent(pair.baseline.best_validation_metrics.top5_accuracy)),
      node("td", "", percent(pair.sequence.best_validation_metrics.top5_accuracy)));
    return row;
  }));
  $("seed-means").replaceChildren();
  if (comparison.pairs.length) {
    const row = node("tr");
    row.append(node("td", "", `Mean (${comparison.pairs.length} seeds)`),
      node("td", "", percent(comparison.baselineMean)), node("td", "", percent(comparison.sequenceMean)));
    $("seed-means").append(row);
  }
}

function renderValue() {
  const value = report.runs.find((run) => run.run === "draft_value");
  if (!value) {
    $("value-metrics").replaceChildren(node("span", "", "No recorded value-model run"));
    $("value-epoch").textContent = "";
    return;
  }
  $("value-metrics").replaceChildren(...[["AUC", "auc"], ["Log loss", "log_loss"], ["Brier score", "brier_score"]].map(([label, key]) => {
    const element = node("div");
    element.append(node("span", "", label), node("strong", "", value.best_validation_metrics[key].toFixed(4)));
    return element;
  }));
  $("value-epoch").textContent = `Selected epoch ${value.best_epoch} / ${number(value.validation_samples)} factual validation picks`;
}

function showRun(run) {
  $("run-dialog-title").textContent = DraftReport.name(run);
  const content = $("run-dialog-content");
  content.replaceChildren(node("p", "", run.model), node("code", "", run.run));
  const metrics = node("dl");
  for (const [key, value] of Object.entries(run.best_validation_metrics)) {
    metrics.append(node("dt", "", key), node("dd", "", value.toFixed(6)));
  }
  content.append(node("h3", "", "Validation metrics"), metrics, node("h3", "", "Recorded configuration"));
  const config = node("dl");
  const excluded = new Set(["model", "run", "source", "best_validation_metrics", "validation_constant_baseline"]);
  for (const [key, value] of Object.entries(run)) {
    if (!excluded.has(key)) config.append(node("dt", "", key.replaceAll("_", " ")), node("dd", "", value === null ? "None" : String(value)));
  }
  content.append(config, node("h3", "", "Source summary"), node("p", "", run.source.path), node("code", "", `SHA-256 ${run.source.sha256}`));
  $("run-dialog").showModal();
}

function renderProvenance() {
  $("comparison-note").textContent = report.comparison_note;
  $("provenance-note").textContent = report.provenance_note;
  $("report-sources").replaceChildren(...report.dataset.sources.map((source) => {
    const item = node("li", "", source.path);
    item.append(node("code", "", `SHA-256 ${source.sha256}`));
    return item;
  }));
}

async function loadReport() {
  $("reload-report").disabled = true;
  $("report-error").hidden = true;
  $("loading-report").hidden = false;
  $("report-content").hidden = true;
  try {
    const response = await fetch("/api/report", {signal: AbortSignal.timeout(15000)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Report unavailable");
    report = DraftReport.validate(data);
    renderDataset(); renderDecay(); renderExperiments(); renderSeeds(); renderValue(); renderProvenance();
    $("report-content").hidden = false;
  } catch (error) {
    $("snapshot").textContent = "Report unavailable";
    $("report-error-message").textContent = error.message;
    $("report-error").hidden = false;
  } finally {
    $("loading-report").hidden = true;
    $("reload-report").disabled = false;
    icons();
  }
}

for (const id of ["experiment-filter", "experiment-sort"]) $(id).onchange = () => { if (report) renderExperiments(); };
$("experiment-query").oninput = () => { if (report) renderExperiments(); };
for (const input of document.querySelectorAll('input[name="chart-metric"]')) input.onchange = () => { if (report) renderDecay(); };
$("close-run").onclick = () => $("run-dialog").close();
$("run-dialog").onclick = (event) => { if (event.target === $("run-dialog")) $("run-dialog").close(); };
$("reload-report").onclick = loadReport;
$("download-report").onclick = () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: "application/json"}));
  const link = node("a"); link.href = url; link.download = "dota2-validation-results.json";
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
icons();
loadReport();
