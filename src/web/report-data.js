(function (root) {
  "use strict";
  const baselineIds = ["decay_current_60", "model_b_plus_recency60_seed7", "model_b_plus_recency60_seed2026"];
  const sequenceIds = ["model_b_plus_recency60_sequence", "model_b_plus_recency60_sequence_seed7", "model_b_plus_recency60_sequence_seed2026"];
  function group(run) {
    if (run.run === "draft_value") return "value";
    if (run.outcome_weighted_pick_loss) return "weighted";
    if (run.outcome_auxiliary || run.role_auxiliary) return "auxiliary";
    if (run.sequence_encoder) return "sequence";
    if (run.run.startsWith("decay_current_")) return "recency";
    return "baseline";
  }
  function name(run) {
    switch (group(run)) {
      case "value": return "Independent value model";
      case "weighted": return "B+ / outcome-weighted picks";
      case "auxiliary": return run.role_auxiliary ? "B+ + sequence + outcome + role" : "B+ + sequence + outcome";
      case "sequence": return "B+ + sequence Transformer";
      default: return `B+ / ${run.recency_half_life_days ? `${run.recency_half_life_days}-day decay` : "no decay"}`;
    }
  }
  function validate(report) {
    if (report?.schema_version !== 1 || report.evaluation_split !== "validation" || !Array.isArray(report.runs)) {
      throw new Error("Unsupported results report");
    }
    const dataset = report.dataset;
    const nonnegative = (value) => Number.isSafeInteger(value) && value >= 0;
    if (!dataset || ![dataset.raw_matches, dataset.eligible_matches, dataset.excluded_matches, dataset.samples].every(nonnegative)) {
      throw new Error("Invalid dataset counts");
    }
    if (dataset.raw_matches !== dataset.eligible_matches + dataset.excluded_matches || !dataset.raw_matches) {
      throw new Error("Dataset count mismatch");
    }
    const splits = ["train", "validation", "test"];
    if (!splits.every((split) => nonnegative(dataset.match_splits?.[split]?.matches) && nonnegative(dataset.sample_splits?.[split]))) {
      throw new Error("Missing split counts");
    }
    if (splits.reduce((n, s) => n + dataset.match_splits[s].matches, 0) !== dataset.eligible_matches ||
        splits.reduce((n, s) => n + dataset.sample_splits[s], 0) !== dataset.samples) {
      throw new Error("Split totals do not match the dataset");
    }
    if (new Set(report.runs.map((run) => run.run)).size !== report.runs.length) throw new Error("Duplicate run IDs");
    for (const run of report.runs) {
      const metrics = run.best_validation_metrics;
      if (!metrics || !Object.values(metrics).every((value) => typeof value === "number" && Number.isFinite(value))) {
        throw new Error("Invalid experiment metrics");
      }
    }
    if (!report.runs.some((run) => run.run === "decay_current_60")) throw new Error("B+ reference run is missing");
    return report;
  }
  function behaviorRuns(report, filter = "all", query = "", sort = "top5_accuracy") {
    const text = query.trim().toLowerCase();
    return report.runs.filter((run) =>
      Number.isFinite(run.best_validation_metrics?.top5_accuracy) &&
      (filter === "all" || group(run) === filter) &&
      `${name(run)} ${run.run} ${run.seed}`.toLowerCase().includes(text)
    ).sort((a, b) => sort === "name" ? name(a).localeCompare(name(b)) || a.seed - b.seed :
      b.best_validation_metrics[sort] - a.best_validation_metrics[sort] || a.run.localeCompare(b.run));
  }
  function seedComparison(report) {
    const find = (ids) => ids.map((id) => report.runs.find((run) => run.run === id)).filter(Boolean);
    const baseline = find(baselineIds), sequence = find(sequenceIds);
    const pairs = baseline.map((run) => ({seed: run.seed, baseline: run,
      sequence: sequence.find((other) => other.seed === run.seed)})).filter((pair) => pair.sequence);
    const mean = (key) => pairs.length ? pairs.reduce((sum, pair) => sum + pair[key].best_validation_metrics.top5_accuracy, 0) / pairs.length : null;
    return {pairs, baselineMean: mean("baseline"), sequenceMean: mean("sequence")};
  }
  const api = {group, name, validate, behaviorRuns, seedComparison};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DraftReport = api;
})(globalThis);
