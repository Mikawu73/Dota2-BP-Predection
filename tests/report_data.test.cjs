const {test} = require('node:test');
const assert = require('node:assert/strict');
const report = require('../reports/results.json');
const data = require('../src/web/report-data.js');

test('public snapshot has internally consistent counts and validation metrics', () => {
  assert.equal(data.validate(report), report);
  assert.equal(report.dataset.eligible_matches, 25158);
  assert.equal(report.dataset.samples, 603640);
  assert.equal(report.runs.length, 13);
  assert.equal(data.behaviorRuns(report).length, 12);
  assert.equal(data.behaviorRuns(report)[0].run, 'decay_current_60');
  assert.equal(data.behaviorRuns(report)[0].best_validation_metrics.top5_accuracy.toFixed(6), '0.315928');
});

test('experiment filters and search exclude the different outcome prediction task', () => {
  assert.equal(data.behaviorRuns(report, 'recency').length, 4);
  assert.equal(data.behaviorRuns(report, 'sequence').length, 3);
  assert.equal(data.behaviorRuns(report, 'auxiliary').length, 2);
  assert.equal(data.behaviorRuns(report, 'weighted').length, 1);
  assert.equal(data.behaviorRuns(report, 'all', 'draft_value').length, 0);
  assert.equal(data.behaviorRuns(report, 'all', 'not-a-model').length, 0);
  assert.equal(data.behaviorRuns(report, 'all', 'SEED2026').length, 2);
  const sorted = data.behaviorRuns(report, 'all', '', 'mrr');
  for (let i = 1; i < sorted.length; i++) {
    assert.ok(sorted[i - 1].best_validation_metrics.mrr >= sorted[i].best_validation_metrics.mrr);
  }
  assert.match(data.name(report.runs.find(r => r.run === 'model_b_plus_recency60_sequence_role_multitask')), /outcome \+ role/);
});

test('seed means use paired repeat seeds only, not unrelated experiments', () => {
  const summary = data.seedComparison(report);
  assert.deepEqual(summary.pairs.map(p => p.seed), [42, 7, 2026]);
  assert.ok(Math.abs(summary.baselineMean - 0.31239071294731763) < 1e-10);
  assert.ok(Math.abs(summary.sequenceMean - 0.31259318159228705) < 1e-10);
  const subset = {...report, runs: report.runs.filter(r => r.run !== 'model_b_plus_recency60_sequence_seed7')};
  assert.equal(data.seedComparison(subset).pairs.length, 2);
});

test('inconsistent or non-validation reports are not silently rendered', () => {
  for (const mutate of [
    r => { r.evaluation_split = 'test'; },
    r => { r.dataset.samples += 1; },
    r => { r.dataset.excluded_matches += 1; },
    r => { r.runs.push(r.runs[0]); },
    r => { r.runs[0].best_validation_metrics.mrr = NaN; },
  ]) {
    const copy = structuredClone(report);
    mutate(copy);
    assert.throws(() => data.validate(copy));
  }
});
