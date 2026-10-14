const {test} = require('node:test');
const assert = require('node:assert/strict');
const {createTeamPicker} = require('../src/web/team-picker.js');
const picker = createTeamPicker([
  {id: 10, name: 'Team One', aliases: ['Old Team']},
  {id: 20, name: 'Same Name', aliases: []},
  {id: 30, name: 'Same Name', aliases: []},
]);

test('resolve labels, names, aliases, IDs and unknown teams', () => {
  assert.equal(picker.resolve('Team One (#10)'), 10);
  assert.equal(picker.resolve(' team one '), 10);
  assert.equal(picker.resolve('Old Team'), 10);
  assert.equal(picker.resolve('Old Team (#10)'), 10);
  assert.equal(picker.resolve('999'), 999);
  assert.equal(picker.resolve(''), null);
  assert.equal(picker.format(10), 'Team One (#10)');
  assert.equal(picker.format(999), '999');
  assert.equal(picker.format(null), '');
  assert.equal(picker.resolve(picker.format(10)), 10);
});

test('ambiguous and invalid text cannot silently become an unknown team', () => {
  assert.throws(() => picker.resolve('Same Name'), /Multiple teams/);
  assert.equal(picker.resolve('Same Name (#20)'), 20);
  assert.equal(picker.resolve('Same Name (#30)'), 30);
  for (const value of ['Typo', 'Wrong Name (#10)', '-1', '0', '1.5', '99999999999999999999']) {
    assert.throws(() => picker.resolve(value));
  }
});

test('fallback catalog preserves numeric input', () => {
  const empty = createTeamPicker([]);
  assert.equal(empty.resolve('10'), 10);
  assert.equal(empty.format(10), '10');
  assert.deepEqual(empty.options, []);
});
