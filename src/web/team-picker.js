/* Display names never replace numeric model IDs. Also usable in Node tests. */
(function (root) {
  "use strict";
  function createTeamPicker(teams) {
    const byId = new Map(teams.map((team) => [team.id, team]));
    const labels = new Map();
    const names = new Map();
    const options = [];
    for (const team of teams) {
      for (const name of [team.name, ...(team.aliases || [])]) {
        const key = name.trim().toLowerCase();
        if (!names.has(key)) names.set(key, new Set());
        names.get(key).add(team.id);
        const label = `${name} (#${team.id})`;
        labels.set(label.toLowerCase(), team.id);
        options.push({label, id: team.id});
      }
    }
    options.sort((a, b) => a.label.localeCompare(b.label));
    function resolve(value) {
      const text = value.trim();
      if (!text) return null;
      if (/^[0-9]+$/.test(text)) {
        const id = Number(text);
        if (Number.isSafeInteger(id) && id > 0) return id;
        throw new Error("Team ID must be a positive safe integer");
      }
      const key = text.toLowerCase();
      if (labels.has(key)) return labels.get(key);
      const matches = names.get(key);
      if (matches?.size === 1) return [...matches][0];
      if (matches?.size > 1) throw new Error("Multiple teams share this name; select an entry with its ID");
      throw new Error("Team name not found; select a matching entry or enter a numeric ID");
    }
    function format(id) {
      if (id == null) return "";
      const team = byId.get(id);
      return team ? `${team.name} (#${id})` : String(id);
    }
    return {resolve, format, options, byId};
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {createTeamPicker};
  else root.createTeamPicker = createTeamPicker;
})(globalThis);
