// Safe localStorage wrapper. Private browsing, storage-partitioned iframes
// and some locked-down browsers throw on *any* localStorage access, not
// only when it's full — a bare call on a load-time path (e.g. reading a
// saved filter before the first render) takes down the whole page with an
// uncaught exception instead of just failing to persist. Every read/write
// in the battle and free-vote pages goes through here and falls back to an
// in-memory Map for the lifetime of the page once storage is known to be
// unavailable, so the vote flow still completes — it just won't remember
// anything across a reload.
const memory = new Map();
let blocked = false;

export function storageGet(key) {
  if (!blocked) {
    try { return localStorage.getItem(key); } catch { blocked = true; }
  }
  return memory.has(key) ? memory.get(key) : null;
}

export function storageSet(key, value) {
  if (!blocked) {
    try { localStorage.setItem(key, value); return; } catch { blocked = true; }
  }
  memory.set(key, value);
}
