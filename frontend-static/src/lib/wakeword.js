// One league per wake word. A detector is built for one phrase, so a
// "hey jarvis" detector scored on a "hey mycroft" corpus, or ranked beside a
// "hey mycroft" detector, says nothing useful. Everything here derives the
// phrase from registry data — the fighter's hotword config key and the
// dataset's registry fields — so a new wake word needs no code change.

export const WAKE_LEAGUES = new Set(['wake_word', 'ww_stream']);

// "hey_mycroft", "Hey-Mycroft" and "hey mycroft" are one phrase.
export const normPhrase = s => String(s || '').toLowerCase().replace(/[\s_\-]+/g, ' ').trim();

export const phraseLabel = p => `“${p}”`;

// The phrase a wake-word fighter listens for: its hotword config key.
export function fighterPhrase(fighter) {
  const key = Object.keys(fighter?.config?.hotwords || {})[0];
  return key ? normPhrase(key) : '';
}

const quoted = s => (String(s || '').match(/“([^”]+)”|"([^"]+)"/) || []).slice(1).find(Boolean) || '';
const containsWords = (text, phrase) => ` ${text} `.includes(` ${phrase} `);

// The phrase a dataset board tests, resolved against the phrases fighters
// actually listen for. The registry's `wakeword` field is the folder that
// holds the positive clips, which is usually the phrase but not always
// (one corpus names it "positive"), so it only wins when a fighter listens
// for it. After that: the phrase quoted in the dataset's display name, then
// the longest fighter phrase spelled out in the dataset id or name.
export function datasetPhrase(board, knownPhrases) {
  const info = board?.dataset_info || {};
  const known = [...knownPhrases].sort((a, b) => b.length - a.length);
  const direct = [normPhrase(info.wakeword), normPhrase(quoted(info.display_name))].filter(Boolean);
  const hit = direct.find(p => knownPhrases.has(p));
  if (hit) return hit;
  for (const text of [normPhrase(board?.dataset_id), normPhrase(info.display_name)]) {
    const found = known.find(p => containsWords(text, p));
    if (found) return found;
  }
  return direct[0] || normPhrase(board?.dataset_id);
}

// URL-safe key for a phrase ("hey mycroft" -> "hey_mycroft").
export const phraseKey = p => normPhrase(p).replace(/ /g, '_');
