// A small, fixed arena served in place of public/data/ by ui-check.mjs, so
// the behaviour checks do not move when the hourly data refresh does. It
// carries just enough to exercise every case the checks name: two wake
// words with clip and streaming boards, a dataset whose registry folder
// name is not its phrase, a pooled intent ladder with one cast human vote,
// a three-battle STT pool, and boards with performance numbers on two
// hardware tiers.

const at = '2026-01-01T00:00:00+00:00';

const fighter = (competitor_id, modality, extra = {}) => ({
  competitor_id, modality, display_name: competitor_id, plugin: `plugin-${competitor_id}`,
  config: {}, langs: ['en-US'], types: ['neural-net'], has_predictions: true, ...extra,
});
const ww = (id, phrase) => fighter(id, 'wake_word', { config: { hotwords: { [phrase]: {} } } });

const competitors = [
  ww('myc-precise', 'hey_mycroft'),
  ww('myc-oww', 'hey_mycroft'),
  ww('jar-oww', 'hey_jarvis'),
  ww('jar-vosk', 'jarvis'),
  ww('sam-vosk', 'sam'),
  fighter('int-a', 'intent_online'),
  fighter('int-b', 'intent_online', { types: ['keyword-match'] }),
  fighter('stt-a', 'stt', { types: ['transformer'], size: 'small' }),
  fighter('stt-b', 'stt', { types: ['ctc'], langs: ['ca-ES'], size: 'base' }),
];

const entry = (rank, competitor_id, error_rate, extra = {}) => ({
  rank, competitor_id, plugin_id: `plugin-${competitor_id}`, samples: 10,
  metrics: { error_rate }, unranked: false, ...extra,
});
const wwEntries = () => [
  entry(1, 'myc-precise', 0.1), entry(2, 'myc-oww', 0.2), entry(3, 'jar-oww', 0.9),
  entry(4, 'jar-vosk', 0.95), entry(5, 'sam-vosk', 1.0),
];
const board = (modality, dataset_id, display_name, entries, info = {}) => ({
  modality, dataset_id, lang: 'en-US', generated_at: at, primary_metric: 'error_rate',
  dataset_info: { display_name, ...info }, entries,
});

const ladderEntry = (competitor_id, bt_rating, battles = 4) => ({
  competitor_id, plugin_id: `plugin-${competitor_id}`, bt_rating, elo: bt_rating,
  battles, wins: 2, losses: 2, ties: 0,
});
const ladder = (modality, entries, human = 0) => ({
  modality, lang: 'en-US', generated_at: at, vote_count: 20, human_vote_count: human,
  provisional: false, entries, metric_ladders: {},
});

const battle = (battle_id, modality, dataset_id, a, b, extra = {}) => ({
  battle_id, modality, dataset_id, lang: 'en-US', sample_id: battle_id, input_text: null,
  audio_url: null, reference: null, prediction_a: null, prediction_b: null,
  competitor_a: a, competitor_b: b, plugin_a: `plugin-${a}`, plugin_b: `plugin-${b}`, ...extra,
});

const perf = (rtf, peak_rss_mb, tier = 'gpu') => ({ [tier]: { rtf, rtf_n: 10, peak_rss_mb, peak_rss_mb_n: 10 } });

const files = {
  'benchmark-wake_word-synthetic-hey_mycroft-en-US.json': board('wake_word', 'synthetic-hey_mycroft',
    'Synthetic wake word — “hey mycroft”', wwEntries(), { wakeword: 'hey_mycroft' }),
  'benchmark-wake_word-picovoice-jarvis-en-US.json': board('wake_word', 'picovoice-jarvis',
    'Picovoice benchmark — “jarvis”', wwEntries()),
  'benchmark-wake_word-sam-wake-word-en-US.json': board('wake_word', 'sam-wake-word',
    'Sam wake word', wwEntries(), { wakeword: 'positive' }),
  'benchmark-ww_stream-ww_stream_hey_mycroft-en-US.json': board('ww_stream', 'ww_stream_hey_mycroft',
    'Streaming wake word — “hey mycroft”', wwEntries()),
  'benchmark-stt-ds-a-en-US.json': board('stt', 'ds-a', 'Dataset A', [
    entry(1, 'stt-a', 0.1, { perf: perf(0.2, 900) }), entry(2, 'stt-b', 0.3, { perf: perf(0.05, 300, 'cpu-x86') })]),
  'benchmark-stt-ds-b-en-US.json': board('stt', 'ds-b', 'Dataset B', [
    entry(1, 'stt-b', 0.2, { perf: perf(0.06, 310) }), entry(2, 'stt-a', 0.25, { perf: perf(0.21, 910) })]),
  'benchmark-intent_online-ix-en-US.json': board('intent_online', 'ix', 'Intent set', [
    entry(1, 'int-a', 0.1), entry(2, 'int-b', 0.2)]),
  'leaderboard-wake_word-en-US.json': ladder('wake_word', [
    ladderEntry('myc-precise', 1300), ladderEntry('jar-oww', 1250), ladderEntry('myc-oww', 1200),
    ladderEntry('jar-vosk', 1100), ladderEntry('sam-vosk', 1000)]),
  'leaderboard-intent-en-US.json': ladder('intent', [ladderEntry('int-a', 1300), ladderEntry('int-b', 1100)], 1),
  'leaderboard-stt-en-US.json': ladder('stt', [ladderEntry('stt-a', 1300), ladderEntry('stt-b', 1100)]),
  'battles-stt-tiny-en-US.json': {
    modality: 'stt', dataset_id: 'tiny', lang: 'en-US', generated_at: at, dataset_info: { display_name: 'Tiny' },
    battles: ['s1', 's2', 's3'].map(id => battle(id, 'stt', 'tiny', 'stt-a', 'stt-b',
      { reference: 'hello there', prediction_a: 'hello there', prediction_b: 'hello bear' })),
  },
  'battles-wake_word-synthetic-hey_mycroft-en-US.json': {
    modality: 'wake_word', dataset_id: 'synthetic-hey_mycroft', lang: 'en-US', generated_at: at,
    dataset_info: { display_name: 'Synthetic wake word — “hey mycroft”' },
    battles: [
      battle('w-cross', 'wake_word', 'synthetic-hey_mycroft', 'myc-precise', 'jar-oww',
        { reference: 'positive', prediction_a: 'detected', prediction_b: 'not_detected' }),
      battle('w-same', 'wake_word', 'synthetic-hey_mycroft', 'myc-precise', 'myc-oww',
        { reference: 'positive', prediction_a: 'detected', prediction_b: 'not_detected' }),
    ],
  },
  'battles-intent-freeform-en-US.json': {
    modality: 'intent', dataset_id: 'freeform', lang: 'en-US', generated_at: at,
    battles: [battle('ff1', 'intent', 'freeform', 'int-a', 'int-b')],
  },
  'votes.jsonl': JSON.stringify({
    battle_id: 'ff1', choice: 'a', competitor_a: 'int-a', competitor_b: 'int-b',
    modality: 'intent', lang: 'en-US', issue: 1,
  }) + '\n',
  'vote-audit.json': { generated_at: at, counted: 1, discarded: [], downweighted: [] },
  'metrics-legend.json': { metrics: {}, leagues: {} },
  'competitors.json': { generated_at: at, competitors, plugin_families: {} },
};

const meta = (file, extra = {}) => {
  const body = files[file];
  return { file, modality: body.modality, dataset_id: body.dataset_id ?? null, lang: body.lang,
    generated_at: at, count: (body.entries || body.battles || []).length, ...extra };
};
const names = Object.keys(files);
files['index.json'] = {
  generated_at: at,
  auto_vote_weight: 0.25,
  leagues: [
    { id: 'intent_online', label: 'Intent · Trains at boot', battle_group: 'intent', order: 0, voteless: false },
    { id: 'stt', label: 'STT', battle_group: 'stt', order: 3, voteless: false },
    { id: 'wake_word', label: 'Wake Word', battle_group: 'wake_word', order: 5, voteless: false },
    { id: 'ww_stream', label: 'Wake Word · Streaming', battle_group: 'ww_stream', order: 8, voteless: false },
  ],
  leaderboards: names.filter(n => n.startsWith('leaderboard-')).map(n => meta(n,
    { human_vote_count: files[n].human_vote_count, vote_count: files[n].vote_count })),
  benchmarks: names.filter(n => n.startsWith('benchmark-')).map(n => meta(n)),
  battles_pools: names.filter(n => n.startsWith('battles-') && !n.includes('-freeform-')).map(n => meta(n)),
  freeform_pools: names.filter(n => n.includes('-freeform-')).map(n => meta(n)),
};

export default files;
