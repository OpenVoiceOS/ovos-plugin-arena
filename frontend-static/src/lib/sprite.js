// Procedural fighter sprites — deterministic pixel creatures from an id hash.
// Shared by the bestiary (real fighters) and the battle page (masked fighters).

export const TYPE_HUE = {
  'gofai': 45, 'fuzzy-match': 25, 'neural-net': 212, 'template-match': 135,
  'keyword-match': 185, 'embedding': 270, 'llm': 355, 'ensemble': 310,
  'transformer': 212, 'statistical': 80, 'classical-ml': 160, 'cloud': 200,
};

export function hashString(str) {
  let h = 2166136261 >>> 0; // FNV-1a
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

function* bitStream(seed) {
  let state = seed >>> 0;
  while (true) {
    state ^= state << 13; state >>>= 0;
    state ^= state >> 17;
    state ^= state << 5; state >>>= 0;
    yield state;
  }
}

function escAttr(str) {
  return String(str).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

/**
 * Render one pixel-creature SVG.
 *
 * @param seedStr  stable identity string (competitor_id, or battle_id+side)
 * @param primaryType  first architecture type — drives the colour family
 * @param size  rendered px size
 * @param masked  grey "mystery fighter" with a ? badge (blind battles)
 */
export function spriteSvg(seedStr, primaryType = '', size = 72, masked = false) {
  const seed = hashString(seedStr);
  const bits = bitStream(seed);
  const next = () => bits.next().value;

  const slug = (primaryType || '').toLowerCase().replace(/[^a-z0-9]+/g, '-');
  const hue = masked ? 0 : (TYPE_HUE[slug] ?? (seed % 360));
  const hueJitter = (next() % 21) - 10;
  const bodyH = (hue + hueJitter + 360) % 360;
  const sat = masked ? 0 : 55 + next() % 25;
  const body = `hsl(${bodyH} ${sat}% ${masked ? 30 : 48 + next() % 14}%)`;
  const shade = `hsl(${bodyH} ${masked ? 0 : 60}% ${masked ? 18 : 30}%)`;
  const bg = `hsl(${bodyH} ${masked ? 0 : 35}% ${masked ? 9 : 14}%)`;

  // 5×5 mirrored body mask, ~60% fill
  const grid = [];
  for (let y = 0; y < 5; y++) {
    const row = [];
    for (let x = 0; x < 3; x++) row.push(next() % 100 < 62);
    grid.push([row[0], row[1], row[2], row[1], row[0]]);
  }
  grid[1][2] = true;
  grid[2][1] = grid[2][2] = grid[2][3] = true;

  const eyeRow = 1 + (next() % 2);
  const eyeCol = next() % 2 ? 1 : 0;
  grid[eyeRow][1 + eyeCol] = true;
  grid[eyeRow][3 - eyeCol] = true;

  const cell = 10, pad = 6;
  let pixels = '';
  for (let y = 0; y < 5; y++) {
    for (let x = 0; x < 5; x++) {
      if (!grid[y][x]) continue;
      const edge = y === 4 || !grid[y + 1]?.[x];
      pixels += `<rect x="${pad + x * cell}" y="${pad + y * cell}" width="${cell}" height="${cell}" rx="2.5" fill="${edge ? shade : body}"/>`;
    }
  }
  for (const ex of [1 + eyeCol, 3 - eyeCol]) {
    const cx = pad + ex * cell + cell / 2;
    const cy = pad + eyeRow * cell + cell / 2;
    if (masked) {
      pixels += `<circle cx="${cx}" cy="${cy}" r="3.2" fill="#2a2a2e"/>`
              + `<circle cx="${cx}" cy="${cy}" r="1.5" fill="#c9c9d1"/>`;
    } else {
      pixels += `<circle cx="${cx}" cy="${cy}" r="3.2" fill="#f4f7fa"/>`
              + `<circle cx="${cx + (ex < 2 ? 0.9 : -0.9)}" cy="${cy + 0.6}" r="1.6" fill="#10151c"/>`;
    }
  }
  const badge = masked
    ? `<text x="50" y="16" font-size="13" font-weight="800" fill="#ff4d4f" font-family="monospace">?</text>`
    : '';
  return `<svg width="${size}" height="${size}" viewBox="0 0 62 62" role="img"
               aria-label="${masked ? 'Masked fighter' : `Generated sprite for ${escAttr(seedStr)}`}">
    <rect width="62" height="62" rx="12" fill="${bg}"/>${pixels}${badge}</svg>`;
}
