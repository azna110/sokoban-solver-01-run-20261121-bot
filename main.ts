// Sokoban solver: push-optimal, room by room.
//
// Idea: every generated level is a hallway with small rooms off it, and a box
// pushed out of a room can never come back, so rooms are independent: the
// level optimum is the sum of the room optima. Each room (3-4 boxes in a
// 6x6 slot) is solved on its own with A* over pushes, the other rooms'
// boxes standing as walls, and the room solutions are stitched together
// with the player's walks between them. That decomposition is what makes
// the 16-box levels tractable: a joint search would meet the product of
// the rooms' state spaces.
//
// The per-room search: a state is the sorted box squares plus the player's
// reachable region (named by its lowest square). Cost is pushes, and the
// heuristic is the minimum-cost matching of boxes to goals under
// shortest-path distances on open floor (each push moves one box one
// square, so the matching is an admissible lower bound on pushes left).
// Pruning: a box on a dead square (no push sequence, ignoring other
// boxes, can ever bring it to a goal) and a frozen 2x2 block of walls
// and boxes with a box off its goal. Both prune only dead states, so
// the first solution A* pops is push-optimal. Work-bounded only:
// expansion caps, never the clock, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MAX_ROOM_STATES = 250_000;
const MAX_WHOLE_STATES = 250_000;

const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
] as const;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares a lone box can be pushed from onto some goal (pull-back from goals). */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y - d < 0 || y - d >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Flood the player's reachable region from `from`; returns its lowest square. */
function reachLow(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** True when the box pushed to `t` sits in a frozen 2x2 block off-goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b])) return true;
    if ((occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower-case LURD. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Player square after following `moves` (case ignored) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * Push-optimal A* for one (sub-)level. Returns {pushes, moves} or null when
 * the expansion cap is hit or the level is unsolvable.
 */
function solvePushOptimal(level: Level, maxStates: number): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  // Shortest-path distances on open floor from each goal (walls only).
  const dist: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi]!;
      for (const [dx, dy] of DIRS) {
        const n = p + dy * w + dx;
        if (n < 0 || n >= size || wall[n] || d[n] >= 0) continue;
        d[n] = d[p]! + 1;
        q.push(n);
      }
    }
    return d;
  });
  // All assignments of boxes to goals, as permutations of goal indices.
  const n = goals.length;
  const perms: number[][] = [];
  const used = new Array(n).fill(false);
  const cur: number[] = [];
  const genPerm = (): void => {
    if (cur.length === n) {
      perms.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      genPerm();
      cur.pop();
      used[i] = false;
    }
  };
  genPerm();
  const hCache = new Map<string, number>();
  const heuristic = (bs: number[]): number => {
    const key = bs.join(',');
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let best = Infinity;
    for (const perm of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = dist[perm[i]!]![bs[i]!]!;
        if (d < 0) {
          s = Infinity;
          break;
        }
        s += d;
      }
      if (s < best) best = s;
    }
    hCache.set(key, best);
    return best;
  };
  if (heuristic(level.boxes) === Infinity) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const keyOf = (bs: number[], low: number): number => {
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };

  // Binary heap on [f, -g, h, seq]: lowest f, then deepest, then lowest h.
  const heap: [number, number, number, number, number][] = [];
  const less = (a: [number, number, number, number, number], b: [number, number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && (a[2] < b[2] || (a[2] === b[2] && a[3] < b[3])))));
  const heapPush = (e: [number, number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const stateG: number[] = [0];
  const best = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  best.set(keyOf(level.boxes, reachLow(level, occupied, level.player, seen, ++stamp, stack)), 0);
  for (const b of level.boxes) occupied[b] = 0;
  const h0 = heuristic(level.boxes);
  heapPush([h0, 0, h0, 0, 0]);
  let seq = 1;
  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, negG, , , s] = heapPop();
    const g = -negG;
    if (g !== stateG[s]) continue;
    const boxes = statesBoxes[s]!;
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= maxStates) return null;
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    reachLow(level, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(level, occupied, goal, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        const h = heuristic(next);
        let key = -1;
        if (h !== Infinity) key = keyOf(next, reachLow(level, occupied, b, mark, ++stamp, stack));
        occupied[t] = 0;
        occupied[b] = 1;
        // Keep the cheapest known cost for a state: a later, cheaper path to
        // the same boxes and player region must replace the earlier one.
        if (h === Infinity) continue;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= g + 1) continue;
        best.set(key, g + 1);
        const idx = statesBoxes.length;
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(s);
        stateG.push(g + 1);
        heapPush([g + 1 + h, -(g + 1), h, seq++, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

type Room = { squares: number[]; boxes: number[]; goals: number[]; done: boolean };

/**
 * Split a level into rooms: cut the hallway row's dead squares (a box there
 * reaches no goal) out of the floor; each remaining part holding boxes is a
 * room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve a level room by room; falls back to a whole-level search. */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solved = solvePushOptimal(sub, MAX_ROOM_STATES);
      if (solved === null) {
        ok = false;
        break;
      }
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
    }
    if (ok) return moves;
  }
  const whole = solvePushOptimal(level, MAX_WHOLE_STATES);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
