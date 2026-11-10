/**
 * Name patterns from a document, matched on a Pike VM: the cost is O(program × name)
 * whatever the pattern, where V8's backtracking RegExp can take exponential time.
 */
const MAX_PROGRAM = 2000;
/** A repeated empty group emits no instruction, so only this ends its loop. */
const MAX_EMITS = 4 * MAX_PROGRAM;

type Test = (c: string) => boolean;
type Node =
  | { k: 'char'; test: Test }
  | { k: 'assert'; at: '^' | '$' }
  | { k: 'cat'; parts: Node[] }
  | { k: 'alt'; parts: Node[] }
  | { k: 'rep'; node: Node; min: number; max: number };
interface Ins { op: 'char' | 'assert' | 'split' | 'jmp' | 'match'; test?: Test; at?: string; x?: number; y?: number }

function parse(src: string): Node {
  let i = 0;
  const fail = (why: string): never => {
    throw new Error(`name pattern ${JSON.stringify(src)}: ${why} at ${i}`);
  };
  const single = (re: RegExp): Node => ({ k: 'char', test: (c) => re.test(c) });
  const alt = (): Node => {
    const parts = [cat()];
    while (src[i] === '|') { i++; parts.push(cat()); }
    return parts.length === 1 ? parts[0] : { k: 'alt', parts };
  };
  const cat = (): Node => {
    const parts: Node[] = [];
    while (i < src.length && src[i] !== '|' && src[i] !== ')') parts.push(quant(atom()));
    return { k: 'cat', parts };
  };
  const quant = (node: Node): Node => {
    const brace = src[i] === '{' ? /^\{(\d+)(,(\d*))?\}/.exec(src.slice(i)) : null;
    let min: number, max: number;
    if (src[i] === '*') [min, max] = [0, Infinity];
    else if (src[i] === '+') [min, max] = [1, Infinity];
    else if (src[i] === '?') [min, max] = [0, 1];
    else if (brace) [min, max] = [+brace[1], brace[2] ? (brace[3] ? +brace[3] : Infinity) : +brace[1]];
    else return node;
    if (max < min) fail('numbers out of order');
    i += brace ? brace[0].length : 1;
    if (src[i] === '?') i++; // lazy matches the same names
    return { k: 'rep', node, min, max };
  };
  const atom = (): Node => {
    const c = src[i++];
    if (c === '(') {
      if (src.startsWith('?:', i)) i += 2;
      else if (src[i] === '?') fail('lookaround and named groups are not supported');
      const inner = alt();
      if (src[i++] !== ')') fail('unclosed group');
      return inner;
    }
    if (c === '[') {
      const start = i - 1;
      while (i < src.length && src[i] !== ']') i += src[i] === '\\' ? 2 : 1;
      if (src[i++] !== ']') fail('unclosed class');
      return single(new RegExp(`^${src.slice(start, i)}$`)); // one character: cannot backtrack
    }
    if (c === '.') return single(/^.$/);
    if (c === '^' || c === '$') return { k: 'assert', at: c };
    if (c === '\\') {
      const e = src[i++] ?? fail('trailing backslash');
      if ('dDwWsS'.includes(e)) return single(new RegExp(`^\\${e}$`));
      if (/[0-9A-Za-z]/.test(e)) fail(`unsupported escape \\${e}`);
      return { k: 'char', test: (ch) => ch === e };
    }
    if ('*+?'.includes(c)) fail('nothing to repeat');
    return { k: 'char', test: (ch) => ch === c };
  };
  const root = alt();
  if (i < src.length) fail('unmatched )');
  return root;
}

function toProgram(root: Node): Ins[] {
  const prog: Ins[] = [];
  const push = (ins: Ins): Ins => {
    if (prog.length >= MAX_PROGRAM) throw new Error('name pattern too large');
    prog.push(ins);
    return ins;
  };
  let emits = 0;
  const emit = (n: Node): void => {
    if (++emits > MAX_EMITS) throw new Error('name pattern too large');
    if (n.k === 'char') push({ op: 'char', test: n.test });
    else if (n.k === 'assert') push({ op: 'assert', at: n.at });
    else if (n.k === 'cat') n.parts.forEach(emit);
    else if (n.k === 'alt') {
      const jumps: Ins[] = [];
      n.parts.forEach((part, idx) => {
        if (idx === n.parts.length - 1) return emit(part);
        const split = push({ op: 'split', x: prog.length + 1 });
        emit(part);
        jumps.push(push({ op: 'jmp' }));
        split.y = prog.length;
      });
      for (const j of jumps) j.x = prog.length;
    } else {
      for (let r = 0; r < n.min; r++) emit(n.node);
      const splits: Ins[] = [];
      if (n.max === Infinity) {
        const loop = prog.length;
        splits.push(push({ op: 'split', x: loop + 1 }));
        emit(n.node);
        push({ op: 'jmp', x: loop });
      } else {
        for (let r = n.min; r < n.max; r++) {
          splits.push(push({ op: 'split', x: prog.length + 1 }));
          emit(n.node);
        }
      }
      for (const s of splits) s.y = prog.length;
    }
  };
  emit(root);
  push({ op: 'match' });
  return prog;
}

/** Full-match test for an action-term name pattern (`^(?:p)$`), linear in the name. */
export function compileNamePattern(pattern: string): (name: string) => boolean {
  const prog = toProgram(parse(pattern));
  return (name) => {
    const seen = new Int32Array(prog.length).fill(-1);
    const add = (list: number[], pc: number, pos: number): void => {
      const stack = [pc];
      while (stack.length) {
        const p = stack.pop()!;
        if (seen[p] === pos) continue;
        seen[p] = pos;
        const ins = prog[p];
        if (ins.op === 'jmp') stack.push(ins.x!);
        else if (ins.op === 'split') stack.push(ins.y!, ins.x!);
        else if (ins.op === 'assert') {
          if (ins.at === '^' ? pos === 0 : pos === name.length) stack.push(p + 1);
        } else list.push(p);
      }
    };
    let cur: number[] = [];
    add(cur, 0, 0);
    for (let i = 0; i < name.length && cur.length; i++) {
      const next: number[] = [];
      for (const p of cur) if (prog[p].op === 'char' && prog[p].test!(name[i])) add(next, p + 1, i + 1);
      cur = next;
    }
    return cur.some((p) => prog[p].op === 'match');
  };
}
