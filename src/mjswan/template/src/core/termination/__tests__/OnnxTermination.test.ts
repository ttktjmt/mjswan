/**
 * Termination terms: traced-ONNX bodies, the legacy native `time_out`, and the manager's
 * OR-reduce with truncation split out.
 *
 * The graph's math is validated Python-side by the parity harness; what matters
 * here is the native half: bool decoding, a verdict that is this step's rather than the
 * last, holding the previous verdict rather than reporting "not done" on missing state,
 * and terminated-vs-truncated bookkeeping.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  FusedTermination,
  isFusedTerminationConfig,
} from '../FusedTermination';
import { OnnxTermination, type OnnxTerminationConfig } from '../OnnxTermination';
import { TerminationManager } from '../TerminationManager';
import { TimeOutTermination } from '../TimeOutTermination';
import type { OnnxSession, OnnxTensorLike } from '../../onnx/session';
import type { PolicyRunner } from '../../policy/PolicyRunner';

async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

/** Returns a scripted bool output; `manual` defers resolution. */
class FakeSession implements OnnxSession {
  readonly calls: Array<Record<string, OnnxTensorLike>> = [];
  private pending: Array<() => void> = [];

  constructor(
    private readonly respond: (call: number) => Uint8Array,
    private readonly manual = false,
  ) {}

  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    const index = this.calls.length;
    this.calls.push(feeds);
    const result = { done: { data: this.respond(index), dims: [1] } };
    if (!this.manual) return Promise.resolve(result);
    return new Promise(resolve => this.pending.push(() => resolve(result)));
  }

  flush(): void {
    this.pending.shift()?.();
  }
}

const runner = {} as unknown as PolicyRunner;

const FELL_OVER_CFG: OnnxTerminationConfig = {
  name: 'fell_over',
  onnx: 'term/fell_over.onnx',
  input_slots: [
    { entity: 'robot', field: 'projected_gravity_b', input: 'robot__projected_gravity_b' },
  ],
};

describe('OnnxTermination', () => {
  it('decodes a bool output and feeds the declared slot', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, 1]),
    });
    await term.step();
    expect(Object.keys(session.calls[0])).toEqual(['robot__projected_gravity_b']);
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('reports false while the graph says not done', async () => {
    const session = new FakeSession(() => Uint8Array.from([0]));
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, -1]),
    });
    await term.step();
    expect(term.evaluate({} as never)).toBe(false);
  });

  it('holds the previous verdict when a slot is unreadable', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    let available = true;
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => (available ? new Float32Array([0, 0, 1]) : null),
    });
    await term.step();
    const callsWhileAvailable = session.calls.length;

    available = false;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await term.step();
    // The graph is not run at all on missing state...
    expect(session.calls.length).toBe(callsWhileAvailable);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
    // ...and the last verdict stands, rather than letting a termination slip through.
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('warns and holds the verdict when the graph run rejects', async () => {
    // Escaping `step()`, a rejection would stop the step loop; unreported, a verdict
    // frozen at `false` is an episode that never ends.
    let fail = false;
    const session: OnnxSession = {
      run: () =>
        fail
          ? Promise.reject(new Error('boom'))
          : Promise.resolve({ done: { data: Uint8Array.from([1]), dims: [1] } }),
    };
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, 1]),
    });
    await term.step();

    fail = true;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(term.step()).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('warns and holds the verdict when a slot read throws', async () => {
    let fail = false;
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session: new FakeSession(() => Uint8Array.from([1])),
      readSlot: () => {
        if (fail) throw new Error('boom');
        return new Float32Array([0, 0, 1]);
      },
    });
    await term.step();

    fail = true;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(term.step()).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('reset() clears the verdict', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, 1]),
    });
    await term.step();
    expect(term.evaluate({} as never)).toBe(true);
    term.reset();
    expect(term.evaluate({} as never)).toBe(false);
  });
});

describe('TimeOutTermination', () => {
  it('fires on the step the counter reaches ceil(episode_length_s / step_dt)', () => {
    // Summing 0.02 five hundred times falls short of 10, so a clock would fire at 501.
    let steps = 0;
    const term = new TimeOutTermination(
      runner,
      { name: 'time_out', episode_length_s: 10 },
      () => steps,
      0.02,
    );
    steps = 499;
    expect(term.evaluate({} as never)).toBe(false);
    steps = 500;
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('never fires without a finite episode length', () => {
    // mjlab's play configs use an infinite horizon; a missing value must not fire always.
    for (const episode_length_s of [undefined, 0]) {
      const term = new TimeOutTermination(
        runner,
        { name: 'time_out', episode_length_s },
        () => 1e9,
        0.02,
      );
      expect(term.evaluate({} as never)).toBe(false);
    }
  });
});

describe('TerminationManager', () => {
  it('counts a legacy native time_out in steps and reports truncation', async () => {
    let steps = 2;
    const manager = new TerminationManager(
      {
        time_out: {
          name: 'time_out',
          native: 'elapsed_s >= episode_length_s',
          episode_length_s: 0.5,
          time_out: true,
        } as never,
      },
      {},
      runner,
      { episodeLength: () => steps, stepDt: 0.2 },
    );
    expect((await manager.evaluate({} as never)).done).toBe(false);
    steps = 3; // ceil(0.5 / 0.2)
    const result = await manager.evaluate({} as never);
    expect(result.done).toBe(true);
    // A timeout is a truncation, not a failure.
    expect(result.truncated).toBe(true);
    expect(result.terminated).toBe(false);
    expect(result.reasons).toEqual(['time_out']);
  });

  it('builds an ONNX term and reports it as a termination, not a truncation', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    const manager = new TerminationManager(
      { fell_over: { ...FELL_OVER_CFG } as never },
      {},
      runner,
      {
        onnxSessions: { get: () => session } as never,
        readOnnxSlot: () => new Float32Array([0, 0, 1]),
      },
    );
    const result = await manager.evaluate({} as never);
    expect(result.done).toBe(true);
    expect(result.terminated).toBe(true);
    expect(result.truncated).toBe(false);
    expect(result.reasons).toEqual(['fell_over']);
  });

  it('waits for the graph, so a verdict is never a step late', async () => {
    // A traced `time_out` read a step late would end every episode one step long.
    const session = new FakeSession(call => Uint8Array.from([call]), true);
    const manager = new TerminationManager(
      { fell_over: { ...FELL_OVER_CFG } as never },
      {},
      runner,
      {
        onnxSessions: { get: () => session } as never,
        readOnnxSlot: () => new Float32Array([0, 0, 1]),
      },
    );
    let settled = false;
    const first = manager.evaluate({} as never).then(result => {
      settled = true;
      return result;
    });
    await settle();
    expect(settled).toBe(false);
    session.flush();
    expect((await first).done).toBe(false);

    const second = manager.evaluate({} as never);
    session.flush();
    expect((await second).done).toBe(true);
  });

  it('warns and skips an ONNX term whose session is missing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const manager = new TerminationManager(
      { fell_over: { ...FELL_OVER_CFG } as never },
      {},
      runner,
    );
    // Losing one reset condition beats taking down the whole scene.
    expect(manager.size).toBe(0);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe('FusedTermination', () => {
  const FUSED = {
    fused: 'term/terminations.onnx',
    input_slots: [
      { entity: 'robot', field: 'root_link_pos_w', input: 'robot__root_link_pos_w' },
    ],
    lanes: [
      { name: 'anchor_pos' },
      { name: 'anchor_ori' },
      { name: 'ee_body_pos' },
      { name: 'time_out', time_out: true },
    ],
  };

  /** Emits one bool lane per term, as the fused graph does. */
  function laneSession(lanes: number[]): FakeSession {
    return new FakeSession(() => Uint8Array.from(lanes));
  }

  it('fans lanes back out into per-term reasons from one run', async () => {
    const session = laneSession([0, 1, 0, 0]);
    const manager = new TerminationManager({ __fused__: FUSED } as never, {}, runner, {
      onnxSessions: { get: () => session } as never,
      readOnnxSlot: () => new Float32Array([0, 0, 0.4]),
    });
    // Four terms, but the manager only ever runs one graph.
    expect(manager.size).toBe(4);

    const result = await manager.evaluate({} as never);
    expect(session.calls.length).toBe(1);
    expect(result.reasons).toEqual(['anchor_ori']);
    expect(result.terminated).toBe(true);
    expect(result.truncated).toBe(false);
  });

  it('keeps the truncation split per lane', async () => {
    // `time_out` truncates while its neighbour does not — a single OR would lose that.
    const session = laneSession([0, 0, 0, 1]);
    const manager = new TerminationManager({ __fused__: FUSED } as never, {}, runner, {
      onnxSessions: { get: () => session } as never,
      readOnnxSlot: () => new Float32Array([0, 0, 0.4]),
    });
    const result = await manager.evaluate({} as never);
    expect(result.reasons).toEqual(['time_out']);
    expect(result.truncated).toBe(true);
    expect(result.terminated).toBe(false);
  });

  it('reports every lane that fired', async () => {
    const session = laneSession([1, 0, 1, 0]);
    const manager = new TerminationManager({ __fused__: FUSED } as never, {}, runner, {
      onnxSessions: { get: () => session } as never,
      readOnnxSlot: () => new Float32Array([0, 0, 0.4]),
    });
    expect((await manager.evaluate({} as never)).reasons).toEqual([
      'anchor_pos',
      'ee_body_pos',
    ]);
  });

  it('holds every lane when a slot is unreadable', async () => {
    const session = laneSession([1, 1, 1, 1]);
    let available = true;
    const group = new FusedTermination(FUSED, {
      session,
      readSlot: () => (available ? new Float32Array([0, 0, 0.4]) : null),
    });
    await group.step();
    expect(group.verdict(0)).toBe(true);

    available = false;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await group.step();
    // Not re-run, and the verdicts stand rather than letting a termination through.
    expect(session.calls.length).toBe(1);
    expect(group.verdict(0)).toBe(true);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('warns and holds every lane when the graph run rejects', async () => {
    let fail = false;
    const session: OnnxSession = {
      run: () =>
        fail
          ? Promise.reject(new Error('boom'))
          : Promise.resolve({ done: { data: Uint8Array.from([1, 0, 0, 1]), dims: [4] } }),
    };
    const group = new FusedTermination(FUSED, {
      session,
      readSlot: () => new Float32Array([0, 0, 0.4]),
    });
    await group.step();

    fail = true;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(group.step()).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
    expect(FUSED.lanes.map((_, i) => group.verdict(i))).toEqual([true, false, false, true]);
  });

  it('reset() clears every lane', async () => {
    const group = new FusedTermination(FUSED, {
      session: laneSession([1, 1, 1, 1]),
      readSlot: () => new Float32Array([0, 0, 0.4]),
    });
    await group.step();
    group.reset();
    expect(FUSED.lanes.map((_, i) => group.verdict(i))).toEqual([
      false,
      false,
      false,
      false,
    ]);
  });

  it('warns and skips the whole group when the session is missing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const manager = new TerminationManager({ __fused__: FUSED } as never, {}, runner);
    expect(manager.size).toBe(0);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe('isFusedTerminationConfig', () => {
  it('separates a fused group from a single traced term', () => {
    expect(isFusedTerminationConfig({ fused: 'term/x.onnx', lanes: [] })).toBe(true);
    expect(isFusedTerminationConfig(FELL_OVER_CFG)).toBe(false);
    // `lanes` is what makes it usable; a path alone is not enough.
    expect(isFusedTerminationConfig({ fused: 'term/x.onnx' })).toBe(false);
  });
});
