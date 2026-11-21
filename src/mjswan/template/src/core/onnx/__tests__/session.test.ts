/**
 * Input-slot naming contract.
 *
 * The build decides each slot's graph input name and ships it in the slot's own
 * `input` field (`mjswan.compile.tracer.slot_to_json`), so the runtime never has
 * to reproduce the naming scheme — which it could not do for sensor slots, whose
 * MJCF-path names get folded to identifiers at build time.
 */
import { describe, expect, it } from 'vitest';

import { queueOrtRun } from '../runQueue';
import { OnnxSessionCache, slotInputName, type OnnxSession } from '../session';

describe('slotInputName', () => {
  it('uses the build-supplied input name for an entity-data slot', () => {
    expect(
      slotInputName({ entity: 'robot', field: 'heading_w', input: 'robot__heading_w' }),
    ).toBe('robot__heading_w');
  });

  it('uses the build-supplied input name for a sensor slot', () => {
    // The sensor's real name keeps its MJCF path; only the graph input name is folded.
    expect(
      slotInputName({ sensor: 'robot/imu_lin_vel', input: 'sensor__robot_imu_lin_vel' }),
    ).toBe('sensor__robot_imu_lin_vel');
  });

  it('uses the build-supplied input name for a command-state slot', () => {
    expect(
      slotInputName({
        command: 'lift_height',
        field: 'target_pos',
        input: 'command__lift_height_target_pos',
      }),
    ).toBe('command__lift_height_target_pos');
  });

  it('falls back to the legacy entity__field scheme when input is absent', () => {
    expect(slotInputName({ entity: 'robot', field: 'joint_pos' })).toBe('robot__joint_pos');
  });

  it('falls back to the entity placeholder for a null entity', () => {
    expect(slotInputName({ entity: null, field: 'joint_pos' })).toBe('entity__joint_pos');
  });
});

describe('OnnxSessionCache.clear', () => {
  /** Queued and timer-held like a real `OrtSession`; `started` marks the body running. */
  type Fake = OnnxSession & {
    released: boolean;
    releasedDuringRun: boolean;
    started: Promise<void>;
  };

  function fake(): Fake {
    let running = false;
    let onStart!: () => void;
    const started = new Promise<void>((resolve) => {
      onStart = resolve;
    });
    return {
      released: false,
      releasedDuringRun: false,
      started,
      run() {
        return queueOrtRun(async () => {
          running = true;
          onStart();
          await new Promise((resolve) => setTimeout(resolve, 0));
          running = false;
          return {};
        });
      },
      async release() {
        this.released = true;
        this.releasedDuringRun = running;
      },
    };
  }

  it('releases every session it drops', async () => {
    const [a, b] = [fake(), fake()];
    const cache = new OnnxSessionCache(async (bytes) =>
      new Uint8Array(bytes)[0] === 1 ? a : b,
    );
    await cache.load([
      { name: 'obs/a.onnx', data: Uint8Array.of(1).buffer },
      { name: 'term/b.onnx', data: Uint8Array.of(2).buffer },
    ]);

    await cache.clear();

    expect([a.released, b.released]).toEqual([true, true]);
    expect(cache.size).toBe(0);
    expect(cache.get('obs/a.onnx')).toBeUndefined();
  });

  it('waits for an in-flight run rather than releasing under it', async () => {
    const session = fake();
    const cache = new OnnxSessionCache(async () => session);
    await cache.load([{ name: 'obs/a.onnx', data: new ArrayBuffer(1) }]);

    // Not awaited: the release must queue behind the run, not race it.
    const run = cache.get('obs/a.onnx')!.run({});
    await session.started;
    await cache.clear();
    await run;

    expect(session.released).toBe(true);
    expect(session.releasedDuringRun).toBe(false);
  });

  it('is a no-op on an empty cache', async () => {
    await expect(new OnnxSessionCache().clear()).resolves.toBeUndefined();
  });

  it('releases the rest, and does not throw, when one release fails', async () => {
    const good = fake();
    const bad = { ...fake(), release: () => Promise.reject(new Error('boom')) };
    const cache = new OnnxSessionCache(async (bytes) =>
      new Uint8Array(bytes)[0] === 1 ? bad : good,
    );
    await cache.load([
      { name: 'a.onnx', data: Uint8Array.of(1).buffer },
      { name: 'b.onnx', data: Uint8Array.of(2).buffer },
    ]);

    await expect(cache.clear()).resolves.toBeUndefined();

    expect(good.released).toBe(true);
  });
});
