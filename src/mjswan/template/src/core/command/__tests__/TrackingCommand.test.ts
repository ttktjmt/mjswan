/**
 * `motion` stays a native command (a clip lookup is data, not math), so the tracking task's
 * traced observations and terminations read their `{command: "motion", field}` slots off
 * `TrackingCommand` itself. This pins the one part of that which can be silently wrong: the
 * re-anchoring frame, mjlab's `update_relative_body_poses`.
 */
import JSZip from 'jszip';
import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import { TrackingCommand, reanchorBodyPositions } from '../TrackingCommand';
import type { CommandConfigEntry, CommandTermContext } from '../types';

function close(actual: Float32Array, expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 6);
}

describe('reanchorBodyPositions', () => {
  it('takes x/y from the robot anchor, z from the reference, and rotates by the yaw between them', () => {
    const s = Math.SQRT1_2; // 90 deg about +z
    const relative = reanchorBodyPositions(
      // Two reference bodies: one 1 m ahead of the anchor, one 1 m above it.
      Float32Array.from([1, 0, 1, 0, 0, 2]),
      [0, 0, 1],
      [1, 0, 0, 0],
      [5, 7, 2],
      [s, 0, 0, s],
    );
    // Anchor lands at (5, 7, 1) — the robot's x/y, the reference's z, so a reader
    // that took the robot's z would put the whole skeleton 1 m too high.
    // The +x offset rotates onto +y; the +z offset is untouched by yaw.
    close(relative, [5, 8, 1, 5, 7, 2]);
  });

  it('is the identity when the robot sits exactly on the reference anchor', () => {
    const bodies = Float32Array.from([1, 2, 3, -1, 0, 0.5]);
    close(reanchorBodyPositions(bodies, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [1, 0, 0, 0]), [
      1, 2, 3, -1, 0, 0.5,
    ]);
  });
});

/**
 * The `ref_*` look-ahead window, which mjlab's `MotionCommand` has no equivalent of. A
 * wrong window is silent — the policy runs, tracking the wrong part of the clip — so the
 * offset→frame mapping, edge clamping and not-ready fallback are all pinned. No model
 * needed: the reference buffers are the command's own state.
 */
function trackingCommand(timeSteps: number[], frames: number): TrackingCommand {
  const config = {
    name: 'TrackingCommand',
    time_steps: timeSteps,
  } as unknown as CommandConfigEntry;
  const context = {
    mujoco: {},
    mjModel: null,
    mjData: null,
    scene: new THREE.Scene(),
  } as unknown as CommandTermContext;
  const term = new TrackingCommand('motion', config, context);
  term.refLen = frames;
  term.nJoints = 2;
  // Frame i is identifiable in every field: position (i, 0, 0), joints (i, -i).
  term.refRootPos = Array.from({ length: frames }, (_, i) => Float32Array.from([i, 0, 0]));
  term.refRootQuat = Array.from({ length: frames }, () => Float32Array.from([1, 0, 0, 0]));
  term.refJointPos = Array.from({ length: frames }, (_, i) => Float32Array.from([i, -i]));
  // `isReady()` also needs a selected motion; the window never reads its contents.
  (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
  return term;
}

describe('TrackingCommand command slot', () => {
  it('serves `command` as getCommand(): what `get_command("motion")` reads in mjlab', () => {
    const term = trackingCommand([0], 4);
    (term as unknown as { selectedMotion: unknown }).selectedMotion = {
      jointVel: Array.from({ length: 4 }, (_, i) => Float32Array.from([10 * i, -10 * i])),
    };
    term.refIdx = 2;
    // Joint positions, then joint velocities.
    close(term.getStateField('command')!, [2, -2, 20, -20]);
  });
});

describe('TrackingCommand ref window', () => {
  it('samples each field at every time_steps offset, in order', () => {
    const term = trackingCommand([0, 2, -1], 10);
    term.refIdx = 5;
    // Offsets 0/+2/-1 of frame 5 -> frames 5, 7, 4.
    close(term.getStateField('ref_root_pos_w')!, [5, 0, 0, 7, 0, 0, 4, 0, 0]);
    close(term.getStateField('ref_joint_pos')!, [5, -5, 7, -7, 4, -4]);
  });

  it('clamps a window running off either end rather than wrapping', () => {
    const term = trackingCommand([-4, 0, 4], 3);
    term.refIdx = 0;
    // Wrapping would read frame 2 for the -4 offset; clamping repeats frame 0.
    close(term.getStateField('ref_root_pos_w')!, [0, 0, 0, 0, 0, 0, 2, 0, 0]);
  });

  it('reports readiness, and falls back to finite values before a clip loads', () => {
    const term = trackingCommand([0, 1], 2);
    (term as unknown as { selectedMotion: unknown }).selectedMotion = null;
    close(term.getStateField('is_ready')!, [0]);
    close(term.getStateField('ref_root_pos_w')!, [0, 0, 0, 0, 0, 0]);
    // Identity quats, not zeros: the term normalizes these, and a zero quat is NaN.
    close(term.getStateField('ref_root_quat_w')!, [1, 0, 0, 0, 1, 0, 0, 0]);

    (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
    close(term.getStateField('is_ready')!, [1]);
  });

  it('defaults to the current frame alone when no time_steps are configured', () => {
    const config = { name: 'TrackingCommand' } as unknown as CommandConfigEntry;
    const context = {
      mujoco: {},
      mjModel: null,
      mjData: null,
      scene: new THREE.Scene(),
    } as unknown as CommandTermContext;
    const term = new TrackingCommand('motion', config, context);
    term.refLen = 4;
    term.refIdx = 2;
    term.refRootPos = Array.from({ length: 4 }, (_, i) => Float32Array.from([i, 0, 0]));
    (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
    close(term.getStateField('ref_root_pos_w')!, [2, 0, 0]);
  });
});

/** mjlab's `_debug_vis_impl` in `"ghost"` mode. */
describe('TrackingCommand ghost', () => {
  function ghostCommand(config: Record<string, unknown> = {}): { term: TrackingCommand; scene: THREE.Scene } {
    const scene = new THREE.Scene();
    const link = new THREE.Group();
    link.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()));
    const context = {
      mujoco: { MjData: class {} },
      // Body 1 hangs off a joint, so the ghost clones it.
      mjModel: { nbody: 2, body_jntnum: [0, 1], body_parentid: [0, 0] },
      mjData: null,
      scene,
      bodies: { 1: link },
    } as unknown as CommandTermContext;
    const term = new TrackingCommand(
      'motion',
      { name: 'TrackingCommand', ...config } as unknown as CommandConfigEntry,
      context,
    );
    (term as unknown as { selectedMotion: unknown }).selectedMotion = {};
    term.refLen = 1;
    return { term, scene };
  }

  const ghostOf = (scene: THREE.Scene) => scene.getObjectByName('Tracking Ghost')!;

  it('draws only while Debug Viz and its own checkbox are both on', () => {
    const { term, scene } = ghostCommand({ debug_vis: true });
    term.updateDebugVisuals(true);
    expect(ghostOf(scene).visible).toBe(true);

    term.updateDebugVisuals(false);
    expect(ghostOf(scene).visible).toBe(false);

    term.setDebugVisEnabled(false);
    term.updateDebugVisuals(true);
    expect(ghostOf(scene).visible).toBe(false);
    expect(term.debugVisEnabled()).toBe(false);
  });

  it('stays hidden with no motion selected', () => {
    const { term, scene } = ghostCommand();
    (term as unknown as { selectedMotion: unknown }).selectedMotion = null;
    term.updateDebugVisuals(true);
    expect(ghostOf(scene).visible).toBe(false);
  });

  it('builds no ghost and offers no checkbox when the task leaves debug_vis off', () => {
    const { term, scene } = ghostCommand({ debug_vis: false });
    expect(term.debugVisEnabled()).toBeNull();
    expect(scene.getObjectByName('Tracking Ghost')).toBeUndefined();
  });

  it('draws for a document written before debug_vis was carried', () => {
    expect(ghostCommand().term.debugVisEnabled()).toBe(true);
  });

  it("paints the ghost in the task's ghost_color at viser's ghost opacity", () => {
    const { scene } = ghostCommand({ ghost_color: [1, 0, 0, 0.25] });
    const mesh = ghostOf(scene).getObjectByProperty('type', 'Mesh') as THREE.Mesh;
    const material = mesh.material as THREE.MeshStandardMaterial;
    expect(material.color.toArray()).toEqual([1, 0, 0]);
    // The color's alpha only marks visual geoms in mjlab; `add_ghost_mesh` draws at 0.5.
    expect(material.opacity).toBe(0.5);
    expect(mesh.castShadow).toBe(false);
  });
});

/** mjlab's `viz.mode="frames"`: reference and robot frames for each body and the anchor. */
describe('TrackingCommand frames', () => {
  it('draws three axes for each of the four frames of one body, and builds no ghost', () => {
    const scene = new THREE.Scene();
    const context = {
      mujoco: { MjData: class {} },
      mjModel: {
        nbody: 2,
        body: (i: number) => ({ name: ['world', 'robot/pelvis'][i] }),
        body_jntnum: [0, 1],
        body_parentid: [0, 0],
      },
      mjData: {
        xpos: Float64Array.from([0, 0, 0, 0.2, 0, 0.9]),
        xquat: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0]),
      },
      scene,
      bodies: { 1: new THREE.Group() },
    } as unknown as CommandTermContext;
    const motion = { name: 'clip', body_names: ['pelvis'], anchor_body_name: 'pelvis' };
    const term = new TrackingCommand(
      'motion',
      { name: 'TrackingCommand', viz_mode: 'frames', motions: [motion] } as unknown as CommandConfigEntry,
      context,
    );
    const shafts = () =>
      (scene.getObjectByName('Tracking Frames')!.children[0] as THREE.InstancedMesh).count;
    expect(scene.getObjectByName('Tracking Ghost')).toBeUndefined();
    expect(term.debugVisEnabled()).toBe(true);

    term.updateDebugVisuals(true);
    expect(shafts()).toBe(0); // No clip yet.

    const internals = term as unknown as Record<string, unknown>;
    internals.selectedMotion = motion;
    internals.refBodyPosW = [Float32Array.from([0, 0, 1])];
    internals.refBodyQuatW = [Float32Array.from([1, 0, 0, 0])];
    term.refLen = 1;
    term.updateDebugVisuals(true);
    expect(shafts()).toBe(3 * 4);
    term.updateDebugVisuals(false);
    expect(shafts()).toBe(0);
  });
});

/** One float32 `.npy`, as `np.save` writes it. */
function npy(shape: number[], values: number[]): Uint8Array {
  const header = new TextEncoder().encode(
    `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')},), }\n`,
  );
  const out = new Uint8Array(10 + header.length + 4 * values.length);
  const view = new DataView(out.buffer);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  view.setUint16(8, header.length, true);
  out.set(header, 10);
  values.forEach((value, i) => view.setFloat32(10 + header.length + 4 * i, value, true));
  return out;
}

/** A body_world clip of `frames` frames whose `latent` row i is (10i, 10i + 1). */
function clip(frames: number, extra: Record<string, Uint8Array> = {}): Promise<ArrayBuffer> {
  const zip = new JSZip();
  const zeros = (width: number) => npy([frames, width], new Array(frames * width).fill(0));
  const frameIndex = Array.from({ length: frames }, (_, i) => i);
  zip.file('joint_pos.npy', npy([frames, 1], frameIndex));
  zip.file('joint_vel.npy', zeros(1));
  zip.file('body_pos_w.npy', zeros(3));
  zip.file('body_quat_w.npy', npy([frames, 4], frameIndex.flatMap(() => [1, 0, 0, 0])));
  zip.file('body_lin_vel_w.npy', zeros(3));
  zip.file('body_ang_vel_w.npy', zeros(3));
  zip.file('latent.npy', npy([frames, 2], frameIndex.flatMap((i) => [10 * i, 10 * i + 1])));
  for (const [name, bytes] of Object.entries(extra)) zip.file(`${name}.npy`, bytes);
  return zip.generateAsync({ type: 'arraybuffer', compression: 'STORE' });
}

/**
 * A motion's `frame_fields`: arrays of its own `.npz` a traced graph reads as
 * `{command: "motion", field}` slots, at the frame the native fields are read at.
 */
describe('TrackingCommand frame fields', () => {
  async function tracker(motions: Array<Record<string, unknown>>, extra: Record<string, unknown> = {}) {
    const requestReset = vi.fn();
    const context = {
      mujoco: {},
      mjModel: null,
      mjData: null,
      scene: new THREE.Scene(),
      requestReset,
    } as unknown as CommandTermContext;
    const config = { name: 'TrackingCommand', motions, ...extra } as unknown as CommandConfigEntry;
    return { term: new TrackingCommand('motion', config, context), requestReset };
  }
  const frameOf = (term: TrackingCommand) => term.getStateField('latent')![0] / 10;
  const walk = async (frames = 4) => ({
    name: 'walk',
    data: await clip(frames),
    fps: 10,
    anchor_body_name: 'pelvis',
    body_names: ['pelvis'],
    frame_fields: ['latent'],
  });

  it('serves the current frame after select, step, loop and reset', async () => {
    const { term, requestReset } = await tracker([await walk()]);
    expect(term.getStateField('latent')).toBeNull(); // Nothing selected yet.

    await term.setSelectedMotion('walk');
    close(term.getStateField('latent')!, [10, 11]);
    // The same frame as the native fields.
    close(term.getStateField('joint_pos')!, [1]);
    term.update(0.1);
    close(term.getStateField('latent')!, [20, 21]);

    term.update(0.1);
    close(term.getStateField('latent')!, [30, 31]);
    term.update(0.1); // Past the last frame: wraps and asks for an episode reset.
    close(term.getStateField('latent')!, [0, 1]);
    expect(requestReset).toHaveBeenCalledTimes(1);
    term.reset(); // That reset, which starts at the frame it writes.
    close(term.getStateField('latent')!, [0, 1]);

    term.update(0.1);
    term.update(0.1);
    close(term.getStateField('latent')!, [20, 21]);
    term.reset();
    close(term.getStateField('latent')!, [10, 11]);
  });

  it("serves mjlab's frames across a reset, its steps and an auto-reset", async () => {
    // mjlab's `time_steps` after `env.reset()`, three steps, a step that auto-resets, and
    // one more: each reset writes frame 0, and the compute that follows it steps to 1.
    const { term } = await tracker([await walk(8)]);
    await term.setSelectedMotion('walk');
    const served = [frameOf(term)];
    for (let i = 0; i < 3; i++) {
      term.update(0.1);
      served.push(frameOf(term));
    }
    term.reset();
    term.update(0.1, true);
    served.push(frameOf(term));
    term.update(0.1);
    served.push(frameOf(term));
    expect(served).toEqual([1, 2, 3, 4, 1, 2]);
  });

  it('serves the frame a reset writes first for a command that holds it', async () => {
    const { term } = await tracker([await walk(8)], { hold_reset_frame: true });
    await term.setSelectedMotion('walk');
    const served = [frameOf(term)];
    term.update(0.1);
    served.push(frameOf(term));
    term.reset();
    term.update(0.1, true);
    served.push(frameOf(term));
    term.update(0.1);
    served.push(frameOf(term));
    expect(served).toEqual([0, 1, 0, 1]);
  });

  it('hands out a copy, and null for a field no motion carries', async () => {
    const { term } = await tracker([await walk()]);
    await term.setSelectedMotion('walk');
    term.getStateField('latent')!.fill(99);
    close(term.getStateField('latent')!, [10, 11]);
    expect(term.getStateField('phase')).toBeNull();
  });

  it("serves the selected motion's own rows", async () => {
    const run = { ...(await walk()), name: 'run', data: await clip(2, { latent: npy([2, 2], [5, 6, 7, 8]) }) };
    const { term } = await tracker([await walk(), run]);
    await term.setSelectedMotion('run');
    close(term.getStateField('latent')!, [7, 8]);
    await term.setSelectedMotion('walk');
    close(term.getStateField('latent')!, [10, 11]);
  });

  it('serves a qpos clip the same way', async () => {
    const zip = new JSZip();
    zip.file('qpos.npy', npy([3, 1], [0, 0, 0]));
    zip.file('latent.npy', npy([3, 1], [4, 5, 6]));
    const data = await zip.generateAsync({ type: 'arraybuffer', compression: 'STORE' });
    const { term } = await tracker([
      { ...(await walk()), data, clip_format: 'qpos' },
    ]);
    await term.setSelectedMotion('walk');
    close(term.getStateField('latent')!, [5]);
    term.update(0.1);
    close(term.getStateField('latent')!, [6]);
  });

  it('refuses a clip missing a frame field it lists', async () => {
    const { term } = await tracker([{ ...(await walk()), frame_fields: ['latent', 'phase'] }]);
    await expect(term.setSelectedMotion('walk')).rejects.toThrow("missing frame field 'phase'");
  });
});

describe('TrackingCommand reference state', () => {
  it("writes the root's angular velocity into qvel in the body's frame, as mjlab does", async () => {
    // Yawed 90 degrees and turning about the world's x axis, which is the body's -y.
    const s = Math.SQRT1_2;
    const zip = new JSZip();
    zip.file('joint_pos.npy', npy([2, 1], [0, 0]));
    zip.file('joint_vel.npy', npy([2, 1], [0, 0]));
    zip.file('body_pos_w.npy', npy([2, 3], [0, 0, 1, 0, 0, 1]));
    zip.file('body_quat_w.npy', npy([2, 4], [s, 0, 0, s, s, 0, 0, s]));
    zip.file('body_lin_vel_w.npy', npy([2, 3], [0.5, 0, 0, 0.5, 0, 0]));
    zip.file('body_ang_vel_w.npy', npy([2, 3], [1, 0, 0, 1, 0, 0]));
    const data = await zip.generateAsync({ type: 'arraybuffer', compression: 'STORE' });
    const mjModel = { njnt: 1, nbody: 2, jnt_type: [0], jnt_qposadr: [0], jnt_dofadr: [0] };
    const mjData = { qpos: new Float64Array(7), qvel: new Float64Array(6) };
    const context = {
      mujoco: { mj_forward: vi.fn(), MjData: class { qpos = new Float64Array(7); } },
      mjModel,
      mjData,
      scene: new THREE.Scene(),
    } as unknown as CommandTermContext;
    const motion = { name: 'turn', data, fps: 10, anchor_body_name: 'pelvis', body_names: ['pelvis'] };
    const config = { name: 'TrackingCommand', debug_vis: false, motions: [motion] };
    const term = new TrackingCommand('motion', config as unknown as CommandConfigEntry, context);

    await term.setSelectedMotion('turn');
    close(Float32Array.from(mjData.qvel), [0.5, 0, 0, 0, -1, 0]);
  });
});

describe('TrackingCommand rate', () => {
  async function tracker(motion: Record<string, unknown>) {
    const requestReset = vi.fn();
    const context = {
      mujoco: {},
      mjModel: null,
      mjData: null,
      scene: new THREE.Scene(),
      requestReset,
    } as unknown as CommandTermContext;
    const walk = {
      name: 'walk',
      data: await clip(4),
      fps: 10,
      anchor_body_name: 'pelvis',
      body_names: ['pelvis'],
      frame_fields: ['latent'],
      ...motion,
    };
    const config = { name: 'TrackingCommand', motions: [walk] } as unknown as CommandConfigEntry;
    const term = new TrackingCommand('motion', config, context);
    return { term, requestReset, frame: () => term.getStateField('latent')![0] / 10 };
  }

  it('advances several frames in one step, wrapping as one frame at a time did', async () => {
    const { term, requestReset, frame } = await tracker({ fps: 30 });
    await term.setSelectedMotion('walk');
    expect(frame()).toBe(1);
    term.update(0.1); // Three frames: past the last of four, so back to the first.
    expect(frame()).toBe(0);
    expect(requestReset).toHaveBeenCalledTimes(1);
  });

  it('holds the last frame of a clip that does not loop', async () => {
    const { term, requestReset, frame } = await tracker({ fps: 100, loop: false });
    await term.setSelectedMotion('walk');
    term.update(0.1);
    expect(frame()).toBe(3);
    expect(requestReset).not.toHaveBeenCalled();
  });

  // Infinity is what JSON's 1e309 parses to; 1e18 is past where adding 1 changes nothing.
  it.each([Infinity, 1e18, 0, -10, Number.NaN])('refuses fps %s', async (fps) => {
    const { term } = await tracker({ fps });
    await expect(term.setSelectedMotion('walk')).rejects.toThrow(/fps/);
  });
});
