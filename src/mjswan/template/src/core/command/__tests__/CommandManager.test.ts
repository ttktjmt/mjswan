/**
 * `CommandManager`: `OnnxCommand` registration.
 *
 * `OnnxCommand` bypasses the class registry the same way `OnnxEvent` bypasses
 * `EventManager`'s: it needs a session and rng the registry's
 * `new Term(name, config, context)` shape has no room for. These tests pin that bypass
 * and its failure mode (missing session/rng warns and skips rather than throwing and
 * taking down every other command).
 */
import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import { SeededRng } from '../../rng';
import { OnnxSessionCache, type OnnxSession, type OnnxTensorLike } from '../../onnx/session';
import { CommandManager } from '../CommandManager';
import type { CommandTermContext, CommandsConfig } from '../types';
import type { OnnxCommandConfig } from '../OnnxCommand';

/** Tensor data as a plain number array (the data union needs one narrowing). */
function values(tensor: OnnxTensorLike): number[] {
  return Array.from(tensor.data as ArrayLike<number>, Number);
}

function fakeSession(run: (feeds: Record<string, OnnxTensorLike>) => Record<string, OnnxTensorLike>): OnnxSession {
  return { run: (feeds) => Promise.resolve(run(feeds)) };
}

async function contextWithSession(path: string, session: OnnxSession): Promise<CommandTermContext> {
  const sessions = new OnnxSessionCache(() => Promise.resolve(session));
  await sessions.load([{ name: path, data: new ArrayBuffer(0) }]);
  return {
    mujoco: {} as unknown as CommandTermContext['mujoco'],
    mjModel: null,
    mjData: null,
    scene: {} as unknown as CommandTermContext['scene'],
    rng: new SeededRng(1),
    onnxSessions: sessions,
  };
}

const VELOCITY_CFG: OnnxCommandConfig = {
  name: 'OnnxCommand',
  onnx: 'command/twist.onnx',
  command_field: 'vel_command_b',
  rand_dim: 6,
  state_fields: [{ name: 'vel_command_b', shape: [1, 3], dtype: 'float32' }],
};

describe('CommandManager: resetTerms', () => {
  /** A term that records when its reset starts and finishes. */
  function recordingTerm(name: string, order: string[], delayed: boolean) {
    return {
      getCommand: () => new Float32Array(0),
      reset: async () => {
        order.push(`${name}:start`);
        if (delayed) await new Promise(resolve => setTimeout(resolve, 0));
        order.push(`${name}:done`);
      },
    };
  }

  it('leaves the UI values alone when a reset redraws the command', async () => {
    // The panel used to mirror `getCommand()` by position, so the drawn forward velocity
    // landed in the `enabled` checkbox and flipped the Joystick toggle on every reset.
    const context = await contextWithSession(
      'command/twist.onnx',
      fakeSession(() => ({ next_vel_command_b: { data: new Float32Array([0.9, 0, 0]), dims: [1, 3] } })),
    );
    const mgr = new CommandManager();
    mgr.initialize(
      {
        twist: {
          ...VELOCITY_CFG,
          ui: {
            inputs: [
              { type: 'checkbox', name: 'enabled', label: 'Joystick', default: false },
              { type: 'slider', name: 'lin_vel_x', label: 'Forward', min: -1, max: 1, step: 0.05, default: 0.5 },
            ],
          },
        } as OnnxCommandConfig,
      },
      context,
    );

    await mgr.resetTerms();

    expect(mgr.getValues()['twist:enabled']).toBe(0);
    expect(mgr.getValues()['twist:lin_vel_x']).toBe(0.5);
  });

  it('awaits each term in config order, not concurrently', () => {
    // Each reset *is* the term's resample and may write to the sim, so overlaps resolve
    // last-writer-wins by config order. `Promise.all` would interleave the four events.
    const order: string[] = [];
    const manager = new CommandManager();
    const terms = (manager as unknown as { terms: Map<string, unknown> }).terms;
    terms.set('slow', recordingTerm('slow', order, true));
    terms.set('fast', recordingTerm('fast', order, false));

    return manager.resetTerms().then(() => {
      expect(order).toEqual(['slow:start', 'slow:done', 'fast:start', 'fast:done']);
    });
  });
});

describe('CommandManager: OnnxCommand registration', () => {
  it('bypasses the class registry and constructs an OnnxCommand', async () => {
    const context = await contextWithSession(
      'command/twist.onnx',
      fakeSession(() => ({ next_vel_command_b: { data: new Float32Array([1, 2, 3]), dims: [1, 3] } })),
    );
    const mgr = new CommandManager();
    const commands: CommandsConfig = { twist: VELOCITY_CFG };
    mgr.initialize(commands, context);

    const term = mgr.getTerm('twist');
    expect(term).toBeDefined();
    await (term as unknown as { step(resample: boolean): Promise<void> }).step(true);
    expect(Array.from(mgr.getCommand('twist'))).toEqual([1, 2, 3]);
  });

  it('does not throw when a registry name is unrelated to OnnxCommand', () => {
    const mgr = new CommandManager();
    const context: CommandTermContext = {
      mujoco: {} as unknown as CommandTermContext['mujoco'],
      mjModel: null,
      mjData: null,
      scene: {} as unknown as CommandTermContext['scene'],
    };
    expect(() => mgr.initialize({ ui: { name: 'UiCommand' } }, context)).not.toThrow();
    expect(mgr.getTerm('ui')).toBeDefined();
  });

  it('warns and skips when no onnxSessions/rng are supplied (does not throw)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mgr = new CommandManager();
    const context: CommandTermContext = {
      mujoco: {} as unknown as CommandTermContext['mujoco'],
      mjModel: null,
      mjData: null,
      scene: {} as unknown as CommandTermContext['scene'],
    };
    expect(() => mgr.initialize({ twist: VELOCITY_CFG }, context)).not.toThrow();
    expect(mgr.getTerm('twist')).toBeUndefined();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('warns and skips when the named onnx session was never loaded', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const context = await contextWithSession('command/other.onnx', fakeSession(() => ({})));
    const mgr = new CommandManager();
    expect(() => mgr.initialize({ twist: VELOCITY_CFG }, context)).not.toThrow();
    expect(mgr.getTerm('twist')).toBeUndefined();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('threads context.readOnnxSlot into the constructed OnnxCommand', async () => {
    const runFn = vi.fn((feeds: Record<string, OnnxTensorLike>) => {
      expect(values(feeds.robot__heading_w)).toEqual([1.5]);
      return { next_vel_command_b: { data: new Float32Array(3), dims: [1, 3] } };
    });
    const context = await contextWithSession('command/twist.onnx', fakeSession(runFn));
    context.readOnnxSlot = (slot) => (slot.field === 'heading_w' ? new Float32Array([1.5]) : null);
    const mgr = new CommandManager();
    mgr.initialize(
      { twist: { ...VELOCITY_CFG, input_slots: [{ entity: 'robot', field: 'heading_w' }] } },
      context,
    );
    const term = mgr.getTerm('twist');
    await (term as unknown as { step(resample: boolean): Promise<void> }).step(true);
    expect(runFn).toHaveBeenCalledTimes(1);
  });
});

/**
 * A `UiCommand`'s value as a traced graph's slot. A term that merely forwards a
 * command is native, but one that does arithmetic on it is a traced body whose
 * `{command, field: 'command'}` slot has to resolve against the UI command itself —
 * this command has no Python side to read it from.
 */
describe('CommandManager: UiCommand state field', () => {
  const context = {
    mujoco: {} as unknown as CommandTermContext['mujoco'],
    mjModel: null,
    mjData: null,
    scene: {} as unknown as CommandTermContext['scene'],
  };
  const config: CommandsConfig = {
    compliance: {
      name: 'UiCommand',
      ui: {
        inputs: [
          { type: 'checkbox', name: 'enabled', label: 'Enabled', default: true },
          { type: 'slider', name: 'force', label: 'Force', min: 10, max: 20, default: 12, step: 0.5 },
        ],
      },
    },
  } as unknown as CommandsConfig;

  it('serves the UI values under the `command` field, and nothing else', () => {
    const mgr = new CommandManager();
    mgr.initialize(config, context as CommandTermContext);
    const term = mgr.getTerm('compliance') as unknown as {
      getStateField(field: string): Float32Array | null;
    };
    expect(Array.from(term.getStateField('command')!)).toEqual([1, 12]);
    expect(term.getStateField('vel_command_b')).toBeNull();
  });

  it('tracks a slider change, so the graph sees the live value', () => {
    const mgr = new CommandManager();
    mgr.initialize(config, context as CommandTermContext);
    mgr.setValue('compliance:force', 18);
    const term = mgr.getTerm('compliance') as unknown as {
      getStateField(field: string): Float32Array | null;
    };
    expect(Array.from(term.getStateField('command')!)).toEqual([1, 18]);
  });
});

/**
 * Button commands: a press has to reach the term, and the values it moved have to reach
 * back — mjlab's `Zero` sets its own sliders and the panel reads them off the mirror.
 */
describe('CommandManager: button commands', () => {
  const context = {
    mujoco: {} as unknown as CommandTermContext['mujoco'],
    mjModel: null,
    mjData: null,
    scene: {} as unknown as CommandTermContext['scene'],
  };
  const withButton = (buttonName: string): CommandsConfig =>
    ({
      twist: {
        name: 'UiCommand',
        ui: {
          inputs: [
            { type: 'slider', name: 'lin_vel_x', label: 'Forward', min: -1, max: 1, default: 0.5, step: 0.05 },
            { type: 'slider', name: 'ang_vel_z', label: 'Yaw', min: -1, max: 1, default: -0.4, step: 0.05 },
            { type: 'button', name: buttonName, label: 'Zero' },
          ],
        },
      },
    }) as unknown as CommandsConfig;

  it('registers a button as a command of its own, with no value', () => {
    const mgr = new CommandManager();
    mgr.initialize(withButton('zero'), context as CommandTermContext);
    const button = mgr.getCommands().find(cmd => cmd.config.type === 'button');
    expect(button?.id).toBe('twist:zero');
    // Only value inputs are mirrored; a button has nothing to mirror.
    expect(mgr.getValues()['twist:zero']).toBeUndefined();
  });

  it('zeroes the sliders it belongs to, and the panel sees it', () => {
    const mgr = new CommandManager();
    mgr.initialize(withButton('zero'), context as CommandTermContext);
    expect(mgr.getValues()['twist:lin_vel_x']).toBe(0.5);
    mgr.triggerButton('twist:zero');
    // The term moved its own sliders; the mirror the panel reads must follow.
    expect(mgr.getValues()['twist:lin_vel_x']).toBe(0);
    expect(mgr.getValues()['twist:ang_vel_z']).toBe(0);
    const term = mgr.getTerm('twist') as unknown as {
      getStateField(field: string): Float32Array | null;
    };
    expect(Array.from(term.getStateField('command')!)).toEqual([0, 0]);
  });

  it('emits a button event so a host app can hear the press', () => {
    const mgr = new CommandManager();
    mgr.initialize(withButton('zero'), context as CommandTermContext);
    const seen: string[] = [];
    mgr.addEventListener(event => {
      if (event.type === 'button') seen.push(event.commandId);
    });
    mgr.triggerButton('twist:zero');
    expect(seen).toEqual(['twist:zero']);
  });

  it('warns once for a name no term answers to, rather than looking live', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mgr = new CommandManager();
    mgr.initialize(withButton('launch'), context as CommandTermContext);
    mgr.triggerButton('twist:launch');
    mgr.triggerButton('twist:launch');
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain('launch');
    warn.mockRestore();
  });

  it('ignores an id that is not a button', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mgr = new CommandManager();
    mgr.initialize(withButton('zero'), context as CommandTermContext);
    mgr.triggerButton('twist:lin_vel_x');
    mgr.triggerButton('nope');
    expect(mgr.getValues()['twist:lin_vel_x']).toBe(0.5);
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });
});

/**
 * What the Debug Viz panel section lists, mirroring mjlab's `create_debug_vis_gui`:
 * only terms that draw, toggled through the term itself rather than a UI-side copy.
 */
describe('CommandManager: debug-vis toggles', () => {
  const VIZ_CFG: OnnxCommandConfig = {
    ...VELOCITY_CFG,
    debug_vis: true,
    viz: [{ shape: 'sphere', radius: 0.03, color: [1, 0, 0, 1], origin: { state: 'vel_command_b' } }],
  };

  async function managerWith(config: OnnxCommandConfig): Promise<CommandManager> {
    const context = await contextWithSession('command/twist.onnx', fakeSession(() => ({})));
    context.scene = new THREE.Scene();
    const mgr = new CommandManager();
    mgr.initialize({ twist: config, ui: { name: 'UiCommand' } }, context);
    return mgr;
  }

  it('lists only the terms that draw something', async () => {
    const mgr = await managerWith(VIZ_CFG);
    // `ui` draws nothing, so it gets no checkbox.
    expect(mgr.getDebugVisTerms()).toEqual([{ name: 'twist', enabled: true }]);
  });

  it('omits a term whose task left debug_vis off', async () => {
    const mgr = await managerWith({ ...VIZ_CFG, debug_vis: false });
    expect(mgr.getDebugVisTerms()).toEqual([]);
  });

  it('omits a term with no drawing even when debug_vis is on', async () => {
    const mgr = await managerWith({ ...VELOCITY_CFG, debug_vis: true });
    expect(mgr.getDebugVisTerms()).toEqual([]);
  });

  it('toggling reaches the term and is reported back', async () => {
    const mgr = await managerWith(VIZ_CFG);
    mgr.setDebugVisEnabled('twist', false);
    expect(mgr.getDebugVisTerms()).toEqual([{ name: 'twist', enabled: false }]);
  });

  it('emits so a subscribed panel re-reads the state', async () => {
    // Without the event the checkbox flips back on the next unrelated refresh.
    const mgr = await managerWith(VIZ_CFG);
    const seen: string[] = [];
    mgr.addEventListener(event => seen.push(event.type));
    mgr.setDebugVisEnabled('twist', true);
    expect(seen).toContain('debug_vis');
  });
});
