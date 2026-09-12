import { describe, it, expect } from 'vitest';
import { MAX_DOCUMENT_FORMAT, parseManifest, sanitizeName, type Manifest, type ByteSource } from './index';
import NAME2ID_CASES from './name2id_cases.json';

/** Records every requested path; returns tagged bytes for each. */
function fakeSource(): { source: ByteSource; requested: string[] } {
  const requested: string[] = [];
  const source: ByteSource = (relPath) => {
    requested.push(relPath);
    return async () => new TextEncoder().encode(`bytes:${relPath}`).buffer as ArrayBuffer;
  };
  return { source, requested };
}

const MANIFEST: Manifest = {
  format: 1,
  version: '0.0.0',
  projects: [
    {
      id: 'demo',
      name: 'Demo',
      scenes: [
        {
          id: 'humanoid',
          name: 'Humanoid',
          scene: 'scene.mjz',
          control_dt: 0.02,
          camera: { distance: 5, origin_type: 'ASSET_BODY', body_name: 'torso' },
          splat_section: true,
          splats: [
            { id: 'room', name: 'Room', path: 'assets/room.spz', control: true, scale: 2, x_offset: 0.5 },
          ],
          mdps: [
            {
              id: 'mdp_0',
              observations: {
                actor: [
                  { name: 'joint_pos', onnx: 'mdp/mdp_0/obs/joint_pos.onnx' },
                  { name: 'actions', native: 'prev_action' },
                ],
              },
              actions: { joint_pos: { type: 'joint_position' } },
              terminations: { fell_over: { name: 'fell_over', onnx: 'mdp/mdp_0/term/fell_over.onnx' } },
              commands: { twist: { name: 'OnnxCommand', onnx: 'mdp/mdp_0/command/twist.onnx' } },
              events: [
                { name: 'push_robot', onnx: 'mdp/mdp_0/event/push_robot.onnx' },
                { name: 'randomize_terrain' },
              ] as never,
            },
            { id: 'mdp_1', actions: { joint_pos: { type: 'joint_position' } } },
          ],
          policies: [
            {
              id: 'walk',
              name: 'walk',
              default: true,
              mdp: 'mdp_0',
              onnx: 'policy/walk.onnx',
              policy_joint_names: ['a', 'b'],
              motions: [{ name: 'clip', path: 'assets/walk_clip.npz', fps: 50, default: true }],
            },
            { id: 'run', name: 'run', mdp: 'mdp_1', onnx: 'policy/run.onnx', in_keys: ['obs'], out_keys: ['action'] },
          ],
        },
      ],
    },
  ],
};

describe('parseManifest', () => {
  it('builds a single-project catalog with scene/policy/splat entries carrying ids', () => {
    const catalog = parseManifest(MANIFEST, fakeSource().source);
    expect(catalog.projects.map((p) => [p.id, p.name, p.default])).toEqual([['demo', 'Demo', true]]);
    const scene = catalog.projects[0].scenes[0];
    expect([scene.id, scene.name]).toEqual(['humanoid', 'Humanoid']);
    expect(scene.splatSection).toBe(true);
    expect(scene.policies.map((p) => p.id)).toEqual(['walk', 'run']);
    expect(scene.policies[0].default).toBe(true);
    expect(scene.policies[0].motions).toEqual([{ name: 'clip', default: true }]);
    expect(scene.splats[0]).toMatchObject({ id: 'room', name: 'Room', control: true, transform: { scale: 2, xOffset: 0.5 } });
  });

  it('resolves every path under a scene against <project-id>/<scene-id>/', async () => {
    const { source, requested } = fakeSource();
    const catalog = parseManifest(MANIFEST, source);
    const input = await catalog.projects[0].scenes[0].buildScene({ policy: 'walk', splat: 'room' });
    expect(requested).toContain('demo/humanoid/scene.mjz');
    expect(requested).toContain('demo/humanoid/policy/walk.onnx');
    expect(requested).toContain('demo/humanoid/assets/walk_clip.npz');
    expect(requested).toContain('demo/humanoid/assets/room.spz');
    expect(input.controlDt).toBe(0.02);
    expect(input.policy?.motions?.map((m) => m.name)).toEqual(['clip']);
  });

  it('maps the snake_case camera onto the engine ViewerConfig', async () => {
    const catalog = parseManifest(MANIFEST, fakeSource().source);
    const input = await catalog.projects[0].scenes[0].buildScene();
    expect(input.viewer).toEqual({ distance: 5, originType: 'ASSET_BODY', bodyName: 'torso' });
  });

  it('defaults the policy and omits the splat, and accepts a JSON string', async () => {
    const catalog = parseManifest(JSON.stringify(MANIFEST), fakeSource().source);
    const input = await catalog.projects[0].scenes[0].buildScene();
    expect(input.policy).not.toBeNull(); // default policy 'walk'
    expect(input.splat).toBeNull(); // no splat requested
  });

  it('gives the engine the policy entry merged with its MDP, slot tables at the top level', async () => {
    const catalog = parseManifest(MANIFEST, fakeSource().source);
    const run = await catalog.projects[0].scenes[0].policies[1].build();
    const config = run.config as Record<string, unknown>;
    expect(config.actions).toEqual({ joint_pos: { type: 'joint_position' } });
    expect(config).not.toHaveProperty('observations');
    expect(config.in_keys).toEqual(['obs']);
    expect(config.out_keys).toEqual(['action']);
    // Bookkeeping keys stay out of what the engine interprets; the network is bytes.
    expect(config).not.toHaveProperty('mdp');
    expect(config).not.toHaveProperty('id');
    expect(config).not.toHaveProperty('onnx');
    const walk = await catalog.projects[0].scenes[0].policies[0].build();
    expect((walk.config as Record<string, unknown>).policy_joint_names).toEqual(['a', 'b']);
    // No table declared: the engine assumes the single `actor` input.
    expect((walk.config as Record<string, unknown>).in_keys).toBeUndefined();
  });

  it('delivers the traced graphs of the whole MDP, keyed by their scene-relative refs', async () => {
    const { source, requested } = fakeSource();
    const catalog = parseManifest(MANIFEST, source);
    const input = await catalog.projects[0].scenes[0].buildScene({ policy: 'walk' });

    expect(Object.keys(input.policy?.graphs ?? {}).sort()).toEqual([
      'mdp/mdp_0/command/twist.onnx',
      'mdp/mdp_0/event/push_robot.onnx',
      'mdp/mdp_0/obs/joint_pos.onnx',
      'mdp/mdp_0/term/fell_over.onnx',
    ]);
    expect(requested).toContain('demo/humanoid/mdp/mdp_0/obs/joint_pos.onnx');
    expect(requested).toContain('demo/humanoid/mdp/mdp_0/event/push_robot.onnx');
    // The network's own `onnx` is a path on the policy entry, not a term reference.
    expect(input.policy?.graphs).not.toHaveProperty('policy/walk.onnx');
  });

  it('hands events to the engine inside the policy config, never on the scene', async () => {
    // Events belong to the MDP (ADR 0006 §3): switching policy switches them, so the
    // scene input carries none and the policy config carries its MDP's.
    const catalog = parseManifest(MANIFEST, fakeSource().source);
    const asWalk = await catalog.projects[0].scenes[0].buildScene({ policy: 'walk' });
    expect(asWalk).not.toHaveProperty('events');
    const walk = asWalk.policy?.config as { events?: Array<{ name: string }> };
    expect(walk.events?.map((e) => e.name)).toEqual(['push_robot', 'randomize_terrain']);
    const asRun = await catalog.projects[0].scenes[0].buildScene({ policy: 'run' });
    expect((asRun.policy?.config as { events?: unknown }).events).toBeUndefined();
  });

  it('refuses a policy whose mdp the scene does not declare', async () => {
    const broken: Manifest = JSON.parse(JSON.stringify(MANIFEST));
    broken.projects[0].scenes[0].policies[1].mdp = 'ghost';
    const catalog = parseManifest(broken, fakeSource().source);
    await expect(catalog.projects[0].scenes[0].policies[1].build()).rejects.toThrow(/ghost/);
  });

  it('exposes all projects, the default first, and throws on an empty catalog', () => {
    const multi: Manifest = {
      format: 1,
      version: '0',
      projects: [
        { id: 'extra', name: 'Extra', scenes: [] },
        { id: 'main', name: 'Main', default: true, scenes: [] },
      ],
    };
    const catalog = parseManifest(multi, fakeSource().source);
    expect(catalog.projects.map((p) => p.name)).toEqual(['Main', 'Extra']);
    expect(catalog.projects.map((p) => p.default)).toEqual([true, false]);
    expect(() => parseManifest({ format: 1, version: '0', projects: [] }, fakeSource().source)).toThrow();
  });

  it('with no default flagged, the first project in document order is the default', () => {
    const multi: Manifest = {
      format: 1,
      version: '0',
      projects: [
        { id: 'first', name: 'First', scenes: [] },
        { id: 'second', name: 'Second', scenes: [] },
      ],
    };
    const catalog = parseManifest(multi, fakeSource().source);
    expect(catalog.projects.map((p) => [p.id, p.default])).toEqual([
      ['first', true],
      ['second', false],
    ]);
  });

  it('surfaces the runtime plugin module path when present', () => {
    const withPlugins = { ...MANIFEST, plugins: 'assets/plugins.js' };
    expect(parseManifest(withPlugins, fakeSource().source).pluginsPath).toBe('assets/plugins.js');
    expect(parseManifest(MANIFEST, fakeSource().source).pluginsPath).toBeUndefined();
  });
});

describe('document format', () => {
  it('accepts the current format and every earlier one', () => {
    for (let format = 1; format <= MAX_DOCUMENT_FORMAT; format++) {
      expect(() => parseManifest({ ...MANIFEST, format }, fakeSource().source)).not.toThrow();
    }
  });

  it('refuses a document with no format as one that predates the layout', () => {
    const legacy: Partial<Manifest> = { ...MANIFEST };
    delete legacy.format;
    expect(() => parseManifest(legacy as Manifest, fakeSource().source)).toThrow(
      /predates document format 1/,
    );
  });

  it('refuses a newer format, naming both numbers and the writing version', () => {
    expect(() => parseManifest({ ...MANIFEST, format: 99, version: '9.9.9' }, fakeSource().source)).toThrow(
      new RegExp(`format 99 .*9\\.9\\.9.*up to format ${MAX_DOCUMENT_FORMAT}`),
    );
  });

  it('does not gate on version: an unknown version with a known format parses', () => {
    expect(() => parseManifest({ ...MANIFEST, format: 1, version: '99.0.0' }, fakeSource().source)).not.toThrow();
  });
});

describe('sanitizeName', () => {
  it('lowercases and underscores spaces and hyphens', () => {
    expect(sanitizeName('Foo Bar-Baz')).toBe('foo_bar_baz');
  });

  // The same table pins Python's `name2id` (tests/test_utils.py): a URL is resolved by
  // this function and a directory named by that one, so a case they disagree on is a
  // link that opens the wrong scene.
  it.each(NAME2ID_CASES as Array<[string, string]>)('mirrors name2id: %j → %j', (name, id) => {
    expect(sanitizeName(name)).toBe(id);
  });
});
