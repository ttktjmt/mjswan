import { describe, it, expect } from 'vitest';
import { MAX_DOCUMENT_FORMAT, parseManifest, sanitizeName, type Manifest, type ByteSource } from './index';
import NAME2ID_CASES from './name2id_cases.json';
import { DEFAULT_GRAPHICS } from '../core/scene/graphicsSettings';

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
          skins: [
            {
              id: 'sample_a',
              name: 'Sample A',
              format: 'vrm',
              path: 'assets/sample_a.vrm',
              bones: { hips: 'pelvis' },
              hide_robot: false,
            },
            { id: 'fox', name: 'Fox', format: 'gltf', path: 'assets/fox.glb', bones: { hips: 'trunk' }, nodes: { hips: 4 } },
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
              motions: [{ name: 'clip', path: 'assets/walk_clip.npz', fps: 50, default: true, group: 'Walk' }],
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
    expect(scene.policies[0].motions).toEqual([{ name: 'clip', default: true, group: 'Walk' }]);
    expect(scene.splats[0]).toMatchObject({ id: 'room', name: 'Room', control: true, transform: { scale: 2, xOffset: 0.5 } });
    expect(scene.skins.map((s) => [s.id, s.name])).toEqual([
      ['sample_a', 'Sample A'],
      ['fox', 'Fox'],
    ]);
  });

  it('builds the skin asked for, by id, and none when none is', async () => {
    const { source, requested } = fakeSource();
    const scene = parseManifest(MANIFEST, source).projects[0].scenes[0];
    expect((await scene.buildScene()).skin).toBeNull();
    expect(requested.some((r) => r.endsWith('.vrm') || r.endsWith('.glb'))).toBe(false);

    const input = await scene.buildScene({ skin: 'fox' });
    expect(input.skin?.config).toMatchObject({ format: 'gltf', bones: { hips: 'trunk' }, nodes: { hips: 4 } });
    expect(input.skin?.config.hide_robot).toBeUndefined();
    expect(requested).toContain('demo/humanoid/assets/fox.glb');

    const picked = await scene.skins[0].build();
    expect(picked.config.hide_robot).toBe(false);
    expect(requested).toContain('demo/humanoid/assets/sample_a.vrm');
  });

  it('lists the credits and fetches a text only when asked, scene and work paths resolved apart', async () => {
    const { source, requested } = fakeSource();
    const [project] = MANIFEST.projects;
    const manifest: Manifest = {
      ...MANIFEST,
      projects: [
        {
          ...project,
          work: { license: 'LICENSE', spdx: 'Apache-2.0', copyright: 'Copyright (c) Demo' },
          scenes: [
            {
              ...project.scenes[0],
              credits: [
                { component: 'unitree_g1', license: '../licenses/LICENSE.unitree_g1', spdx: 'BSD-3-Clause' },
                { component: 'skin-fox', skin: 'fox', license: 'LICENSE.skin-fox', notice: 'NOTICE.skin-fox' },
                {
                  component: 'skin-sample_a',
                  skin: 'sample_a',
                  vrm: { authors: ['mjswan'], license: 'CC0', other_license_url: 'https://example.com' },
                },
              ],
            },
          ],
        },
      ],
    };
    const catalog = parseManifest(manifest, source);
    const [scene] = catalog.projects[0].scenes;
    expect(requested).toEqual([]);
    expect(scene.credits.map((c) => [c.component, c.skin, c.spdx])).toEqual([
      ['unitree_g1', undefined, 'BSD-3-Clause'],
      ['skin-fox', 'fox', undefined],
      ['skin-sample_a', 'sample_a', undefined],
    ]);
    expect(scene.credits[0].notice).toBeUndefined();
    expect(scene.credits[2].vrm).toEqual({
      name: undefined,
      authors: ['mjswan'],
      license: 'CC0',
      otherLicenseUrl: 'https://example.com',
    });
    expect(await scene.credits[1].notice?.()).toBe('bytes:demo/humanoid/NOTICE.skin-fox');
    // The project's licenses/, as mjswan writes them now; a scene's own, as it did before.
    expect(await scene.credits[0].license?.()).toBe('bytes:demo/licenses/LICENSE.unitree_g1');
    expect(await catalog.projects[0].work?.license?.()).toBe('bytes:demo/LICENSE');
    expect(catalog.projects[0].work?.copyright).toBe('Copyright (c) Demo');
  });

  it('has no credits and no work for a build without license files', () => {
    const catalog = parseManifest(MANIFEST, fakeSource().source);
    expect(catalog.projects[0].scenes[0].credits).toEqual([]);
    expect(catalog.projects[0].work).toBeUndefined();
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

  it('draws a scene as its graphics say, and one without as MuJoCo would by default', async () => {
    const graphics = {
      color_management: 'filament',
      tone_mapping: 'pbr_neutral',
      exposure: 0.5,
      output_color_space: 'srgb',
      base_color_type: 'linear',
      enable_shadows: false,
      enable_reflections: true,
      enable_haze: true,
    } as const;
    const withGraphics = { ...MANIFEST.projects[0].scenes[0], graphics };
    const manifest = { ...MANIFEST, format: 9, projects: [{ ...MANIFEST.projects[0], scenes: [withGraphics] }] };
    const [scene] = parseManifest(manifest, fakeSource().source).projects[0].scenes;
    // MuJoCo's name for three.js's operator is the operator.
    const expected = { colorManagement: 'filament', toneMapping: 'neutral', exposure: 0.5, outputColorSpace: 'srgb', baseColorType: 'linear', shadows: false, reflections: true, haze: true };
    expect(scene.graphics).toEqual(expected);
    expect((await scene.buildScene()).graphics).toEqual(expected);

    // Format 8 and below carry the switch in the camera, unread.
    const older = { ...MANIFEST.projects[0].scenes[0], camera: { enable_shadows: false } as never };
    const [old] = parseManifest({ ...MANIFEST, projects: [{ ...MANIFEST.projects[0], scenes: [older] }] }, fakeSource().source)
      .projects[0].scenes;
    expect(old.graphics).toEqual(DEFAULT_GRAPHICS);
    expect(old.camera).toEqual({});
  });

  it("hands a scene's term_defaults to the engine as overrides, and none without them", async () => {
    const term_defaults = {
      terminations: { bad_orientation: { enabled: true, on_fire: { delay: 0.2 } }, illegal_contact: { enabled: true, on_fire: 'freeze' } },
      events: { push_robot: { enabled: false } },
    };
    const withDefaults = { ...MANIFEST.projects[0].scenes[0], term_defaults };
    const manifest = { ...MANIFEST, format: 10, projects: [{ ...MANIFEST.projects[0], scenes: [withDefaults] }] };
    const [scene] = parseManifest(manifest, fakeSource().source).projects[0].scenes;
    expect((await scene.buildScene()).termDefaults).toEqual({
      terminations: { bad_orientation: { enabled: true, onFire: { delay: 0.2 } }, illegal_contact: { enabled: true, onFire: 'freeze' } },
      events: { push_robot: { enabled: false } },
    });
    const [plain] = parseManifest(MANIFEST, fakeSource().source).projects[0].scenes;
    expect((await plain.buildScene()).termDefaults).toBeUndefined();
  });

  it('turns an azimuth from before format 3, which pointed at the camera, to keep its view', () => {
    const azimuthIn = (format: number) => {
      const scene = { ...MANIFEST.projects[0].scenes[0], camera: { azimuth: 90 } };
      const manifest: Manifest = { ...MANIFEST, format, projects: [{ ...MANIFEST.projects[0], scenes: [scene] }] };
      return parseManifest(manifest, fakeSource().source).projects[0].scenes[0].camera?.azimuth;
    };
    expect(azimuthIn(2)).toBe(270);
    expect(azimuthIn(3)).toBe(90);
  });

  it('tells the engine the model format the build named the scene file for', async () => {
    const withScene = (scene: string): Manifest => ({
      ...MANIFEST,
      projects: [{ ...MANIFEST.projects[0], scenes: [{ ...MANIFEST.projects[0].scenes[0], scene }] }],
    });
    const formatOf = async (scene: string) =>
      (await parseManifest(withScene(scene), fakeSource().source).projects[0].scenes[0].buildScene())
        .modelFormat;
    expect(await formatOf('scene.mjz')).toBe('mjz');
    expect(await formatOf('scene.mjb')).toBe('mjb');
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
    // Events belong to the MDP: switching policy switches them, so the scene input
    // carries none and the policy config carries its MDP's.
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
      version: '0.0.0',
      projects: [
        { id: 'extra', name: 'Extra', scenes: [] },
        { id: 'main', name: 'Main', default: true, scenes: [] },
      ],
    };
    const catalog = parseManifest(multi, fakeSource().source);
    expect(catalog.projects.map((p) => p.name)).toEqual(['Main', 'Extra']);
    expect(catalog.projects.map((p) => p.default)).toEqual([true, false]);
    expect(() => parseManifest({ format: 1, version: '0.0.0', projects: [] }, fakeSource().source)).toThrow(
      /no projects/,
    );
  });

  it('with no default flagged, the first project in document order is the default', () => {
    const multi: Manifest = {
      format: 1,
      version: '0.0.0',
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

  it("surfaces a custom-TS build's plugin module path, and no other", () => {
    const withPlugins = { ...MANIFEST, uses_custom_js: true, plugins: 'assets/plugins.js' };
    expect(parseManifest(withPlugins, fakeSource().source).pluginsPath).toBe('assets/plugins.js');
    expect(parseManifest(MANIFEST, fakeSource().source).pluginsPath).toBeUndefined();
    // A build writes `plugins` only beside `uses_custom_js: true`.
    for (const flag of [false, undefined]) {
      const named = { ...MANIFEST, uses_custom_js: flag, plugins: 'assets/plugins.js' };
      expect(parseManifest(named, fakeSource().source).pluginsPath).toBeUndefined();
    }
  });
});

/** MANIFEST with one change made to a copy of it. */
function edited(edit: (m: Manifest) => void): Manifest {
  const copy: Manifest = JSON.parse(JSON.stringify(MANIFEST));
  edit(copy);
  return copy;
}

const scene0 = (m: Manifest) => m.projects[0].scenes[0];

/** Paths that a URL parser would take out of the document, or that are not paths at all. */
const OUTSIDE = [
  '/x',
  '//evil.test/x',
  '\\evil.test/x',
  '%2e%2e/x',
  'a\\..\\..\\x',
  '.\t./x',
  'x?y',
  'x#y',
  'https://evil.test/x',
  '../../../x',
  'a//b',
  './x',
  '',
];

/** Each field naming a document file: how deep below the root it resolves, and how to set it. */
const PATH_FIELDS: Array<[string, number, (m: Manifest, path: string) => void]> = [
  ['scene', 2, (m, path) => (scene0(m).scene = path)],
  ['policy onnx', 2, (m, path) => (scene0(m).policies[0].onnx = path)],
  ['motion', 2, (m, path) => (scene0(m).policies[0].motions![0].path = path)],
  ['splat', 2, (m, path) => (scene0(m).splats![0].path = path)],
  ['collider', 2, (m, path) => (scene0(m).splats![0].collider_url = path)],
  ['skin', 2, (m, path) => (scene0(m).skins![0].path = path)],
  ['credit license', 2, (m, path) => (scene0(m).credits = [{ component: 'c', license: path }])],
  ['credit notice', 2, (m, path) => (scene0(m).credits = [{ component: 'c', notice: path }])],
  ['work', 1, (m, path) => (m.projects[0].work = { license: path })],
  ['plugins', 0, (m, path) => Object.assign(m, { uses_custom_js: true, plugins: path })],
  [
    'term graph',
    2,
    (m, path) =>
      ((scene0(m).mdps[0].observations as { actor: Array<{ onnx?: string }> }).actor[0].onnx = path),
  ],
  ['fused graph', 2, (m, path) => (scene0(m).mdps[0].observations = { actor: { fused: path } as never })],
  ['action graph', 2, (m, path) => (scene0(m).mdps[0].actions = { joint_pos: { process: { onnx: path } } })],
];

describe('what a document may name', () => {
  it.each(['\\evil.test', 'http:evil.test', '%2e%2e', '..', '.', 'a/b', '', 'a b'])(
    'refuses a project or scene id that is not a name a build writes (%j)',
    (id) => {
      for (const set of [(m: Manifest) => (m.projects[0].id = id), (m: Manifest) => (scene0(m).id = id)]) {
        const { source, requested } = fakeSource();
        expect(() => parseManifest(edited(set), source)).toThrow(/is not a name a build writes/);
        expect(requested).toEqual([]);
      }
    },
  );

  describe.each(PATH_FIELDS)('the %s path', (field, depth, set) => {
    // A collider may also be a URL, fetched through `remote` alone (below).
    it.each(OUTSIDE.filter((path) => !(field === 'collider' && path.startsWith('https:'))))(
      'refuses %j before anything is fetched',
      (path) => {
        const { source, requested } = fakeSource();
        expect(() => parseManifest(edited((m) => set(m, path)), source)).toThrow(/mjswan\/manifest/);
        expect(requested).toEqual([]);
      },
    );

    it('takes a path inside the document, up to its root and no further', () => {
      const parse = (path: string) => () => parseManifest(edited((m) => set(m, path)), fakeSource().source);
      expect(parse('assets/x-1.y_z')).not.toThrow();
      expect(parse(`${'../'.repeat(depth)}x`)).not.toThrow();
      expect(parse(`${'../'.repeat(depth + 1)}x`)).toThrow(/climbs out of the document/);
    });
  });

  it('asks the source for every file by its path in the document', async () => {
    const { source, requested } = fakeSource();
    const manifest = edited((m) => (scene0(m).splats![0].collider_url = '../shared/room.glb'));
    const scene = parseManifest(manifest, source).projects[0].scenes[0];
    await scene.buildScene({ policy: 'walk', splat: 'room', skin: 'fox' });
    expect(requested).toEqual([
      'demo/humanoid/scene.mjz',
      'demo/humanoid/assets/walk_clip.npz',
      'demo/humanoid/policy/walk.onnx',
      'demo/humanoid/mdp/mdp_0/command/twist.onnx',
      'demo/humanoid/mdp/mdp_0/event/push_robot.onnx',
      'demo/humanoid/mdp/mdp_0/obs/joint_pos.onnx',
      'demo/humanoid/mdp/mdp_0/term/fell_over.onnx',
      'demo/humanoid/assets/room.spz',
      'demo/shared/room.glb',
      'demo/humanoid/assets/fox.glb',
    ]);
  });
});

describe('a URL the document names outside itself', () => {
  const urlSplat = (m: Manifest) =>
    (scene0(m).splats = [
      { id: 'room', name: 'Room', path: 'assets/room.spz', collider_url: 'https://cdn.test/room.glb' },
      { id: 'park', name: 'Park', url: 'https://cdn.test/park.spz' },
    ]);

  it('is fetched by no one without remote: its splat is left out, its collider dropped', async () => {
    const { source, requested } = fakeSource();
    const scene = parseManifest(edited(urlSplat), source).projects[0].scenes[0];
    expect(scene.splats.map((s) => s.id)).toEqual(['room']);
    const splat = await scene.splats[0].build();
    expect(splat.collider).toBeUndefined();
    expect((await scene.buildScene({ splat: 'park' })).splat).toBeNull();
    expect(requested.every((path) => !path.includes('cdn.test'))).toBe(true);
  });

  it('is handed to remote as a URL, when the host passes one', async () => {
    const asked: string[] = [];
    const remote = (url: URL) => {
      asked.push(url.href);
      return async () => new ArrayBuffer(0);
    };
    const scene = parseManifest(edited(urlSplat), fakeSource().source, { remote }).projects[0].scenes[0];
    expect(scene.splats.map((s) => s.id)).toEqual(['room', 'park']);
    await scene.splats[0].build();
    await scene.splats[1].build();
    expect(asked).toEqual(['https://cdn.test/room.glb', 'https://cdn.test/park.spz']);
  });

  it.each(['javascript:alert(1)', 'data:text/plain,x', 'file:///etc/passwd', 'blob:https://a.test/x', 'cdn.test/x'])(
    'must be http(s), with or without remote (%s)',
    (url) => {
      const remote = () => async () => new ArrayBuffer(0);
      for (const options of [{}, { remote }]) {
        const splat = edited((m) => (scene0(m).splats = [{ id: 'park', name: 'Park', url }]));
        expect(() => parseManifest(splat, fakeSource().source, options)).toThrow(/not an http\(s\) URL/);
        // A collider without a scheme is a path in the document instead.
        const collider = edited((m) => (scene0(m).splats![0].collider_url = url));
        const parse = () => parseManifest(collider, fakeSource().source, options);
        if (url.includes(':')) expect(parse).toThrow(/not an http\(s\) URL/);
        else expect(parse).not.toThrow();
      }
    },
  );
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

  // A host picks an engine by `version`, and the newer-format error quotes it.
  it.each(['0', '../../evil/x', 'latest', '1.2.3/../../x', 'v1.2.3', undefined, 13])(
    'refuses a version no release has (%j)',
    (version) => {
      const manifest = { ...MANIFEST, format: 99, version } as unknown as Manifest;
      expect(() => parseManifest(manifest, fakeSource().source)).toThrow(/not one a release has/);
    },
  );
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
