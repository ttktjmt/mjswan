import { test, expect, type Page } from '@playwright/test';

/**
 * No headset: `navigator.xr` is a stand-in that supports both sessions and records each
 * request, and three's `setSession` is stubbed, since no fake session satisfies it.
 */

/** Two hands of 7 grip bones (a mocap target and its welded body) and 9 wall bones. */
const HAND_BODIES = 2 * (7 * 2 + 9);

interface XrState {
  xrSessions: Array<{ id: string; label: string; active: boolean }>;
  handTracking: { available: boolean; reason?: string; enabled: boolean } | null;
  loading: boolean;
  error: Error | null;
}

type Fn = (...args: unknown[]) => Promise<unknown>;

interface HarnessEngine {
  getState(): XrState;
  loadScene(input: {
    model: ArrayBuffer;
    modelFormat?: 'mjz' | 'mjb';
    policy?: { config: Record<string, unknown>; onnx: ArrayBuffer };
  }): Promise<void>;
  setPolicy(input: null): Promise<void>;
  dispose(): void;
  xr: {
    enter(id: 'vr' | 'ar'): Promise<void>;
    exit(): Promise<void>;
    setHandTracking(enabled: boolean): void;
  };
  // Past the public API: what a test has to see is the model, and the steps that swap it.
  runtime: {
    mjModel: { nbody: number } | null;
    mjData: { time: number } | null;
    running: boolean;
    renderer: { xr: { setSession(session: unknown): Promise<void> } };
    handMocap: { bind(...args: unknown[]): void };
    policyGraphs: { clear(): Promise<void> };
    loadPolicyConfig: Fn;
    rebuildModel: Fn;
  };
}

declare global {
  interface Window {
    __xrRequests?: Array<{ mode: string; features: string[] }>;
    __handedToThree?: number;
    /** How long the stand-in's permission prompt stays up. */
    __xrDelay?: number;
    __xrEnded?: number;
    __xrMaxChecks?: number;
  }
}

async function openHarness(page: Page, errors: string[]): Promise<void> {
  page.on('pageerror', (err) => errors.push(String(err)));
  await page.addInitScript(() => {
    const requests: Array<{ mode: string; features: string[] }> = [];
    window.__xrRequests = requests;
    window.__xrEnded = 0;
    window.__xrMaxChecks = 0;
    class FakeSession extends EventTarget {
      private ended = false;
      async end(): Promise<void> {
        // As a real session does: a second end is refused.
        if (this.ended) throw new DOMException('The session has already ended.', 'InvalidStateError');
        this.ended = true;
        window.__xrEnded = (window.__xrEnded ?? 0) + 1;
        this.dispatchEvent(new Event('end'));
      }
    }
    let checks = 0;
    const xr = new EventTarget() as EventTarget & Record<string, unknown>;
    xr.isSessionSupported = async (mode: string) => {
      checks += 1;
      window.__xrMaxChecks = Math.max(window.__xrMaxChecks ?? 0, checks);
      await new Promise((resolve) => setTimeout(resolve, 20));
      checks -= 1;
      return mode === 'immersive-vr' || mode === 'immersive-ar';
    };
    xr.requestSession = async (mode: string, init: { optionalFeatures?: string[] }) => {
      requests.push({ mode, features: [...(init.optionalFeatures ?? [])] });
      const delay = window.__xrDelay ?? 0;
      if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
      return new FakeSession();
    };
    Object.defineProperty(navigator, 'xr', { value: xr, configurable: true });
  });
  await page.goto('/harness-interaction.html');
  await page.waitForFunction(() => window.__ready === true, undefined, { timeout: 90_000 });
  await page.evaluate(() => {
    const engine = window.__engine as HarnessEngine;
    window.__handedToThree = 0;
    engine.runtime.renderer.xr.setSession = async () => {
      window.__handedToThree = (window.__handedToThree ?? 0) + 1;
    };
  });
}

const state = (page: Page) => page.evaluate(() => (window.__engine as HarnessEngine).getState());
const bodies = (page: Page) =>
  page.evaluate(() => (window.__engine as HarnessEngine).runtime.mjModel?.nbody ?? -1);

test('the host enters XR through the API, and the model takes the hands on entry', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  // The engine draws no XR buttons of its own.
  await expect(page.locator('button')).toHaveCount(0);

  const initial = await state(page);
  expect(initial.xrSessions).toEqual([
    { id: 'vr', label: 'Enter VR', active: false },
    { id: 'ar', label: 'Start AR', active: false },
  ]);
  expect(initial.handTracking).toEqual({ available: true, enabled: false });

  const base = await bodies(page);

  // The switch alone never rebuilds.
  await page.evaluate(() => (window.__engine as HarnessEngine).xr.setHandTracking(true));
  expect((await state(page)).handTracking?.enabled).toBe(true);
  expect(await bodies(page)).toBe(base);

  await page.evaluate(() => (window.__engine as HarnessEngine).xr.enter('vr'));
  expect(await page.evaluate(() => window.__xrRequests)).toEqual([
    { mode: 'immersive-vr', features: ['local-floor', 'bounded-floor', 'layers', 'hand-tracking'] },
  ]);
  expect(await bodies(page)).toBe(base + HAND_BODIES);
  expect(await page.evaluate(() => window.__handedToThree)).toBe(1);
  expect((await state(page)).xrSessions[0]).toEqual({ id: 'vr', label: 'Exit VR', active: true });

  // The rebuilt model runs.
  const t0 = await page.evaluate(() => (window.__engine as HarnessEngine).runtime.mjData!.time);
  await page.waitForTimeout(300);
  const t1 = await page.evaluate(() => (window.__engine as HarnessEngine).runtime.mjData!.time);
  expect(t1).toBeGreaterThan(t0);

  await page.evaluate(() => (window.__engine as HarnessEngine).xr.exit());
  expect((await state(page)).xrSessions.every((s) => !s.active)).toBe(true);

  // Switched off, the next entry drops the hands again, and asks for none.
  await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    engine.xr.setHandTracking(false);
    await engine.xr.enter('ar');
  });
  expect((await page.evaluate(() => window.__xrRequests))![1]).toEqual({
    mode: 'immersive-ar',
    features: ['local-floor'],
  });
  expect(await bodies(page)).toBe(base);
  await page.evaluate(() => (window.__engine as HarnessEngine).xr.exit());

  // A scene load with the switch on builds the hands in, so entering needs no rebuild.
  const loaded = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    engine.xr.setHandTracking(true);
    const model = await (await fetch('/fixtures/blocks.mjz')).arrayBuffer();
    await engine.loadScene({ model });
    return engine.runtime.mjModel!.nbody;
  });
  expect(loaded).toBe(base + HAND_BODIES);

  expect(errors).toEqual([]);
});

test('a compiled scene cannot take the hands, and says why', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  const report = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    engine.xr.setHandTracking(true);
    const model = await (await fetch('/fixtures/container.mjb')).arrayBuffer();
    await engine.loadScene({ model, modelFormat: 'mjb' });
    const before = engine.runtime.mjModel!.nbody;
    await engine.xr.enter('vr');
    return { handTracking: engine.getState().handTracking, before, after: engine.runtime.mjModel!.nbody };
  });
  expect(report.handTracking?.available).toBe(false);
  expect(report.handTracking?.reason).toMatch(/compiled/);
  expect(report.after).toBe(report.before);
  expect((await page.evaluate(() => window.__xrRequests))![0].features).not.toContain('hand-tracking');

  expect(errors).toEqual([]);
});

test('a rebuild that fails on entry is reported, and the next load is not held by it', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  const result = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    const mocap = engine.runtime.handMocap;
    const bind = mocap.bind.bind(mocap);
    let fail = true;
    // Fails inside the model build, where a compile or out-of-memory error would.
    mocap.bind = (...args: unknown[]) => {
      if (fail) {
        fail = false;
        throw new Error('bind failed');
      }
      bind(...args);
    };
    const base = engine.runtime.mjModel!.nbody;
    engine.xr.setHandTracking(true);
    const entry = await engine.xr.enter('vr').then(() => 'entered', (e: Error) => e.message);
    const afterEntry = engine.getState();
    const left = engine.runtime.mjModel?.nbody ?? null;

    // With no model left behind, entering again retries the rebuild.
    const retry = await engine.xr.enter('vr').then(() => 'entered', (e: Error) => e.message);
    const retried = engine.runtime.mjModel!.nbody - base;
    await engine.xr.exit();

    engine.xr.setHandTracking(false);
    const model = await (await fetch('/fixtures/blocks.mjz')).arrayBuffer();
    const reload = await engine.loadScene({ model }).then(() => 'loaded', (e: Error) => e.message);
    return {
      entry,
      error: afterEntry.error?.message ?? null,
      loading: afterEntry.loading,
      left,
      retry,
      retried,
      ended: window.__xrEnded,
      reload,
      errorAfter: engine.getState().error?.message ?? null,
    };
  });
  expect(result).toEqual({
    entry: 'bind failed',
    error: 'bind failed',
    loading: false,
    left: null,
    retry: 'entered',
    retried: HAND_BODIES,
    ended: 2,
    reload: 'loaded',
    errorAfter: null,
  });

  expect(errors).toEqual([]);
});

test('the rebuild on entry waits for a policy switch already running', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  const log = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    const runtime = engine.runtime;
    const log: string[] = [];
    const loadPolicy = runtime.loadPolicyConfig.bind(runtime);
    runtime.loadPolicyConfig = async (...args: unknown[]) => {
      const model = runtime.mjModel;
      log.push('policy:start');
      await new Promise((resolve) => setTimeout(resolve, 250));
      await loadPolicy(...args);
      log.push(runtime.mjModel === model ? 'policy:end' : 'policy:end on another model');
    };
    const rebuild = runtime.rebuildModel.bind(runtime);
    runtime.rebuildModel = async (...args: unknown[]) => {
      log.push('rebuild:start');
      await rebuild(...args);
      log.push('rebuild:end');
    };
    engine.xr.setHandTracking(true);
    await Promise.all([engine.setPolicy(null), engine.xr.enter('vr')]);
    return log;
  });
  // The rebuild's own policy reload sits inside it.
  expect(log).toEqual(['policy:start', 'policy:end', 'rebuild:start', 'policy:start', 'policy:end', 'rebuild:end']);

  expect(errors).toEqual([]);
});

test('a scene that came with a policy takes no hands, even one that arrives during the prompt', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  const result = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    engine.xr.setHandTracking(true);
    window.__xrDelay = 300;
    const entry = engine.xr.enter('vr');
    // A policy scene whose policy fails to load, which leaves it as a cleared one would:
    // built for a policy, with none running.
    const model = await (await fetch('/fixtures/blocks.mjz')).arrayBuffer();
    const load = await engine
      .loadScene({ model, policy: { config: {}, onnx: new ArrayBuffer(0) } })
      .then(() => 'loaded', (e: Error) => e.message);
    const loaded = engine.runtime.mjModel!.nbody;
    await entry;
    await engine.setPolicy(null);
    return { load, loaded, after: engine.runtime.mjModel!.nbody, handTracking: engine.getState().handTracking };
  });
  expect(result.load).toMatch(/policy_joint_names/);
  expect(result.after).toBe(result.loaded);
  expect(result.handTracking).toEqual({
    available: false,
    reason: "Hands would change what this scene's policy reads.",
    enabled: true,
  });

  expect(errors).toEqual([]);
});

test('a session granted after dispose is ended, not handed to a disposed renderer', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  const result = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    window.__xrDelay = 200;
    const entry = engine.xr.enter('vr').catch(() => undefined);
    engine.dispose();
    await entry;
    await new Promise((resolve) => setTimeout(resolve, 300));
    return { ended: window.__xrEnded, handed: window.__handedToThree };
  });
  expect(result).toEqual({ ended: 1, handed: 0 });

  expect(errors).toEqual([]);
});

test('disposing during the rebuild on entry leaves nothing running', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  const result = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    const runtime = engine.runtime;
    const rebuild = runtime.rebuildModel.bind(runtime);
    runtime.rebuildModel = (...args: unknown[]) => {
      const run = rebuild(...args);
      engine.dispose();
      return run;
    };
    engine.xr.setHandTracking(true);
    await engine.xr.enter('vr').catch(() => undefined);
    await new Promise((resolve) => setTimeout(resolve, 500));
    return { running: runtime.running, ended: window.__xrEnded };
  });
  expect(result).toEqual({ running: false, ended: 1 });

  expect(errors).toEqual([]);
});

test('the rebuild on entry keeps the policy sessions, and support is asked for both modes at once', async ({ page }) => {
  const errors: string[] = [];
  await openHarness(page, errors);

  const result = await page.evaluate(async () => {
    const engine = window.__engine as HarnessEngine;
    const graphs = engine.runtime.policyGraphs;
    const clear = graphs.clear.bind(graphs);
    let cleared = 0;
    graphs.clear = async () => {
      cleared += 1;
      await clear();
    };
    engine.xr.setHandTracking(true);
    await engine.xr.enter('vr');
    return { cleared, maxChecks: window.__xrMaxChecks };
  });
  expect(result).toEqual({ cleared: 0, maxChecks: 2 });

  expect(errors).toEqual([]);
});
