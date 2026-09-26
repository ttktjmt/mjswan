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
}

interface HarnessEngine {
  getState(): XrState;
  loadScene(input: { model: ArrayBuffer; modelFormat?: 'mjz' | 'mjb' }): Promise<void>;
  xr: {
    enter(id: 'vr' | 'ar'): Promise<void>;
    exit(): Promise<void>;
    setHandTracking(enabled: boolean): void;
  };
  runtime: {
    mjModel: { nbody: number } | null;
    mjData: { time: number } | null;
    renderer: { xr: { setSession(session: unknown): Promise<void> } };
  };
}

declare global {
  interface Window {
    __xrRequests?: Array<{ mode: string; features: string[] }>;
    __handedToThree?: number;
  }
}

async function openHarness(page: Page, errors: string[]): Promise<void> {
  page.on('pageerror', (err) => errors.push(String(err)));
  await page.addInitScript(() => {
    const requests: Array<{ mode: string; features: string[] }> = [];
    window.__xrRequests = requests;
    class FakeSession extends EventTarget {
      async end(): Promise<void> {
        this.dispatchEvent(new Event('end'));
      }
    }
    const xr = new EventTarget() as EventTarget & Record<string, unknown>;
    xr.isSessionSupported = async (mode: string) => mode === 'immersive-vr' || mode === 'immersive-ar';
    xr.requestSession = async (mode: string, init: { optionalFeatures?: string[] }) => {
      requests.push({ mode, features: [...(init.optionalFeatures ?? [])] });
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
