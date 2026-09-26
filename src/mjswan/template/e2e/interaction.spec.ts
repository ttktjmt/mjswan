import { test, expect } from '@playwright/test';

/**
 * The pointer modes with a real mouse on a real canvas: the raycast, the coordinate
 * swizzle, the `OrbitControls` claim and the step-loop hook, which the WASM unit tests
 * never reach. Each mode is asserted on its effect, so none can quietly become a no-op.
 */

interface ModeReport {
  id: string;
  available: boolean;
  reason?: string;
}

interface HarnessEngine {
  getState(): { interactions: ModeReport[]; interactionMode: string };
  interaction: { setMode(id: string): void };
  camera: { get(): { azimuth: number; elevation: number } };
  reset(): void;
}

test('every pointer mode does its own job', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(String(err)));
  page.on('console', (message) => {
    // Vite's dev server 404s its own optional assets; only script errors matter here.
    if (message.type() === 'error' && !message.text().includes('404')) errors.push(message.text());
  });

  await page.goto('/harness-interaction.html');
  await page.waitForFunction(() => window.__ready === true, undefined, { timeout: 90_000 });

  const modes = await page.evaluate(() => (window.__engine as HarnessEngine).getState().interactions);
  expect(modes.map((m) => m.id)).toEqual(['view', 'pull', 'push', 'weld']);
  expect(modes.filter((m) => !m.available).map((m) => `${m.id}: ${m.reason}`)).toEqual([]);

  const canvas = (await page.locator('canvas').boundingBox())!;
  // The fixture's first block sits at the origin, which the fallback view centres.
  const x = canvas.x + canvas.width / 2;
  const y = canvas.y + canvas.height / 2;

  /** Put the blocks back, so every gesture below aims at the same one. */
  const arm = async (mode: string): Promise<void> => {
    await page.evaluate((next) => {
      const engine = window.__engine as HarnessEngine;
      engine.reset();
      engine.interaction.setMode(next);
    }, mode);
    await page.waitForTimeout(250);
    await page.evaluate(() => window.__probe!.reset());
  };

  // ── view: a drag on the block orbits and touches nothing ──────────────
  await arm('view');
  const view = () => page.evaluate(() => (window.__engine as HarnessEngine).camera.get());
  const before = await view();
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 8; step++) await page.mouse.move(x + step * 12, y);
  await page.mouse.up();
  await page.waitForTimeout(300);
  const viewed = await page.evaluate(() => window.__probe!.read());
  expect((await view()).azimuth, 'view hands the press to the camera').not.toBeCloseTo(before.azimuth, 1);
  expect(viewed.peakForce, 'view puts nothing on the block').toBe(0);
  expect(viewed.welded, 'view holds nothing').toBe(false);

  // ── pull: a held drag has to put the block under a force ──────────────
  await arm('pull');
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 8; step++) await page.mouse.move(x + step * 6, y - step * 5);
  await page.waitForTimeout(300);
  const pulled = await page.evaluate(() => window.__probe!.read());
  await page.mouse.up();
  expect(pulled.peakForce, 'pull puts the block under a force').toBeGreaterThan(1);

  // ── push: a tap, whose shove lasts one control step ────────────────────
  await arm('push');
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.up();
  await page.waitForTimeout(300);
  expect((await page.evaluate(() => window.__probe!.read())).peakForce, 'a tap shoves').toBeGreaterThan(1);

  // ── weld: carrying a block means an active constraint ──────────────────
  await arm('weld');
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 10; step++) await page.mouse.move(x, y - step * 6);
  await page.waitForTimeout(300);
  const held = await page.evaluate(() => window.__probe!.read());
  const cursor = () => page.evaluate(() => document.querySelector('canvas')!.style.cursor);
  expect(await cursor(), 'a held body shows the grabbing hand').toBe('grabbing');
  await page.mouse.up();
  expect(held.welded, 'grabbing activates a weld').toBe(true);
  expect(await cursor(), 'letting go hands the cursor back').toBe('');

  expect(errors, errors.join('\n')).toEqual([]);
});
