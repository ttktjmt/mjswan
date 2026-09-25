import { test, expect } from '@playwright/test';

/**
 * Runtime tier for the pointer modes: a real mouse against a real canvas.
 *
 * The unit tests drive the mechanisms with the real WASM but no browser, so everything
 * between a `pointerdown` and a body moving (the raycast, the coordinate swizzle, the
 * claim handed to `OrbitControls`, the step-loop hook) is only exercised here. Each mode
 * is asserted on the effect it is *for*, so none of the three can quietly become a no-op.
 */

interface ModeReport {
  id: string;
  available: boolean;
  reason?: string;
}

interface HarnessEngine {
  getState(): { interactions: ModeReport[]; interactionMode: string };
  interaction: { setMode(id: string): void };
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
  expect(modes.map((m) => m.id)).toEqual(['pull', 'push', 'weld']);
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

  // ── pull: a held drag has to put the block under a force ──────────────
  await arm('pull');
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 8; step++) await page.mouse.move(x + step * 6, y - step * 5);
  await page.waitForTimeout(300);
  const pulled = await page.evaluate(() => window.__probe!.read());
  await page.mouse.up();
  expect(pulled.maxForce, 'pull puts the block under a force').toBeGreaterThan(1);

  // ── push: a tap, whose shove lasts one control step ────────────────────
  await arm('push');
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.up();
  await page.waitForTimeout(300);
  expect((await page.evaluate(() => window.__probe!.read())).maxForce, 'a tap shoves').toBeGreaterThan(1);

  // ── weld: carrying a block means an active constraint ──────────────────
  await arm('weld');
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 10; step++) await page.mouse.move(x, y - step * 6);
  await page.waitForTimeout(300);
  const held = await page.evaluate(() => window.__probe!.read());
  await page.mouse.up();
  expect(held.welded, 'grabbing activates a weld').toBe(true);

  expect(errors, errors.join('\n')).toEqual([]);
});
