import { test, expect } from '@playwright/test';

/**
 * Runtime tier for the pointer modes: a real mouse against a real canvas.
 *
 * The unit tests drive the mechanisms with the real WASM but no browser, so everything
 * between a `pointerdown` and a body moving — the raycast, the coordinate swizzle, the
 * claim handed to `OrbitControls`, the step-loop hook — is only exercised here.
 */

interface ModeReport {
  id: string;
  available: boolean;
  reason?: string;
}

interface HarnessEngine {
  getState(): { interactions: ModeReport[]; interactionMode: string };
  interaction: { setMode(id: string): void };
}

test('every pointer mode runs, and a throw puts a box in the scene', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(String(err)));
  page.on('console', (message) => {
    // Vite's dev server 404s its own optional assets; only script errors matter here.
    if (message.type() === 'error' && !message.text().includes('404')) errors.push(message.text());
  });

  await page.goto('/harness-interaction.html');
  await page.waitForFunction(() => window.__ready === true, undefined, { timeout: 90_000 });

  const modes = await page.evaluate(() => (window.__engine as HarnessEngine).getState().interactions);
  expect(modes.map((m) => m.id)).toEqual(['pull', 'push', 'weld', 'spawn']);
  expect(modes.filter((m) => !m.available).map((m) => `${m.id}: ${m.reason}`)).toEqual([]);

  const canvas = (await page.locator('canvas').boundingBox())!;
  const x = canvas.x + canvas.width / 2;
  const y = canvas.y + canvas.height / 2;

  // Drag through the middle of the scene: whatever is under it gets pulled.
  await page.evaluate(() => (window.__engine as HarnessEngine).interaction.setMode('pull'));
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 8; step++) await page.mouse.move(x + step * 6, y - step * 5);
  await page.waitForTimeout(300);
  await page.mouse.up();

  // A tap, which is the whole of the push gesture.
  await page.evaluate(() => (window.__engine as HarnessEngine).interaction.setMode('push'));
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.up();
  await page.waitForTimeout(200);

  // Grab and carry upward, then let go.
  await page.evaluate(() => (window.__engine as HarnessEngine).interaction.setMode('weld'));
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 10; step++) await page.mouse.move(x, y - step * 6);
  await page.waitForTimeout(300);
  await page.mouse.up();
  await page.waitForTimeout(200);

  expect((await page.evaluate(() => window.__pool!())).thrown, 'nothing thrown yet').toBe(0);

  // Press a surface below the middle, drag down to load, release to fire.
  await page.evaluate(() => (window.__engine as HarnessEngine).interaction.setMode('spawn'));
  await page.mouse.move(x + 40, y + 80);
  await page.mouse.down();
  for (let step = 1; step <= 12; step++) await page.mouse.move(x + 40, y + 80 + step * 14);
  await page.mouse.up();
  await page.waitForTimeout(600);

  const pool = await page.evaluate(() => window.__pool!());
  expect(pool.thrown, `pool heights ${JSON.stringify(pool.poolZ)}`).toBe(1);
  expect(errors, errors.join('\n')).toEqual([]);
});
