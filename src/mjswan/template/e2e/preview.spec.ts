import { test, expect, type Page } from '@playwright/test';

/**
 * The preview's handles with a real mouse on a real canvas: the hit test through
 * the real camera, the press a handle keeps from the camera, and each kind changing only its
 * own part of the geom. The drag maths is unit-tested in Node; this is what it meets in a
 * browser.
 */

type Vec = [number, number, number];

interface Geom {
  size: number[];
  pos: number[];
  quat: number[];
}

interface HarnessEngine {
  preview: {
    show(input: object): void;
    hit(clientX: number, clientY: number): boolean;
    onChange(listener: (geom: Geom, done: boolean) => void): () => void;
  };
  camera: { get(): { azimuth: number }; set(view: object): void };
  interaction: { setMode(id: string): void };
  pause(): void;
}

const BOX = { type: 'box', size: [0.1, 0.1, 0.1], pos: [0, 0, 0.6] as Vec };

/** Where a MuJoCo point lands on the page, through the engine's own camera. */
function screenOf(page: Page, point: Vec): Promise<{ x: number; y: number }> {
  return page.evaluate(([x, y, z]) => {
    type Three = { Vector3: new (x: number, y: number, z: number) => { project(camera: unknown): { x: number; y: number } } };
    const three = (globalThis as unknown as { __mjswanThree: Three }).__mjswanThree;
    const camera = (window.__engine as unknown as { runtime: { camera: unknown } }).runtime.camera;
    const ndc = new three.Vector3(x, z, -y).project(camera);
    const rect = document.querySelector('canvas')!.getBoundingClientRect();
    return { x: rect.left + ((ndc.x + 1) / 2) * rect.width, y: rect.top + ((1 - ndc.y) / 2) * rect.height };
  }, point);
}

async function drag(page: Page, path: { x: number; y: number }[]): Promise<void> {
  await page.mouse.move(path[0].x, path[0].y);
  await page.mouse.down();
  for (let i = 1; i < path.length; i++) {
    const [a, b] = [path[i - 1], path[i]];
    for (let step = 1; step <= 4; step++) {
      await page.mouse.move(a.x + ((b.x - a.x) * step) / 4, a.y + ((b.y - a.y) * step) / 4);
    }
  }
  await page.mouse.up();
}

test('the handles move, resize and turn the preview, and keep their press from the camera', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(String(err)));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('404')) errors.push(message.text());
  });

  await page.goto('/harness-interaction.html');
  await page.waitForFunction(() => window.__ready === true, undefined, { timeout: 90_000 });

  /** Aim the camera at the box, wait for the orbit's damping, and show the preview with `handles`. */
  const stage = async (handles: string, view: object): Promise<void> => {
    await page.evaluate(
      ([box, kind, camera]) => {
        const engine = window.__engine as HarnessEngine;
        engine.camera.set({ lookat: box.pos, distance: 2.5, fovy: 45, ...camera });
        engine.preview.show({ ...box, handles: kind });
      },
      [BOX, handles, view] as const,
    );
    await page.waitForTimeout(400);
  };
  const last = () => page.evaluate(() => window.__changes!.at(-1));
  const azimuth = () => page.evaluate(() => (window.__engine as HarnessEngine).camera.get().azimuth);

  await page.evaluate(() => {
    const engine = window.__engine as HarnessEngine;
    engine.pause();
    engine.interaction.setMode('view');
    window.__changes = [];
    engine.preview.onChange((geom, done) => window.__changes!.push({ geom, done }));
  });

  // A side view a little from above, where no world axis points at the eye.
  const side = { azimuth: 30, elevation: -20 };

  // ── move: the z arrow starts at the centre; the geom follows the pointer up ──
  await stage('move', side);
  const before = await azimuth();
  await drag(page, [await screenOf(page, [0, 0, 0.75]), await screenOf(page, [0, 0, 0.95])]);
  const moved = await last();
  expect(moved?.done, 'a release reports done').toBe(true);
  // Within a centimetre: the pointer lands on whole pixels, a few millimetres apart here.
  expect(Math.abs(moved!.geom.pos[2] - 0.8), 'moved up by the drag').toBeLessThan(0.01);
  expect(Math.abs(moved!.geom.pos[0]) + Math.abs(moved!.geom.pos[1]), 'along z only').toBeLessThan(0.002);
  expect(moved!.geom.size, 'a move leaves the size').toEqual(BOX.size);
  expect(await azimuth(), 'the camera never saw the press').toBeCloseTo(before, 3);

  // ── size: the z arrow starts on the top face, and the bottom stays ──
  await stage('size', side);
  await drag(page, [await screenOf(page, [0, 0, 0.8]), await screenOf(page, [0, 0, 0.9])]);
  const sized = await last();
  expect(Math.abs(sized!.geom.size[2] - 0.15), 'the top face followed the pointer').toBeLessThan(0.01);
  expect(Math.abs(sized!.geom.pos[2] - sized!.geom.size[2] - 0.5), 'the bottom stayed').toBeLessThan(1e-6);

  // ── turn: from above, a quarter sweep anticlockwise round the z ring ──
  await stage('turn', { azimuth: 30, elevation: -89 });
  const centre = await screenOf(page, BOX.pos);
  const on = (degrees: number) => ({
    x: centre.x + 64 * Math.cos((degrees * Math.PI) / 180),
    y: centre.y - 64 * Math.sin((degrees * Math.PI) / 180),
  });
  // Clear of the x and y rings, which this view sees edge-on at 60 and 150 degrees.
  await drag(page, [on(15), on(60), on(105)]);
  const turned = await last();
  const [w, x, y, z] = turned!.geom.quat;
  const sign = Math.sign(w) || 1;
  expect([w, x, y, z].map((v) => Number((v * sign).toFixed(2)) + 0), 'a quarter turn about z').toEqual([0.71, 0, 0, 0.71]);
  expect(turned!.geom.pos, 'a turn leaves the centre').toEqual(BOX.pos);

  // ── the preview knows where it is, and a press off the handles orbits ──
  const hit = (point: { x: number; y: number }) =>
    page.evaluate(([px, py]) => (window.__engine as HarnessEngine).preview.hit(px, py), [point.x, point.y] as const);
  expect(await hit(centre), 'the box is under its centre').toBe(true);
  const away = { x: centre.x + 240, y: centre.y + 120 };
  expect(await hit(away), 'nothing of it far off').toBe(false);
  const count = await page.evaluate(() => window.__changes!.length);
  const orbit = await azimuth();
  await drag(page, [away, { x: away.x + 96, y: away.y }]);
  await page.waitForTimeout(300);
  expect(await azimuth(), 'a press off the handles orbits').not.toBeCloseTo(orbit, 1);
  expect(await page.evaluate(() => window.__changes!.length), 'and changes nothing').toBe(count);

  expect(errors, errors.join('\n')).toEqual([]);
});
