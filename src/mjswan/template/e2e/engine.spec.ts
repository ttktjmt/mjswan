import { test, expect, type Page } from '@playwright/test';

async function runHarness(page: Page, query = ''): Promise<{ result: Window['__harness']; pageErrors: string[] }> {
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));

  await page.goto(`/harness.html${query}`);
  await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 60_000 });
  return { result: await page.evaluate(() => window.__harness), pageErrors };
}

// Runtime-tier acceptance: a real browser loads the React-free harness, `createEngine`
// builds a scene from `.mjz` bytes, and the captured frame must be a non-blank render.
test('createEngine renders a scene from bytes, React-free', async ({ page }) => {
  const { result, pageErrors } = await runHarness(page);
  expect(result?.ok, result?.error).toBe(true);
  expect(result?.running).toBe(true);
  expect(result?.nonBlank, `luminance range ${JSON.stringify(result?.luminanceRange)}`).toBe(true);
  // An app has to be able to choose the term seed and read back the one in use, or a
  // recorded session has nothing to replay from. The harness's 0xc0ffee differs from
  // the built-in default, so a dropped option shows up rather than falling back.
  expect(result?.termSeed).toBe(0xc0ffee);
  expect(pageErrors).toEqual([]);
});

// `container.mjb` is `container.mjz` compiled and saved by the Python mujoco the build
// pins, its builtin textures shrunk to keep it small. An `.mjb` loads only in the MuJoCo
// version that saved it, so it has to be written again when that pin moves.
test('createEngine renders the .mjb that add_scene(model=...) writes', async ({ page }) => {
  const { result, pageErrors } = await runHarness(page, '?scene=/fixtures/container.mjb&format=mjb');
  expect(result?.ok, result?.error).toBe(true);
  expect(result?.running).toBe(true);
  expect(result?.nonBlank, `luminance range ${JSON.stringify(result?.luminanceRange)}`).toBe(true);
  expect(pageErrors).toEqual([]);
});
