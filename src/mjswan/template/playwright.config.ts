import { defineConfig, devices } from '@playwright/test';

// Runtime-tier E2E: drives the React-free engine harness in a real browser.
// It drives the installed Google Chrome, not a build Playwright downloads. Headless
// Chrome renders WebGL via SwiftShader (software GL), so no GPU is required in CI. Unit tests (pure logic) live under vitest; run: npm run test:e2e.
const PORT = 5178;

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  fullyParallel: false,
  webServer: {
    command: `npm run dev -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/harness.html`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  use: {
    baseURL: `http://localhost:${PORT}`,
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        launchOptions: {
          // Playwright turns Chrome's own sandbox off unless asked.
          chromiumSandbox: true,
          args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
        },
      },
    },
  ],
});
