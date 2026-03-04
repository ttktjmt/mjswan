/**
 * Version information for mjswan
 * This version should match the version in src/mjswan/__init__.py
 */
export const MJSWAN_VERSION = __APP_VERSION__;

// Declare the global constant injected by vite
declare const __APP_VERSION__: string;

// GitHub contributors for the mjswan project.
// Run `uv run sync_contributors.py` to update.
export interface Contributor {
  login: string;
  html_url: string;
}

export const GITHUB_CONTRIBUTORS: Contributor[] = [
  {
    login: "ttktjmt",
    html_url: "https://github.com/ttktjmt",
  },
  {
    login: "claude",
    html_url: "https://github.com/claude",
  },
  {
    login: "julien-blanchon",
    html_url: "https://github.com/julien-blanchon",
  },
  {
    login: "brentyi",
    html_url: "https://github.com/brentyi",
  },
  {
    login: "Axellwppr",
    html_url: "https://github.com/Axellwppr",
  },
  {
    login: "CharlieLeee",
    html_url: "https://github.com/CharlieLeee",
  },
];
