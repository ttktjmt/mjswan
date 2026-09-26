/**
 * Adding the viewer's own sections to a scene's MJCF text before it compiles.
 *
 * MJCF merges repeated top-level sections, so a `<worldbody>` or `<equality>` appended at
 * the end adds to the model, even one whose bodies live in `<include>`d files. Appending,
 * never inserting, keeps every existing body id and `qpos` address, which
 * `PolicyStateBuilder` and a traced graph's `sim` slots index by.
 *
 * No `/` in an injected name: `slotReader/indexing.ts` calls a model entity-prefixed if
 * any joint, body or site name has one, so a single `injected/box` empties every entity.
 */
import type { MainModule } from 'mujoco';

/** Add `block` (raw MJCF, top-level sections) just before `</mujoco>`. */
export function appendToMjcf(xml: string, block: string): string {
  const close = xml.lastIndexOf('</mujoco>');
  if (close < 0) {
    throw new Error('appendToMjcf: scene XML has no closing </mujoco>');
  }
  return `${xml.slice(0, close)}${block}${xml.slice(close)}`;
}

/** Rewrite a scene XML in the Emscripten VFS in place, ahead of `mj_loadXML`. */
export function rewriteMjcfFile(mujoco: MainModule, path: string, rewrite: (xml: string) => string): void {
  const xml = new TextDecoder().decode(mujoco.FS.readFile(path));
  mujoco.FS.writeFile(path, rewrite(xml));
}

/** The MJCF text of a scene in the VFS. */
export function readMjcfFile(mujoco: MainModule, path: string): string {
  return new TextDecoder().decode(mujoco.FS.readFile(path));
}

/**
 * Whether the model namespaces its elements, as mjlab's `attach` does (`robot/torso`).
 * `buildEntityIndex`'s verdict, read off the text because injection is decided before
 * compiling. It matches every `name` attribute, where that checks only joints, bodies
 * and sites.
 */
export function isEntityPrefixed(xml: string): boolean {
  return /\bname\s*=\s*"[^"]*\/[^"]*"/.test(xml);
}
