/**
 * Appending to a compiled-from-text MJCF scene.
 *
 * MJCF merges repeated top-level sections, so a second `<worldbody>` and `<equality>` at
 * the end of the file add to the first — which is what lets the viewer put its own bodies
 * into a model it knows nothing about, including one whose bodies live in `<include>`d
 * files.
 *
 * **Appended, never inserted.** Everything already in the model keeps its body id and its
 * `qpos` address, and the additions take the tail. `PolicyStateBuilder` reads the robot's
 * root at `qpos[0]`, and a traced graph's `sim` slots index by the ids the build saw.
 *
 * **No `/` in an injected name.** `slotReader/indexing.ts` decides whether a model is
 * entity-prefixed by looking for one anywhere in the names; a name like `injected/box`
 * flips that verdict for the whole model and empties every entity's element list.
 */
import type { MainModule } from 'mujoco';

/** Add `block` — raw MJCF, top-level sections — just before `</mujoco>`. */
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

/** The MJCF text of a scene in the VFS, for a check that has to happen before compiling. */
export function readMjcfFile(mujoco: MainModule, path: string): string {
  return new TextDecoder().decode(mujoco.FS.readFile(path));
}

/**
 * Whether the model namespaces its elements, as mjlab's `attach` does (`robot/torso`).
 *
 * This is the same verdict `buildEntityIndex` reaches from the compiled model, read off
 * the text instead because the decision it gates — whether to inject extra bodies — has
 * to be made before compiling. Deliberately a scan of the raw text: a name attribute is
 * a name attribute wherever it appears, and over-reporting here only means declining to
 * inject into a model that would have tolerated it.
 */
export function isEntityPrefixed(xml: string): boolean {
  return /\bname\s*=\s*"[^"]*\/[^"]*"/.test(xml);
}
