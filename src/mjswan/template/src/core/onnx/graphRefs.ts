/**
 * Lists the traced-graph paths a config refers to, so a caller can resolve them to
 * bytes before `loadScene` (the engine never fetches).
 *
 * Structural rather than schema-aware — every `onnx` string under the term-bearing
 * sections — so a new term kind needs no change here. A missing graph is caught by
 * the manager that wanted it.
 */

/** Sections of a policy config whose entries may carry a traced graph. */
const POLICY_SECTIONS = ['observations', 'actions', 'actuators', 'terminations', 'commands', 'events'] as const;

function collectFrom(value: unknown, into: Set<string>): void {
  if (Array.isArray(value)) {
    for (const item of value) collectFrom(item, into);
    return;
  }
  if (typeof value !== 'object' || value === null) return;
  const record = value as Record<string, unknown>;
  // `onnx` is a per-term graph, `fused` a group's. Only term-bearing sections are
  // scanned, so a policy entry's own `onnx`, the network's path, never lands here.
  for (const key of ['onnx', 'fused'] as const) {
    const ref = record[key];
    if (typeof ref === 'string') into.add(ref);
  }
  for (const nested of Object.values(record)) collectFrom(nested, into);
}

/**
 * The actuator laws' graph paths, which `core/actuator/` evaluates in place of an ONNX
 * Runtime session.
 */
export function actuatorGraphRefs(config: Record<string, unknown>): Set<string> {
  const refs = new Set<string>();
  collectFrom(config.actuators, refs);
  return refs;
}

/** Graph paths a policy's config refers to (scene-relative), sorted for determinism. */
export function policyGraphRefs(config: Record<string, unknown>): string[] {
  const refs = new Set<string>();
  for (const section of POLICY_SECTIONS) collectFrom(config[section], refs);
  return [...refs].sort();
}
