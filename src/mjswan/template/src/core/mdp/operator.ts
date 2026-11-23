/**
 * What an operator may change about a running MDP's terms, and what the terms report back.
 * mjlab's timing is the default unless the build set a term's own (a scene's
 * `term_defaults`); an operator's change is the viewer's, never the document's.
 * See context/frontend.md.
 */

export type OverridableKind = 'termination' | 'event';

/**
 * What follows a termination's firing, one at a time: the reset at once (mjlab's), the reset
 * `delay` seconds later, or a freeze on the step it fired, until a reset (`'freeze'`) or for
 * `freeze` seconds of wall time, after which the reset lands and the simulation plays.
 */
export type OnFire = 'reset' | 'freeze' | { delay: number } | { freeze: number };

export interface TerminationOverride {
  /** Off: evaluated and reported, but never ends the episode. A freeze still pauses. */
  enabled: boolean;
  onFire: OnFire;
}

/** A reset or interval event term's override. */
export interface EventOverride {
  /** Off: no write at its trigger. A trigger by hand still writes. */
  enabled: boolean;
}

export type TerminationOverridePatch = Partial<TerminationOverride>;
export type EventOverridePatch = Partial<EventOverride>;

export const DEFAULT_TERMINATION: Readonly<TerminationOverride> = { enabled: true, onFire: 'reset' };
export const DEFAULT_EVENT: Readonly<EventOverride> = { enabled: true };

/** Past a minute a delay or a timed freeze only means "off" or "until reset", which exist. */
export const MAX_DELAY_S = 60;

/** A build's starting overrides by term name (a scene's `term_defaults`). */
export interface TermDefaults {
  terminations?: Readonly<Record<string, TerminationOverridePatch>>;
  events?: Readonly<Record<string, EventOverridePatch>>;
}

const flag = (value: unknown, current: boolean): boolean => (typeof value === 'boolean' ? value : current);

const seconds = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? Math.min(Math.max(value, 0), MAX_DELAY_S) : null;

/** An `onFire` as the engine keeps it, or null for one it cannot: a zero wait is a reset. */
export function normalOnFire(value: unknown): OnFire | null {
  if (value === 'reset' || value === 'freeze') return value;
  if (value === null || typeof value !== 'object') return null;
  const keys = Object.keys(value);
  if (keys.length !== 1) return null;
  const s = seconds((value as Record<string, unknown>)[keys[0]]);
  if (s === null) return null;
  if (s === 0) return 'reset';
  if (keys[0] === 'delay') return { delay: s };
  if (keys[0] === 'freeze') return { freeze: s };
  return null;
}

/** The seconds an `onFire` holds the reset for while the simulation runs. */
export function delayOf(onFire: OnFire): number {
  return typeof onFire === 'object' && 'delay' in onFire ? onFire.delay : 0;
}

/** Whether a firing freezes the simulation. */
export function isFreeze(onFire: OnFire): boolean {
  return onFire === 'freeze' || (typeof onFire === 'object' && 'freeze' in onFire);
}

/** A freeze's wall-clock seconds; null until a reset (and for an `onFire` that is no freeze). */
export function freezeOf(onFire: OnFire): number | null {
  return typeof onFire === 'object' && 'freeze' in onFire ? onFire.freeze : null;
}

/** A patch over `current`; an `onFire` the engine cannot keep leaves the current one. */
export function mergeTermination(
  current: TerminationOverride,
  patch: TerminationOverridePatch,
): TerminationOverride {
  return {
    enabled: flag(patch.enabled, current.enabled),
    onFire: (patch.onFire !== undefined && normalOnFire(patch.onFire)) || current.onFire,
  };
}

export function mergeEvent(current: EventOverride, patch: EventOverridePatch): EventOverride {
  return { enabled: flag(patch.enabled, current.enabled) };
}

/**
 * Whether `delay` seconds have passed since `since`. Episode time is steps times the
 * control period, so a wait of exactly that many steps lands a rounding error short.
 */
export function elapsed(now: number, since: number, delay: number): boolean {
  return now - since >= delay - 1e-9;
}

/** What a term did, as its manager reports it. */
export interface TermActivity {
  kind: OverridableKind;
  name: string;
  /**
   * A termination `fired` when its condition turned true, `frozen` the simulation when set
   * to freeze, and `reset` when it ended the episode. An event `fired` when its write landed.
   */
  what: 'fired' | 'frozen' | 'reset';
  /** Episode time, in seconds. */
  time: number;
  /** The termination was off, so no reset followed. */
  off?: boolean;
  /** The event was triggered by hand rather than by its mode. */
  byHand?: boolean;
}

export type ActivityListener = (activity: TermActivity) => void;

const same = (a: object, b: object): boolean => JSON.stringify(a) === JSON.stringify(b);

/**
 * Overrides by kind and name, kept across resets and MDP switches: a name is the handle. The
 * build's defaults are the starting point; the operator's changes sit on top until `clear`.
 */
export class TermOverrides {
  private readonly terminations = new Map<string, TerminationOverride>();
  private readonly events = new Map<string, EventOverride>();
  private terminationDefaults = new Map<string, TerminationOverride>();
  private eventDefaults = new Map<string, EventOverride>();

  /** A scene's `term_defaults`. An operator's change that now matches its default is dropped. */
  setDefaults(defaults: TermDefaults | null | undefined): void {
    this.terminationDefaults = new Map(
      Object.entries(defaults?.terminations ?? {}).map(([name, patch]) => [name, mergeTermination(DEFAULT_TERMINATION, patch)]),
    );
    this.eventDefaults = new Map(
      Object.entries(defaults?.events ?? {}).map(([name, patch]) => [name, mergeEvent(DEFAULT_EVENT, patch)]),
    );
    for (const [name, value] of this.terminations) if (same(value, this.terminationDefault(name))) this.terminations.delete(name);
    for (const [name, value] of this.events) if (same(value, this.eventDefault(name))) this.events.delete(name);
  }

  termination(name: string): TerminationOverride {
    return { ...(this.terminations.get(name) ?? this.terminationDefault(name)) };
  }

  event(name: string): EventOverride {
    return { ...(this.events.get(name) ?? this.eventDefault(name)) };
  }

  setTermination(name: string, patch: TerminationOverridePatch): TerminationOverride {
    const next = mergeTermination(this.termination(name), patch);
    if (same(next, this.terminationDefault(name))) this.terminations.delete(name);
    else this.terminations.set(name, next);
    return next;
  }

  setEvent(name: string, patch: EventOverridePatch): EventOverride {
    const next = mergeEvent(this.event(name), patch);
    if (same(next, this.eventDefault(name))) this.events.delete(name);
    else this.events.set(name, next);
    return next;
  }

  /** Whether the operator moved a term off the build's default. */
  changed(kind: OverridableKind, name: string): boolean {
    return kind === 'termination' ? this.terminations.has(name) : this.events.has(name);
  }

  /** Every term with a value other than mjlab's, the build's defaults included. */
  terminationEntries(): ReadonlyArray<[string, TerminationOverride]> {
    const names = new Set([...this.terminationDefaults.keys(), ...this.terminations.keys()]);
    return [...names].map((name) => [name, this.termination(name)]);
  }

  eventEntries(): ReadonlyArray<[string, EventOverride]> {
    const names = new Set([...this.eventDefaults.keys(), ...this.events.keys()]);
    return [...names].map((name) => [name, this.event(name)]);
  }

  /** Back to the build's defaults. */
  clear(): void {
    this.terminations.clear();
    this.events.clear();
  }

  private terminationDefault(name: string): TerminationOverride {
    return this.terminationDefaults.get(name) ?? DEFAULT_TERMINATION;
  }

  private eventDefault(name: string): EventOverride {
    return this.eventDefaults.get(name) ?? DEFAULT_EVENT;
  }
}

export interface LoggedActivity extends TermActivity {
  /** Rises across the engine's life, so a host can key on it. */
  seq: number;
  /** Which episode, counting every reset since the engine started. */
  episode: number;
}

/** The latest activities, oldest first. */
export class ActivityLog {
  private entries: LoggedActivity[] = [];
  private seq = 0;

  constructor(private readonly capacity = 32) {}

  push(activity: TermActivity, episode: number): void {
    this.entries.push({ ...activity, seq: ++this.seq, episode });
    if (this.entries.length > this.capacity) this.entries.shift();
  }

  list(): ReadonlyArray<LoggedActivity> {
    return [...this.entries];
  }
}
