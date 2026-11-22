/**
 * What an operator may change about a running MDP's terms, and what the terms report back.
 * mjlab's timing is the default; an override is the viewer's, never the document's.
 * See context/frontend.md.
 */

export type OverridableKind = 'termination' | 'event';

/**
 * What follows a termination's firing, one at a time: the reset at once (mjlab's), the reset
 * `delay` seconds later, or a pause on the step it fired, the reset it owes landing on play.
 */
export type OnFire = 'reset' | 'stop' | { delay: number };

export interface TerminationOverride {
  /** Off: evaluated and reported, but never ends the episode. Stop still pauses. */
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

/** Past a minute a delay only means "off", which has its own switch. */
export const MAX_DELAY_S = 60;

const flag = (value: unknown, current: boolean): boolean => (typeof value === 'boolean' ? value : current);

/** An `onFire` as the engine keeps it, or null for one it cannot: a zero delay is a reset. */
export function normalOnFire(value: unknown): OnFire | null {
  if (value === 'reset' || value === 'stop') return value;
  const delay = (value as { delay?: unknown } | null)?.delay;
  if (typeof delay !== 'number' || !Number.isFinite(delay)) return null;
  const clamped = Math.min(Math.max(delay, 0), MAX_DELAY_S);
  return clamped === 0 ? 'reset' : { delay: clamped };
}

/** The seconds an `onFire` holds the reset for. */
export function delayOf(onFire: OnFire): number {
  return typeof onFire === 'object' ? onFire.delay : 0;
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
   * A termination `fired` when its condition turned true, `stopped` the simulation when
   * its Stop was on, and `reset` when it ended the episode. An event `fired` when its
   * write landed.
   */
  what: 'fired' | 'stopped' | 'reset';
  /** Episode time, in seconds. */
  time: number;
  /** The termination was off, so no reset followed. */
  off?: boolean;
  /** The event was triggered by hand rather than by its mode. */
  byHand?: boolean;
}

export type ActivityListener = (activity: TermActivity) => void;

const same = (a: object, b: object): boolean => JSON.stringify(a) === JSON.stringify(b);

/** Overrides by kind and name, kept across resets and MDP switches: a name is the handle. */
export class TermOverrides {
  private readonly terminations = new Map<string, TerminationOverride>();
  private readonly events = new Map<string, EventOverride>();

  termination(name: string): TerminationOverride {
    return this.terminations.get(name) ?? { ...DEFAULT_TERMINATION };
  }

  event(name: string): EventOverride {
    return this.events.get(name) ?? { ...DEFAULT_EVENT };
  }

  setTermination(name: string, patch: TerminationOverridePatch): TerminationOverride {
    const next = mergeTermination(this.termination(name), patch);
    if (same(next, DEFAULT_TERMINATION)) this.terminations.delete(name);
    else this.terminations.set(name, next);
    return next;
  }

  setEvent(name: string, patch: EventOverridePatch): EventOverride {
    const next = mergeEvent(this.event(name), patch);
    if (same(next, DEFAULT_EVENT)) this.events.delete(name);
    else this.events.set(name, next);
    return next;
  }

  terminationEntries(): ReadonlyArray<[string, TerminationOverride]> {
    return [...this.terminations];
  }

  eventEntries(): ReadonlyArray<[string, EventOverride]> {
    return [...this.events];
  }

  clear(): void {
    this.terminations.clear();
    this.events.clear();
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
