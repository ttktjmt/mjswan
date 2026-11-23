import { MAX_DELAY_S, delayOf, freezeOf } from '../core/mdp/operator';
import type { MdpState, OnFire, TerminationTermDescriptor } from '../engine';

/** The longest delay the engine takes, in the box's unit. */
export const MAX_DELAY_MS = MAX_DELAY_S * 1000;

/** A delay in seconds as the box writes it: whole milliseconds. */
export const toMs = (seconds: number): number => Math.round(seconds * 1000);

/** What was typed, as whole milliseconds within the engine's range; null when it is not a number. */
export function parseDelayMs(text: string): number | null {
  const digits = text.trim().replace(/\s*ms$/i, '');
  if (!/^\d+(\.\d*)?$/.test(digits)) return null;
  return Math.min(MAX_DELAY_MS, Math.round(Number(digits)));
}

const unit = (value: number): number => Math.min(1, Math.max(0, value));

/** How much of a delayed termination's held reset has run, 0 to 1; 0 while it holds none. */
export function heldProgress(term: Pick<TerminationTermDescriptor, 'onFire' | 'resetAt'>, time: number): number {
  const delay = delayOf(term.onFire);
  if (term.resetAt === null || delay <= 0) return 0;
  return unit(1 - (term.resetAt - time) / delay);
}

/** How much of a timed freeze this term holds has run, 0 to 1, at `now` (`Date.now()`); 0 otherwise. */
export function freezeProgress(frozenBy: MdpState['frozenBy'], name: string, onFire: OnFire, now: number): number {
  const seconds = freezeOf(onFire);
  if (frozenBy?.term !== name || frozenBy.resumesAt === null || !seconds) return 0;
  return unit(1 - (frozenBy.resumesAt - now) / (seconds * 1000));
}
