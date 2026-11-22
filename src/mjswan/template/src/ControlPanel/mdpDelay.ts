import { MAX_DELAY_S } from '../core/mdp/operator';
import type { TerminationTermDescriptor } from '../engine';

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

/** How much of a delayed termination's held reset has run, 0 to 1; 0 while it holds none. */
export function heldProgress(term: Pick<TerminationTermDescriptor, 'onFire' | 'resetAt'>, time: number): number {
  if (term.resetAt === null || typeof term.onFire !== 'object' || term.onFire.delay <= 0) return 0;
  return Math.min(1, Math.max(0, 1 - (term.resetAt - time) / term.onFire.delay));
}
