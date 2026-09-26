import { describe, expect, it } from 'vitest';

import { handTrackingBlocker, xrSessionInit, xrSessionReports } from '../session';

describe('xrSessionReports', () => {
  it('lists only what the device supports, VR first', () => {
    expect(xrSessionReports({ vr: true, ar: true }, null).map((s) => s.id)).toEqual(['vr', 'ar']);
    expect(xrSessionReports({ vr: false, ar: true }, null).map((s) => s.id)).toEqual(['ar']);
    expect(xrSessionReports({ vr: false, ar: false }, null)).toEqual([]);
  });

  it('names the press: entering, or leaving the session that runs', () => {
    expect(xrSessionReports({ vr: true, ar: true }, 'ar')).toEqual([
      { id: 'vr', label: 'Enter VR', active: false },
      { id: 'ar', label: 'Stop AR', active: true },
    ]);
    expect(xrSessionReports({ vr: true, ar: false }, 'vr')).toEqual([
      { id: 'vr', label: 'Exit VR', active: true },
    ]);
  });
});

describe('xrSessionInit', () => {
  it('asks VR for what three VRButton did, and AR for the floor', () => {
    expect(xrSessionInit('vr', false).optionalFeatures).toEqual(['local-floor', 'bounded-floor', 'layers']);
    expect(xrSessionInit('ar', false).optionalFeatures).toEqual(['local-floor']);
  });

  it('asks for hand tracking only with the switch on, in either mode', () => {
    expect(xrSessionInit('vr', true).optionalFeatures).toContain('hand-tracking');
    expect(xrSessionInit('ar', true).optionalFeatures).toContain('hand-tracking');
    expect(xrSessionInit('vr', false).optionalFeatures).not.toContain('hand-tracking');
  });
});

describe('handTrackingBlocker', () => {
  it('lets an MJCF scene take the hands', () => {
    expect(handTrackingBlocker({ format: 'mjz', prefixed: false, hasPolicy: false })).toBeNull();
    expect(handTrackingBlocker({ format: 'mjz', prefixed: true, hasPolicy: true })).toBeNull();
  });

  it('refuses a compiled model, which has no MJCF to add them to', () => {
    expect(handTrackingBlocker({ format: 'mjb', prefixed: false, hasPolicy: false })).toMatch(/compiled/);
  });

  it('refuses a policy scene whose bodies all count as the entity', () => {
    expect(handTrackingBlocker({ format: 'mjz', prefixed: false, hasPolicy: true })).toMatch(/policy/);
  });
});
