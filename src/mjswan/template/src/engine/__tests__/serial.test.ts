import { describe, expect, it } from 'vitest';

import { createSerial } from '../serial';

const tick = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe('createSerial', () => {
  it('runs work one at a time, in call order', async () => {
    const serial = createSerial();
    const log: string[] = [];
    const job = (name: string, ms: number) =>
      serial(async () => {
        log.push(`${name}:start`);
        await tick(ms);
        log.push(`${name}:end`);
        return name;
      });

    const results = await Promise.all([job('a', 20), job('b', 0), job('c', 5)]);

    expect(results).toEqual(['a', 'b', 'c']);
    expect(log).toEqual(['a:start', 'a:end', 'b:start', 'b:end', 'c:start', 'c:end']);
  });

  it('rejects only the failed caller, and runs the next', async () => {
    const serial = createSerial();
    const failed = serial(async () => {
      throw new Error('boom');
    });
    const next = serial(async () => 'ran');

    await expect(failed).rejects.toThrow('boom');
    await expect(next).resolves.toBe('ran');
  });
});
