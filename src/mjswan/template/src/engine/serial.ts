/**
 * Runs async work one at a time, in call order. A rejection reaches only its own caller,
 * so one failed load never blocks the next.
 */
export function createSerial(): <T>(work: () => Promise<T>) => Promise<T> {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(work: () => Promise<T>): Promise<T> => {
    const run = tail.then(work);
    tail = run.catch(() => undefined);
    return run;
  };
}
