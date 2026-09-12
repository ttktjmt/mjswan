/**
 * Every `EntityData` property the browser reads natively, keyed as mjlab names it — one
 * file per section of `mjlab/entity/data.py`, so a mjlab bump maps to a file. The one
 * property with no reader is `joint_torques`, which raises in mjlab too.
 *
 * `compile/tracer.py`'s `READER_FIELDS` names the same set; a field here that Python
 * does not list is traced through instead of read, one listed there but missing here
 * freezes the observation. `slotReaderParity.test.ts` checks both directions.
 */
import { BODY_READERS } from './body';
import { DERIVED_READERS } from './derived';
import { FORCE_READERS } from './forces';
import { GEOM_READERS } from './geom';
import { JOINT_READERS } from './joint';
import { ROOT_READERS } from './root';
import { SITE_READERS } from './site';
import { TENDON_READERS } from './tendon';
import type { FieldReader } from './util';

export type { FieldReader } from './util';

export const FIELD_READERS: Record<string, FieldReader> = {
  ...ROOT_READERS,
  ...BODY_READERS,
  ...GEOM_READERS,
  ...SITE_READERS,
  ...JOINT_READERS,
  ...FORCE_READERS,
  ...TENDON_READERS,
  ...DERIVED_READERS,
};
