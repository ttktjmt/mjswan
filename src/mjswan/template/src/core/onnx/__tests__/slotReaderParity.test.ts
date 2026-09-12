/**
 * Slot-reader parity against mjlab itself.
 *
 * `slotReader.test.ts` checks the reader's logic on a hand-built model where every
 * expected number is computed by hand. This file checks the thing that test
 * cannot: that the semantics are *mjlab's*. The fixture is dumped from live,
 * stepped mjlab envs (`tests/dump_slot_fixture.py`) and carries both the raw
 * `mjModel`/`mjData` arrays the reader indexes and mjlab's own
 * `env.scene[entity].data.<field>` value for each one — every field
 * `READER_FIELDS` names — plus mjlab's `EntityIndexing`, the ids the reader must
 * resolve each entity to.
 *
 * This closes the last gap in the ADR 0005 verification chain: the Python parity
 * harness proves each traced graph reproduces mjlab's term, and this proves the
 * browser feeds that graph the same numbers mjlab would. With `test_onnx_parity.py`'s
 * traced-path sweep (property math in the graph, checked against mjlab), it also
 * gives reader ≡ traced for every property that has a reader.
 *
 * Regenerate the fixture whenever a field is added to the reader — the coverage
 * assertion at the bottom fails on a reader never compared, and every dumped field
 * must be readable, so the two lists cannot drift apart silently.
 */
import { describe, expect, it } from 'vitest';

import fixture from './fixtures/slotFields.json';
import { createSlotReader, isReadableEntityField, type SlotReaderContext } from '../slotReader';
import { FIELD_READERS } from '../slotReader/fields';
import { buildEntityIndex } from '../slotReader/indexing';

type EntityFixture = {
  fields: Record<string, number[]>;
  /** mjlab's per-episode randomized encoder bias, by unprefixed joint name. */
  encoder_bias: Record<string, number>;
  /** mjlab's `EntityIndexing`: the compiled ids and addresses the reader must resolve to. */
  indexing: {
    root_body_id: number;
    body_ids: number[];
    geom_ids: number[];
    site_ids: number[];
    tendon_ids: number[];
    ctrl_ids: number[];
    joint_q_adr: number[];
    joint_v_adr: number[];
  };
};

type TaskFixture = {
  model: Record<string, number | number[]>;
  data: Record<string, number[]>;
  entities: Record<string, EntityFixture>;
  sensors: Record<string, number[]>;
};

const TASKS = fixture as unknown as Record<string, TaskFixture>;

/** Every field compared somewhere, across all tasks: the coverage assertion's input. */
const compared = new Set<string>();

function contextFor(task: TaskFixture): SlotReaderContext {
  const { names, ...rest } = task.model as Record<string, number | number[]>;
  const mjModel = {
    ...rest,
    names: Uint8Array.from(names as number[]).buffer,
  };
  const mjData: Record<string, Float64Array> = {};
  for (const [key, values] of Object.entries(task.data)) {
    mjData[key] = Float64Array.from(values);
  }
  return { mjModel, mjData } as unknown as SlotReaderContext;
}

/** mjData is float64 and the read casts to float32, so compare at float32 resolution. */
function expectClose(actual: Float32Array, expected: number[], label: string): void {
  expect(actual.length, `${label}: length`).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const tolerance = Math.max(1e-5, Math.abs(expected[i]) * 1e-5);
    expect(Math.abs(actual[i] - expected[i]), `${label}[${i}]`).toBeLessThan(tolerance);
  }
}

describe.each(Object.keys(TASKS))('slot reader vs mjlab — %s', taskId => {
  const task = TASKS[taskId];
  // The walking tasks randomize the `encoder_bias` that `joint_pos_biased` observes, so
  // the reader gets the same lookup. Merging every entity's is safe: joint names are unique.
  const jointBias = new Map<string, number>();
  for (const entity of Object.values(task.entities)) {
    for (const [joint, bias] of Object.entries(entity.encoder_bias)) {
      jointBias.set(joint, bias);
    }
  }
  const biasOf = (name: string): number => jointBias.get(name) ?? 0;
  const read = createSlotReader(() => contextFor(task), { jointBias: biasOf });

  // mjlab attaches terrain with `prefix=""`, so it is not name-resolvable — asserted below.
  const entities = Object.entries(task.entities).filter(([entity]) => entity !== 'terrain');
  const cases: Array<[string, string, number[]]> = [];
  for (const [entity, { fields }] of entities) {
    for (const [field, expected] of Object.entries(fields)) {
      cases.push([entity, field, expected]);
      compared.add(field);
    }
  }

  it.each(cases)('%s.%s matches mjlab', (entity, field, expected) => {
    // The fixture dumps `READER_FIELDS`; a field there the browser cannot read would
    // be a slot the build emits and the runtime freezes on.
    expect(isReadableEntityField(field), `${field}: no reader for a field the build emits`).toBe(true);
    const value = read({ entity, field, input: `${entity}__${field}` });
    expect(value, `${entity}.${field} unreadable`).not.toBeNull();
    expectClose(value!, expected, `${entity}.${field}`);
  });

  it.each(entities)('%s resolves to the elements mjlab indexes', (entity, { indexing }) => {
    const { mjModel } = contextFor(task);
    const index = buildEntityIndex(mjModel!, entity, biasOf);
    expect(index.rootBodyId).toBe(indexing.root_body_id);
    expect(index.bodyIds).toEqual(indexing.body_ids);
    expect(index.geomIds).toEqual(indexing.geom_ids);
    expect(index.siteIds).toEqual(indexing.site_ids);
    expect(index.tendonIds).toEqual(indexing.tendon_ids);
    expect(index.ctrlIds).toEqual(indexing.ctrl_ids);
    expect(index.qposAdr).toEqual(indexing.joint_q_adr);
    expect(index.qvelAdr).toEqual(indexing.joint_v_adr);
  });

  const sensorCases = Object.entries(task.sensors);
  it.each(sensorCases)('sensor %s matches mjlab', (sensor, expected) => {
    const value = read({ sensor, input: `sensor__${sensor}` });
    expect(value, `sensor ${sensor} unreadable`).not.toBeNull();
    expectClose(value!, expected, `sensor ${sensor}`);
  });

  it.runIf('terrain' in task.entities)(
    'reports the prefix-free terrain as unavailable, not as another entity',
    () => {
      // `terrain` matching no prefix must not fall back and hand over the robot's joints.
      expect(read({ entity: 'terrain', field: 'joint_pos', input: 'terrain__joint_pos' })).toEqual(
        new Float32Array(0),
      );
      expect(read({ entity: 'terrain', field: 'root_link_pos_w', input: 'x' })).toBeNull();
    },
  );
});

describe('slot reader vs mjlab — coverage', () => {
  it('compares every reader with mjlab in at least one entity', () => {
    // Guards a field added to the reader but never compared: regenerate, and it appears.
    // The tasks carry no tendons, which is what the synthetic entity is in the fixture for.
    expect(Object.keys(FIELD_READERS)).toHaveLength(55);
    for (const field of Object.keys(FIELD_READERS)) {
      expect(compared.has(field), `${field} not compared with mjlab`).toBe(true);
    }
  });
});
