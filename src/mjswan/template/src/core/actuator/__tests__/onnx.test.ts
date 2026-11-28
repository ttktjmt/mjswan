/**
 * The actuator graph reader takes bytes from whoever wrote the document, so a malformed
 * or oversized model has to fail with a message, not hang or read out of bounds.
 */
import { describe, expect, it } from 'vitest';

import { LIMITS, readOnnxGraph } from '../onnx';
import { DATA_TYPE, field, onnxModel, varint } from './onnxWriter';

const ADD = { op: 'Add', inputs: ['pos', 'vel'], outputs: ['torque'] };

describe('readOnnxGraph', () => {
  it('reads nodes, attributes, initializers, inputs, outputs and the opset', () => {
    const graph = readOnnxGraph(
      onnxModel({
        opset: 13,
        initializers: [
          { name: 'kp', dims: [3], values: [1, 2, 3] },
          { name: 'lim', dims: [], values: [4.5], raw: true },
        ],
        nodes: [
          {
            op: 'Clip',
            inputs: ['pos', '', 'lim'],
            outputs: ['torque'],
            attributes: [
              { name: 'f', float: 0.25 },
              { name: 'axis', int: -1 },
              { name: 'fs', floats: [1, 2] },
              { name: 'is', ints: [3, -4] },
              { name: 'mode', string: 'x' },
            ],
          },
        ],
      }),
    );
    expect(graph.opset).toBe(13);
    expect(graph.inputs).toEqual(['pos', 'vel', 'pos_target', 'vel_target', 'effort_target']);
    expect(graph.outputs).toEqual(['torque']);
    expect([...graph.initializers.keys()]).toEqual(['kp', 'lim']);
    expect(graph.initializers.get('kp')).toMatchObject({ dims: [3], dataType: DATA_TYPE.FLOAT });
    expect([...graph.initializers.get('kp')!.values]).toEqual([1, 2, 3]);
    expect([...graph.initializers.get('lim')!.values]).toEqual([4.5]);

    const [clip] = graph.nodes;
    expect(clip).toMatchObject({ opType: 'Clip', domain: '', inputs: ['pos', '', 'lim'], outputs: ['torque'] });
    expect(clip.attributes.get('f')).toEqual({ kind: 'float', value: 0.25 });
    expect(clip.attributes.get('axis')).toEqual({ kind: 'int', value: -1 });
    expect(clip.attributes.get('fs')).toEqual({ kind: 'floats', value: [1, 2] });
    expect(clip.attributes.get('is')).toEqual({ kind: 'ints', value: [3, -4] });
    expect(clip.attributes.get('mode')).toEqual({ kind: 'other' });
  });

  it('reads a tensor attribute and a bool tensor as 0 and 1', () => {
    const graph = readOnnxGraph(
      onnxModel({
        initializers: [
          { name: 'mask', dims: [3], values: [1, 0, 1], dataType: DATA_TYPE.BOOL },
          { name: 'raw_mask', dims: [2], values: [0, 2], dataType: DATA_TYPE.BOOL, raw: true },
        ],
        nodes: [
          {
            op: 'Constant',
            inputs: [],
            outputs: ['torque'],
            attributes: [{ name: 'value', tensor: { dims: [1], values: [-2.5] } }],
          },
        ],
      }),
    );
    expect([...graph.initializers.get('mask')!.values]).toEqual([1, 0, 1]);
    expect([...graph.initializers.get('raw_mask')!.values]).toEqual([0, 1]);
    const value = graph.nodes[0].attributes.get('value');
    expect(value?.kind).toBe('tensor');
    if (value?.kind === 'tensor') expect([...value.value.values]).toEqual([-2.5]);
  });

  it('drops an initializer listed among the inputs, as IR before 4 writes it', () => {
    const graph = readOnnxGraph(
      onnxModel({
        inputs: ['pos', 'vel', 'kp'],
        initializers: [{ name: 'kp', dims: [], values: [2] }],
        nodes: [ADD],
      }),
    );
    expect(graph.inputs).toEqual(['pos', 'vel']);
  });

  it('skips fields it does not read', () => {
    const model = onnxModel({ nodes: [ADD] });
    const extra = new Uint8Array([...field.string(2, 'producer'), ...field.varint(5, 3), ...field.float(99, 1)]);
    const graph = readOnnxGraph(new Uint8Array([...extra, ...model]));
    expect(graph.nodes.map((n) => n.opType)).toEqual(['Add']);
  });

  it.each([
    ['no graph', field.varint(1, 8), /holds no graph/],
    ['a truncated length', new Uint8Array([...field.varint(1, 8), 0x3a, 0x10, 0x01]), /names more bytes than it holds/],
    ['a truncated number', new Uint8Array([0x08, 0x80]), /ends inside a number/],
    ['an overlong number', new Uint8Array([0x08, ...Array(10).fill(0x80), 0x01]), /malformed number/],
    ['a group field', new Uint8Array([0x0b]), /wire type 3/],
    ['a truncated float', new Uint8Array([0x15, 0x00, 0x00]), /ends inside a value/],
  ])('refuses a model with %s', (_, bytes, message) => {
    expect(() => readOnnxGraph(bytes)).toThrow(message);
  });

  it('refuses a model past the byte limit before reading it', () => {
    expect(() => readOnnxGraph(new Uint8Array(LIMITS.bytes + 1))).toThrow(/over 1048576/);
  });

  it('refuses a graph past the node limit', () => {
    const nodes = Array.from({ length: LIMITS.nodes + 1 }, (_, i) => ({
      op: 'Identity',
      inputs: ['pos'],
      outputs: [`v${i}`],
    }));
    expect(() => readOnnxGraph(onnxModel({ nodes }))).toThrow(/over 4096 nodes/);
  });

  it.each([
    ['kept outside the file', { extra: [field.varint(14, 1)] }, /outside the file/],
    ['of int64', { dataType: DATA_TYPE.INT64, raw: true }, /data type 7/],
    ['of a negative dimension', { dims: [-1] }, /shape \[-1\]/],
    ['of too many dimensions', { dims: Array(LIMITS.dims + 1).fill(1), values: [1] }, /shape/],
    ['of too many values', { dims: [LIMITS.tensorElements + 1], values: [] }, /4097 values/],
    ['shorter than its shape', { dims: [4] }, /wrong size/],
    ['of raw bytes shorter than its shape', { dims: [4], raw: true }, /wrong size/],
  ])('refuses a tensor %s', (_, over, message) => {
    const initializer = { name: 'kp', dims: [3], values: [1, 2, 3], ...over };
    expect(() => readOnnxGraph(onnxModel({ nodes: [ADD], initializers: [initializer] }))).toThrow(message);
  });

  it('refuses a sparse initializer', () => {
    const model = onnxModel({ nodes: [ADD], graphExtra: [field.bytes(15, new Uint8Array())] });
    expect(() => readOnnxGraph(model)).toThrow(/sparse initializer/);
  });

  it('refuses a number past 2^53', () => {
    const tensor = new Uint8Array([...field.bytes(1, varint(2n ** 60n)), ...field.varint(2, 1)]);
    const model = onnxModel({ nodes: [ADD], graphExtra: [field.bytes(5, tensor)] });
    expect(() => readOnnxGraph(model)).toThrow(/past 2\^53/);
  });
});
