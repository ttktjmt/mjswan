/**
 * Just enough of an ONNX protobuf writer to build the graphs the actuator tests read,
 * the malformed ones included.
 */

const utf8 = new TextEncoder();

function concat(parts: ArrayLike<number>[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/** A varint, a negative one as its 64-bit two's complement, as protobuf writes int64. */
export function varint(value: number | bigint): Uint8Array {
  let v = BigInt.asUintN(64, BigInt(value));
  const bytes: number[] = [];
  do {
    const byte = Number(v & 0x7fn);
    v >>= 7n;
    bytes.push(v > 0n ? byte | 0x80 : byte);
  } while (v > 0n);
  return Uint8Array.from(bytes);
}

function float32(value: number): Uint8Array {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setFloat32(0, value, true);
  return bytes;
}

const key = (field: number, wire: number) => varint(field * 8 + wire);
export const field = {
  varint: (n: number, value: number | bigint) => concat([key(n, 0), varint(value)]),
  float: (n: number, value: number) => concat([key(n, 5), float32(value)]),
  bytes: (n: number, payload: Uint8Array) => concat([key(n, 2), varint(payload.length), payload]),
  string: (n: number, value: string) => field.bytes(n, utf8.encode(value)),
  floats: (n: number, values: number[]) => field.bytes(n, concat(values.map(float32))),
  varints: (n: number, values: number[]) => field.bytes(n, concat(values.map((v) => varint(v)))),
};

export const DATA_TYPE = { FLOAT: 1, INT64: 7, BOOL: 9 } as const;

export interface TensorSpec {
  name?: string;
  dims: number[];
  values: number[];
  dataType?: number;
  /** Write the values as `raw_data` rather than `float_data` or `int32_data`. */
  raw?: boolean;
  /** Extra fields, written after the rest. */
  extra?: Uint8Array[];
}

export function tensor(spec: TensorSpec): Uint8Array {
  const dataType = spec.dataType ?? DATA_TYPE.FLOAT;
  const parts = [field.varints(1, spec.dims), field.varint(2, dataType)];
  if (spec.raw) {
    const raw =
      dataType === DATA_TYPE.BOOL ? Uint8Array.from(spec.values) : concat(spec.values.map(float32));
    parts.push(field.bytes(9, raw));
  } else if (dataType === DATA_TYPE.BOOL) {
    parts.push(field.varints(5, spec.values));
  } else {
    parts.push(field.floats(4, spec.values));
  }
  if (spec.name !== undefined) parts.push(field.string(8, spec.name));
  parts.push(...(spec.extra ?? []));
  return concat(parts);
}

export interface AttributeSpec {
  name: string;
  float?: number;
  int?: number;
  floats?: number[];
  ints?: number[];
  tensor?: TensorSpec;
  string?: string;
}

function attribute(spec: AttributeSpec): Uint8Array {
  // AttributeProto.AttributeType: FLOAT 1, INT 2, STRING 3, TENSOR 4, FLOATS 6, INTS 7.
  const parts = [field.string(1, spec.name)];
  if (spec.float !== undefined) parts.push(field.float(2, spec.float), field.varint(20, 1));
  else if (spec.int !== undefined) parts.push(field.varint(3, spec.int), field.varint(20, 2));
  else if (spec.string !== undefined) parts.push(field.string(4, spec.string), field.varint(20, 3));
  else if (spec.tensor) parts.push(field.bytes(5, tensor(spec.tensor)), field.varint(20, 4));
  else if (spec.floats) parts.push(field.floats(7, spec.floats), field.varint(20, 6));
  else if (spec.ints) parts.push(field.varints(8, spec.ints), field.varint(20, 7));
  return concat(parts);
}

export interface NodeSpec {
  op: string;
  inputs: string[];
  outputs: string[];
  attributes?: AttributeSpec[];
  domain?: string;
}

function node(spec: NodeSpec): Uint8Array {
  return concat([
    ...spec.inputs.map((name) => field.string(1, name)),
    ...spec.outputs.map((name) => field.string(2, name)),
    field.string(4, spec.op),
    ...(spec.attributes ?? []).map((a) => field.bytes(5, attribute(a))),
    ...(spec.domain !== undefined ? [field.string(7, spec.domain)] : []),
  ]);
}

export interface ModelSpec {
  nodes: NodeSpec[];
  inputs?: string[];
  outputs?: string[];
  initializers?: TensorSpec[];
  opset?: number;
  /** Extra fields for the graph, written after the rest. */
  graphExtra?: Uint8Array[];
}

/** A model over the five actuator inputs and one `torque`, unless told otherwise. */
export function onnxModel(spec: ModelSpec): Uint8Array {
  const valueInfo = (name: string) => field.string(1, name);
  const graph = concat([
    ...spec.nodes.map((n) => field.bytes(1, node(n))),
    field.string(2, 'law'),
    ...(spec.initializers ?? []).map((t) => field.bytes(5, tensor(t))),
    ...(spec.inputs ?? ['pos', 'vel', 'pos_target', 'vel_target', 'effort_target']).map((name) =>
      field.bytes(11, valueInfo(name)),
    ),
    ...(spec.outputs ?? ['torque']).map((name) => field.bytes(12, valueInfo(name))),
    ...(spec.graphExtra ?? []),
  ]);
  return concat([
    field.varint(1, 8),
    field.bytes(8, concat([field.string(1, ''), field.varint(2, spec.opset ?? 17)])),
    field.bytes(7, graph),
  ]);
}
