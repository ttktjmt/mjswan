/**
 * The part of an ONNX model an actuator graph uses, read from its protobuf bytes: the
 * graph's nodes, their attributes, its constant tensors and its input and output names.
 *
 * Written against the protobuf wire format rather than a generated decoder, since the
 * graph is small and a document may come from anyone: every length is checked against
 * the bytes it claims, and a model past `LIMITS` is refused before it is walked.
 */

/** ONNX's `TensorProto.DataType` values this reader keeps. */
export const DATA_TYPE = { FLOAT: 1, BOOL: 9 } as const;

/** What a graph may hold, checked as it is read. */
export const LIMITS = {
  bytes: 1 << 20,
  nodes: 4096,
  tensorElements: 4096,
  dims: 8,
} as const;

export interface OnnxTensor {
  name: string;
  dims: number[];
  dataType: number;
  /** Bools as 0 and 1. */
  values: Float32Array;
}

export type OnnxAttribute =
  | { kind: 'float'; value: number }
  | { kind: 'int'; value: number }
  | { kind: 'floats'; value: number[] }
  | { kind: 'ints'; value: number[] }
  | { kind: 'tensor'; value: OnnxTensor }
  | { kind: 'other' };

export interface OnnxNode {
  opType: string;
  domain: string;
  inputs: string[];
  outputs: string[];
  attributes: Map<string, OnnxAttribute>;
}

export interface OnnxGraph {
  nodes: OnnxNode[];
  initializers: Map<string, OnnxTensor>;
  inputs: string[];
  outputs: string[];
  /** The default domain's opset, 0 when the model declares none. */
  opset: number;
}

const WIRE = { VARINT: 0, FIXED64: 1, LENGTH: 2, FIXED32: 5 } as const;
const utf8 = new TextDecoder('utf-8', { fatal: true });

class Reader {
  private pos: number;

  constructor(
    private readonly bytes: Uint8Array,
    private readonly start = 0,
    private readonly end = bytes.length,
  ) {
    this.pos = start;
  }

  get done(): boolean {
    return this.pos >= this.end;
  }

  varint(): number {
    let value = 0;
    let scale = 1;
    for (let i = 0; i < 10; i++) {
      if (this.pos >= this.end) throw new Error('ONNX graph ends inside a number.');
      const byte = this.bytes[this.pos++];
      value += (byte & 0x7f) * scale;
      if (byte < 0x80) {
        if (!Number.isSafeInteger(value)) throw new Error('ONNX graph holds a number past 2^53.');
        return value;
      }
      scale *= 128;
    }
    throw new Error('ONNX graph holds a malformed number.');
  }

  /** A signed int64, as ONNX writes dims and int attributes: a negative one in ten bytes. */
  int64(): number {
    let value = 0n;
    for (let shift = 0n; shift < 70n; shift += 7n) {
      if (this.pos >= this.end) throw new Error('ONNX graph ends inside a number.');
      const byte = this.bytes[this.pos++];
      value |= BigInt(byte & 0x7f) << shift;
      if (byte < 0x80) {
        const signed = Number(BigInt.asIntN(64, value));
        if (!Number.isSafeInteger(signed)) throw new Error('ONNX graph holds a number past 2^53.');
        return signed;
      }
    }
    throw new Error('ONNX graph holds a malformed number.');
  }

  fixed32(): number {
    const view = this.view(4);
    return view.getFloat32(0, true);
  }

  /** The next length-delimited field's bytes, as a reader over them. */
  sub(): Reader {
    const length = this.varint();
    if (length > this.end - this.pos) throw new Error('ONNX graph names more bytes than it holds.');
    const reader = new Reader(this.bytes, this.pos, this.pos + length);
    this.pos += length;
    return reader;
  }

  string(): string {
    const sub = this.sub();
    return utf8.decode(this.bytes.subarray(sub.start, sub.end));
  }

  rest(): Uint8Array {
    return this.bytes.subarray(this.start, this.end);
  }

  skip(wire: number): void {
    if (wire === WIRE.VARINT) this.varint();
    else if (wire === WIRE.FIXED64) this.view(8);
    else if (wire === WIRE.LENGTH) this.sub();
    else if (wire === WIRE.FIXED32) this.view(4);
    else throw new Error(`ONNX graph holds a field of wire type ${wire}, which no ONNX writer uses.`);
  }

  tag(): { field: number; wire: number } {
    const key = this.varint();
    return { field: Math.floor(key / 8), wire: key % 8 };
  }

  private view(length: number): DataView {
    if (length > this.end - this.pos) throw new Error('ONNX graph ends inside a value.');
    const view = new DataView(this.bytes.buffer, this.bytes.byteOffset + this.pos, length);
    this.pos += length;
    return view;
  }
}

/** A repeated scalar field, packed or one value per tag. */
function repeated(reader: Reader, wire: number, read: (r: Reader) => number, into: number[]): void {
  if (wire !== WIRE.LENGTH) {
    into.push(read(reader));
    return;
  }
  const packed = reader.sub();
  while (!packed.done) into.push(read(packed));
}

function readTensor(reader: Reader): OnnxTensor {
  let name = '';
  const dims: number[] = [];
  let dataType = 0;
  const floats: number[] = [];
  const int32s: number[] = [];
  let raw: Uint8Array | null = null;
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1) repeated(reader, wire, (r) => r.int64(), dims);
    else if (field === 2) dataType = reader.varint();
    else if (field === 4) repeated(reader, wire, (r) => r.fixed32(), floats);
    else if (field === 5) repeated(reader, wire, (r) => r.varint(), int32s);
    else if (field === 8) name = reader.string();
    else if (field === 9) raw = reader.sub().rest();
    else if (field === 13 || field === 14) throw new Error('ONNX graph keeps a tensor outside the file.');
    else reader.skip(wire);
  }
  if (dims.length > LIMITS.dims || dims.some((d) => !Number.isInteger(d) || d < 0)) {
    throw new Error(`ONNX graph holds a tensor of shape [${dims.join(', ')}].`);
  }
  const count = dims.reduce((a, b) => a * b, 1);
  if (count > LIMITS.tensorElements) throw new Error(`ONNX graph holds a tensor of ${count} values.`);
  const values = new Float32Array(count);
  if (dataType === DATA_TYPE.FLOAT) {
    if (raw) {
      if (raw.length !== count * 4) throw new Error('ONNX graph holds a float tensor of the wrong size.');
      const view = new DataView(raw.buffer, raw.byteOffset, raw.length);
      for (let i = 0; i < count; i++) values[i] = view.getFloat32(4 * i, true);
    } else {
      if (floats.length !== count) throw new Error('ONNX graph holds a float tensor of the wrong size.');
      values.set(floats);
    }
  } else if (dataType === DATA_TYPE.BOOL) {
    const source = raw ?? Uint8Array.from(int32s);
    if (source.length !== count) throw new Error('ONNX graph holds a bool tensor of the wrong size.');
    for (let i = 0; i < count; i++) values[i] = source[i] ? 1 : 0;
  } else {
    throw new Error(`ONNX graph holds a tensor of data type ${dataType}; actuator graphs use float and bool.`);
  }
  return { name, dims, dataType, values };
}

function readAttribute(reader: Reader): [string, OnnxAttribute] {
  let name = '';
  let type = 0;
  let f: number | undefined;
  let i: number | undefined;
  let tensor: OnnxTensor | undefined;
  const floats: number[] = [];
  const ints: number[] = [];
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1) name = reader.string();
    else if (field === 2) f = reader.fixed32();
    else if (field === 3) i = reader.int64();
    else if (field === 5) tensor = readTensor(reader.sub());
    else if (field === 7) repeated(reader, wire, (r) => r.fixed32(), floats);
    else if (field === 8) repeated(reader, wire, (r) => r.int64(), ints);
    else if (field === 20) type = reader.varint();
    else reader.skip(wire);
  }
  // AttributeProto.AttributeType: FLOAT 1, INT 2, TENSOR 4, FLOATS 6, INTS 7.
  if (type === 1 && f !== undefined) return [name, { kind: 'float', value: f }];
  if (type === 2 && i !== undefined) return [name, { kind: 'int', value: i }];
  if (type === 4 && tensor) return [name, { kind: 'tensor', value: tensor }];
  if (type === 6) return [name, { kind: 'floats', value: floats }];
  if (type === 7) return [name, { kind: 'ints', value: ints }];
  return [name, { kind: 'other' }];
}

function readNode(reader: Reader): OnnxNode {
  const node: OnnxNode = { opType: '', domain: '', inputs: [], outputs: [], attributes: new Map() };
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1) node.inputs.push(reader.string());
    else if (field === 2) node.outputs.push(reader.string());
    else if (field === 4) node.opType = reader.string();
    else if (field === 7) node.domain = reader.string();
    else if (field === 5) {
      const [name, attribute] = readAttribute(reader.sub());
      node.attributes.set(name, attribute);
    } else reader.skip(wire);
  }
  return node;
}

/** `ValueInfoProto.name`. */
function readValueName(reader: Reader): string {
  let name = '';
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1) name = reader.string();
    else reader.skip(wire);
  }
  return name;
}

function readGraph(reader: Reader, opset: number): OnnxGraph {
  const graph: OnnxGraph = { nodes: [], initializers: new Map(), inputs: [], outputs: [], opset };
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1) {
      graph.nodes.push(readNode(reader.sub()));
      if (graph.nodes.length > LIMITS.nodes) throw new Error(`ONNX graph holds over ${LIMITS.nodes} nodes.`);
    } else if (field === 5) {
      const tensor = readTensor(reader.sub());
      graph.initializers.set(tensor.name, tensor);
    } else if (field === 11) graph.inputs.push(readValueName(reader.sub()));
    else if (field === 12) graph.outputs.push(readValueName(reader.sub()));
    else if (field === 15) throw new Error('ONNX graph holds a sparse initializer.');
    else reader.skip(wire);
  }
  // An initializer may be listed among the inputs too (IR < 4): it is a constant.
  graph.inputs = graph.inputs.filter((name) => !graph.initializers.has(name));
  return graph;
}

/** The graph of the ONNX model in *bytes*. */
export function readOnnxGraph(bytes: ArrayBuffer | Uint8Array): OnnxGraph {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (data.length > LIMITS.bytes) throw new Error(`ONNX graph is ${data.length} bytes, over ${LIMITS.bytes}.`);
  const reader = new Reader(data);
  let graph: Reader | null = null;
  let opset = 0;
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 7) graph = reader.sub();
    else if (field === 8) {
      // OperatorSetIdProto: domain 1, version 2.
      const entry = reader.sub();
      let domain = '';
      let version = 0;
      while (!entry.done) {
        const tag = entry.tag();
        if (tag.field === 1) domain = entry.string();
        else if (tag.field === 2) version = entry.int64();
        else entry.skip(tag.wire);
      }
      if (domain === '' || domain === 'ai.onnx') opset = version;
    } else reader.skip(wire);
  }
  if (!graph) throw new Error('ONNX model holds no graph.');
  return readGraph(graph, opset);
}
