/**
 * An actuator's traced torque law, run synchronously every physics substep.
 *
 * ONNX Runtime Web answers with a Promise, which has no place in the substep loop, so the
 * law runs here instead: elementwise operators over per-joint vectors and scalars,
 * compiled once into steps over preallocated arrays, so a run allocates nothing. Values
 * are float32 as ONNX's are, each step's result rounded to float32; a bool is 0 or 1.
 */

import { DATA_TYPE, readOnnxGraph, type OnnxAttribute, type OnnxGraph, type OnnxNode, type OnnxTensor } from './onnx';

/** What a law reads per joint, in mjlab's `ActuatorCmd` order, and what it writes. */
export const ACTUATOR_INPUTS = ['pos', 'vel', 'pos_target', 'vel_target', 'effort_target'] as const;
export type ActuatorInput = (typeof ACTUATOR_INPUTS)[number];
export const ACTUATOR_OUTPUT = 'torque';

type Unary = (x: number) => number;
type Binary = (a: number, b: number) => number;

const UNARY: Record<string, Unary> = {
  Abs: Math.abs,
  Exp: Math.exp,
  Identity: (x) => x,
  Neg: (x) => -x,
  Not: (x) => (x !== 0 ? 0 : 1),
  Sign: (x) => (x > 0 ? 1 : x < 0 ? -1 : x === 0 ? 0 : NaN),
  Sqrt: Math.sqrt,
  Tanh: Math.tanh,
};

const BINARY: Record<string, Binary> = {
  Add: (a, b) => a + b,
  And: (a, b) => (a !== 0 && b !== 0 ? 1 : 0),
  Div: (a, b) => a / b,
  Equal: (a, b) => (a === b ? 1 : 0),
  Greater: (a, b) => (a > b ? 1 : 0),
  GreaterOrEqual: (a, b) => (a >= b ? 1 : 0),
  Less: (a, b) => (a < b ? 1 : 0),
  LessOrEqual: (a, b) => (a <= b ? 1 : 0),
  Mul: (a, b) => a * b,
  Or: (a, b) => (a !== 0 || b !== 0 ? 1 : 0),
  Pow: Math.pow,
  Sub: (a, b) => a - b,
};

const VARIADIC: Record<string, Binary> = { Max: Math.max, Min: Math.min };

/** The operators this evaluator runs: `compile/actuator.py`'s `SUPPORTED_OPS`. */
export const SUPPORTED_OPS: ReadonlySet<string> = new Set([
  ...Object.keys(UNARY),
  ...Object.keys(BINARY),
  ...Object.keys(VARIADIC),
  'Cast',
  'Clip',
  'Constant',
  'Where',
]);

type Step = () => void;

/** A step's arguments: an absent optional input is `null`. */
type Args = Array<Float32Array | null>;

export class ActuatorGraph {
  /** Fill these before `run`, one value per joint in the law's column order. */
  readonly inputs: Record<ActuatorInput, Float32Array>;
  /** The torque per joint, rewritten by each `run`. */
  readonly output: Float32Array;

  /** Built by `fromOnnx`. */
  constructor(
    inputs: Record<ActuatorInput, Float32Array>,
    output: Float32Array,
    private readonly steps: readonly Step[],
    private readonly result: Float32Array,
  ) {
    this.inputs = inputs;
    this.output = output;
  }

  /** Compile the law in *bytes* over *width* joints; *label* names it in an error. */
  static fromOnnx(bytes: ArrayBuffer | Uint8Array, width: number, label: string): ActuatorGraph {
    let graph: OnnxGraph;
    try {
      graph = readOnnxGraph(bytes);
    } catch (error) {
      throw new Error(`Actuator "${label}": ${error instanceof Error ? error.message : String(error)}`);
    }
    return compile(graph, width, label);
  }

  run(): Float32Array {
    for (const step of this.steps) step();
    if (this.result.length === 1) this.output.fill(this.result[0]);
    else this.output.set(this.result);
    return this.output;
  }
}

function compile(graph: OnnxGraph, width: number, label: string): ActuatorGraph {
  const fail = (why: string) => new Error(`Actuator "${label}": ${why}`);
  if (!Number.isInteger(width) || width < 1) throw fail(`it drives ${width} joints.`);

  const inputs = Object.fromEntries(ACTUATOR_INPUTS.map((name) => [name, new Float32Array(width)])) as Record<
    ActuatorInput,
    Float32Array
  >;
  const values = new Map<string, Float32Array>();
  for (const name of graph.inputs) {
    if (!(ACTUATOR_INPUTS as readonly string[]).includes(name)) {
      throw fail(`its graph reads "${name}", which is none of ${ACTUATOR_INPUTS.join(', ')}.`);
    }
    values.set(name, inputs[name as ActuatorInput]);
  }
  const constant = (tensor: OnnxTensor, what: string): Float32Array => {
    const count = tensor.values.length;
    const vector = count === width && tensor.dims.filter((d) => d !== 1).length <= 1 && tensor.dims[tensor.dims.length - 1] === width;
    if (count !== 1 && !vector) {
      throw fail(`${what} has shape [${tensor.dims.join(', ')}], neither a scalar nor one value per joint (${width}).`);
    }
    return Float32Array.from(tensor.values);
  };
  for (const [name, tensor] of graph.initializers) values.set(name, constant(tensor, `initializer "${name}"`));

  const steps: Step[] = [];
  for (const node of graph.nodes) {
    if (node.domain !== '' && node.domain !== 'ai.onnx') throw fail(`its graph uses operator domain "${node.domain}".`);
    if (!SUPPORTED_OPS.has(node.opType)) {
      throw fail(`its graph uses ${node.opType}, which an actuator law may not: it runs elementwise operators only.`);
    }
    if (node.outputs.length !== 1 || !node.outputs[0]) throw fail(`its ${node.opType} node has ${node.outputs.length} outputs.`);
    if (values.has(node.outputs[0])) throw fail(`its graph writes "${node.outputs[0]}" twice.`);
    const args: Args = node.inputs.map((name) => {
      if (name === '') return null;
      const value = values.get(name);
      if (!value) throw fail(`its ${node.opType} node reads "${name}" before anything writes it.`);
      return value;
    });
    if (node.opType === 'Constant') {
      values.set(node.outputs[0], constant(constantTensor(node, fail), `a Constant`));
      continue;
    }
    const size = args.some((a) => a !== null && a.length !== 1) ? width : 1;
    const out = new Float32Array(size);
    steps.push(step(node, args, out, fail));
    values.set(node.outputs[0], out);
  }

  if (graph.outputs.length !== 1 || graph.outputs[0] !== ACTUATOR_OUTPUT) {
    throw fail(`its graph writes ${JSON.stringify(graph.outputs)}, not one "${ACTUATOR_OUTPUT}".`);
  }
  const result = values.get(ACTUATOR_OUTPUT);
  if (!result) throw fail(`its graph never writes "${ACTUATOR_OUTPUT}".`);
  return new ActuatorGraph(inputs, new Float32Array(width), steps, result);
}

function constantTensor(node: OnnxNode, fail: (why: string) => Error): OnnxTensor {
  const value = node.attributes.get('value');
  if (value?.kind === 'tensor') return value.value;
  const single = node.attributes.get('value_float');
  if (single?.kind === 'float') return { name: '', dims: [], dataType: DATA_TYPE.FLOAT, values: Float32Array.of(single.value) };
  const many = node.attributes.get('value_floats');
  if (many?.kind === 'floats') {
    return { name: '', dims: [many.value.length], dataType: DATA_TYPE.FLOAT, values: Float32Array.from(many.value) };
  }
  throw fail('its graph holds a Constant of a kind other than a float or bool tensor.');
}

function attributeNumber(attribute: OnnxAttribute | undefined): number | undefined {
  return attribute?.kind === 'float' || attribute?.kind === 'int' ? attribute.value : undefined;
}

/** One node as a closure over its arguments' arrays; a scalar argument is read at 0. */
function step(node: OnnxNode, args: Args, out: Float32Array, fail: (why: string) => Error): Step {
  const n = out.length;
  const op = node.opType;
  const malformed = (takes: string) => fail(`its ${op} node reads ${JSON.stringify(node.inputs)}, where ${op} takes ${takes}.`);
  const arity = (count: number) => {
    if (args.length !== count || args.some((a) => a === null)) throw malformed(`${count}`);
  };
  const stride = (a: Float32Array) => (a.length === 1 ? 0 : 1);

  if (op in UNARY) {
    arity(1);
    const f = UNARY[op];
    const a = args[0]!;
    const sa = stride(a);
    return () => {
      for (let i = 0; i < n; i++) out[i] = f(a[i * sa]);
    };
  }
  if (op in BINARY) {
    arity(2);
    const f = BINARY[op];
    const [a, b] = args as Float32Array[];
    const [sa, sb] = [stride(a), stride(b)];
    return () => {
      for (let i = 0; i < n; i++) out[i] = f(a[i * sa], b[i * sb]);
    };
  }
  if (op in VARIADIC) {
    if (args.length === 0 || args.some((a) => a === null)) throw malformed('one or more');
    const f = VARIADIC[op];
    const all = args as Float32Array[];
    const strides = all.map(stride);
    return () => {
      for (let i = 0; i < n; i++) {
        let value = all[0][i * strides[0]];
        for (let k = 1; k < all.length; k++) value = f(value, all[k][i * strides[k]]);
        out[i] = value;
      }
    };
  }
  if (op === 'Where') {
    arity(3);
    const [c, a, b] = args as Float32Array[];
    const [sc, sa, sb] = [stride(c), stride(a), stride(b)];
    return () => {
      for (let i = 0; i < n; i++) out[i] = c[i * sc] !== 0 ? a[i * sa] : b[i * sb];
    };
  }
  if (op === 'Clip') {
    if (args.length < 1 || args.length > 3 || args[0] === null) throw malformed('1 to 3');
    const x = args[0];
    // Opset 11 on takes the bounds as inputs; earlier ones as attributes.
    const lo = args[1] ?? Float32Array.of(attributeNumber(node.attributes.get('min')) ?? -Infinity);
    const hi = args[2] ?? Float32Array.of(attributeNumber(node.attributes.get('max')) ?? Infinity);
    const [sx, sl, sh] = [stride(x), stride(lo), stride(hi)];
    return () => {
      for (let i = 0; i < n; i++) out[i] = Math.min(Math.max(x[i * sx], lo[i * sl]), hi[i * sh]);
    };
  }
  if (op === 'Cast') {
    arity(1);
    const to = attributeNumber(node.attributes.get('to'));
    const a = args[0]!;
    const sa = stride(a);
    if (to === DATA_TYPE.FLOAT) {
      return () => {
        for (let i = 0; i < n; i++) out[i] = a[i * sa];
      };
    }
    if (to === DATA_TYPE.BOOL) {
      return () => {
        for (let i = 0; i < n; i++) out[i] = a[i * sa] !== 0 ? 1 : 0;
      };
    }
    throw fail(`its graph casts to data type ${to}; an actuator law uses float and bool.`);
  }
  throw fail(`its graph uses ${op}, which an actuator law may not.`);
}
