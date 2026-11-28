/**
 * The evaluator is the only thing turning an actuator's targets into torque in the
 * browser, every substep: each operator has to compute what ONNX defines, in float32,
 * and a graph it cannot run has to be refused when it loads rather than mid-episode.
 */
import { describe, expect, it } from 'vitest';

import { ACTUATOR_INPUTS, ActuatorGraph, SUPPORTED_OPS, type ActuatorInput } from '../evaluator';
import { DATA_TYPE, onnxModel, type ModelSpec, type TensorSpec } from './onnxWriter';

const f32 = Math.fround;

function law(spec: ModelSpec, width = 3): ActuatorGraph {
  return ActuatorGraph.fromOnnx(onnxModel(spec), width, 'law');
}

function run(graph: ActuatorGraph, inputs: Partial<Record<ActuatorInput, number[]>>): number[] {
  for (const [name, values] of Object.entries(inputs)) graph.inputs[name as ActuatorInput].set(values);
  return [...graph.run()];
}

const scalar = (name: string, value: number): TensorSpec => ({ name, dims: [], values: [value] });

describe('ActuatorGraph: a PD law', () => {
  // IdealPdActuator's law as the tracer writes it: per-joint gains, a scalar limit.
  const pd = law({
    initializers: [
      { name: 'kp', dims: [1, 3], values: [20, 30, 40] },
      { name: 'kd', dims: [3], values: [1, 2, 3] },
      scalar('lim', 10),
    ],
    nodes: [
      { op: 'Sub', inputs: ['pos_target', 'pos'], outputs: ['e'] },
      { op: 'Mul', inputs: ['kp', 'e'], outputs: ['pe'] },
      { op: 'Sub', inputs: ['vel_target', 'vel'], outputs: ['ve'] },
      { op: 'Mul', inputs: ['kd', 've'], outputs: ['de'] },
      { op: 'Add', inputs: ['pe', 'de'], outputs: ['s'] },
      { op: 'Add', inputs: ['s', 'effort_target'], outputs: ['u'] },
      { op: 'Neg', inputs: ['lim'], outputs: ['lo'] },
      { op: 'Max', inputs: ['u', 'lo'], outputs: ['clipped'] },
      { op: 'Min', inputs: ['clipped', 'lim'], outputs: ['torque'] },
    ],
  });

  it('is kp * (target - pos) + kd * (vel target - vel) + effort, clipped', () => {
    const torque = run(pd, {
      pos: [0.1, -0.2, 0.3],
      vel: [0.5, 0, -1],
      pos_target: [0.2, 0.1, 0.3],
      vel_target: [0, 0, 0],
      effort_target: [0.25, -0.5, 0],
    });
    const kp = [20, 30, 40];
    const kd = [1, 2, 3];
    const expected = [0, 1, 2].map((i) => {
      const e = f32(f32([0.2, 0.1, 0.3][i]) - f32([0.1, -0.2, 0.3][i]));
      const u = f32(f32(f32(kp[i] * e) + f32(kd[i] * -[0.5, 0, -1][i])) + [0.25, -0.5, 0][i]);
      return Math.min(Math.max(u, -10), 10);
    });
    expect(torque).toEqual(expected);
  });

  it('clips at the limit on either side', () => {
    const still = { pos: [0, 0, 0], vel: [0, 0, 0], vel_target: [0, 0, 0], effort_target: [0, 0, 0] };
    expect(run(pd, { ...still, pos_target: [1, -1, 0.1] })).toEqual([
      10,
      -10,
      f32(40 * f32(0.1)),
    ]);
  });

  it('rewrites the same output array each run', () => {
    expect(pd.run()).toBe(pd.output);
    expect(pd.run()).toBe(pd.output);
  });
});

describe('ActuatorGraph: operators', () => {
  const xs = [-1.5, 0, 2.25];

  it.each<[string, (x: number) => number]>([
    ['Abs', Math.abs],
    ['Exp', Math.exp],
    ['Identity', (x) => x],
    ['Neg', (x) => -x],
    ['Not', (x) => (x !== 0 ? 0 : 1)],
    ['Sign', Math.sign],
    ['Sqrt', Math.sqrt],
    ['Tanh', Math.tanh],
  ])('%s', (op, reference) => {
    const graph = law({ nodes: [{ op, inputs: ['pos'], outputs: ['torque'] }] });
    expect(run(graph, { pos: xs })).toEqual(xs.map((x) => f32(reference(x))));
  });

  it.each<[string, (a: number, b: number) => number]>([
    ['Add', (a, b) => a + b],
    ['And', (a, b) => Number(a !== 0 && b !== 0)],
    ['Div', (a, b) => a / b],
    ['Equal', (a, b) => Number(a === b)],
    ['Greater', (a, b) => Number(a > b)],
    ['GreaterOrEqual', (a, b) => Number(a >= b)],
    ['Less', (a, b) => Number(a < b)],
    ['LessOrEqual', (a, b) => Number(a <= b)],
    ['Mul', (a, b) => a * b],
    ['Or', (a, b) => Number(a !== 0 || b !== 0)],
    ['Pow', Math.pow],
    ['Sub', (a, b) => a - b],
  ])('%s, with a scalar on either side', (op, reference) => {
    const operands = (inputs: string[]) =>
      law({ initializers: [scalar('c', 2.25)], nodes: [{ op, inputs, outputs: ['torque'] }] });
    expect(run(operands(['pos', 'c']), { pos: xs })).toEqual(xs.map((x) => f32(reference(x, 2.25))));
    expect(run(operands(['c', 'pos']), { pos: xs })).toEqual(xs.map((x) => f32(reference(2.25, x))));
    expect(run(operands(['pos', 'vel']), { pos: xs, vel: [2.25, 0, -1.5] })).toEqual(
      xs.map((x, i) => f32(reference(x, [2.25, 0, -1.5][i]))),
    );
  });

  it('Max and Min over any number of inputs', () => {
    const extreme = (op: string) =>
      law({
        initializers: [scalar('c', 0.5)],
        nodes: [{ op, inputs: ['pos', 'vel', 'c'], outputs: ['torque'] }],
      });
    const inputs = { pos: [-1, 2, 0.25], vel: [3, -2, 0] };
    expect(run(extreme('Max'), inputs)).toEqual([3, 2, 0.5]);
    expect(run(extreme('Min'), inputs)).toEqual([-1, -2, 0]);
  });

  it('Where picks per joint, a comparison deciding', () => {
    const graph = law({
      initializers: [scalar('zero', 0)],
      nodes: [
        { op: 'Greater', inputs: ['vel', 'zero'], outputs: ['moving'] },
        { op: 'Where', inputs: ['moving', 'pos', 'pos_target'], outputs: ['torque'] },
      ],
    });
    expect(run(graph, { vel: [1, 0, -1], pos: [10, 20, 30], pos_target: [-10, -20, -30] })).toEqual([10, -20, -30]);
  });

  it('Cast to bool and back to float', () => {
    const graph = law({
      nodes: [
        { op: 'Cast', inputs: ['pos'], outputs: ['b'], attributes: [{ name: 'to', int: DATA_TYPE.BOOL }] },
        { op: 'Cast', inputs: ['b'], outputs: ['torque'], attributes: [{ name: 'to', int: DATA_TYPE.FLOAT }] },
      ],
    });
    expect(run(graph, { pos: [-0.5, 0, 3] })).toEqual([1, 0, 1]);
  });

  it('Clip with bounds as inputs, as attributes, or absent', () => {
    const clip = (inputs: string[], attributes = [] as { name: string; float: number }[], opset = 17) =>
      law({
        opset,
        initializers: [scalar('lo', -1), { name: 'hi', dims: [3], values: [1, 2, 3] }],
        nodes: [{ op: 'Clip', inputs, outputs: ['torque'], attributes }],
      });
    const pos = [-5, 2.5, 5];
    expect(run(clip(['pos', 'lo', 'hi']), { pos })).toEqual([-1, 2, 3]);
    expect(run(clip(['pos', '', 'hi']), { pos })).toEqual([-5, 2, 3]);
    expect(run(clip(['pos', 'lo']), { pos })).toEqual([-1, 2.5, 5]);
    expect(run(clip(['pos']), { pos })).toEqual(pos);
    const attributes = [
      { name: 'min', float: -2 },
      { name: 'max', float: 2 },
    ];
    expect(run(clip(['pos'], attributes, 6), { pos })).toEqual([-2, 2, 2]);
  });

  it('a Constant of a tensor, a float or floats', () => {
    const graph = law({
      nodes: [
        { op: 'Constant', inputs: [], outputs: ['a'], attributes: [{ name: 'value', tensor: { dims: [3], values: [1, 2, 3] } }] },
        { op: 'Constant', inputs: [], outputs: ['b'], attributes: [{ name: 'value_float', float: 0.5 }] },
        { op: 'Constant', inputs: [], outputs: ['c'], attributes: [{ name: 'value_floats', floats: [10, 20, 30] }] },
        { op: 'Mul', inputs: ['a', 'b'], outputs: ['ab'] },
        { op: 'Add', inputs: ['ab', 'c'], outputs: ['torque'] },
      ],
    });
    expect(run(graph, {})).toEqual([10.5, 21, 31.5]);
  });

  it('gives every joint a law that reads no per-joint value', () => {
    const graph = law({
      initializers: [scalar('c', 0.75)],
      nodes: [{ op: 'Neg', inputs: ['c'], outputs: ['torque'] }],
    });
    expect(run(graph, {})).toEqual([-0.75, -0.75, -0.75]);
  });

  it('rounds each step to float32, as ONNX does', () => {
    // At 2^24 a float32 steps by 2: 1.5 is lost, where float64 would keep it.
    const graph = law({
      initializers: [scalar('big', 2 ** 24)],
      nodes: [
        { op: 'Add', inputs: ['pos', 'big'], outputs: ['s'] },
        { op: 'Sub', inputs: ['s', 'big'], outputs: ['torque'] },
      ],
    });
    expect(run(graph, { pos: [1.5, 1.5, 1.5] })).toEqual([2, 2, 2]);
  });

  it('runs the operators compile/actuator.py allows, and only those', () => {
    expect([...SUPPORTED_OPS].sort()).toEqual(
      [
        'Abs', 'Add', 'And', 'Cast', 'Clip', 'Constant', 'Div', 'Equal', 'Exp', 'Greater', 'GreaterOrEqual',
        'Identity', 'Less', 'LessOrEqual', 'Max', 'Min', 'Mul', 'Neg', 'Not', 'Or', 'Pow', 'Sign', 'Sqrt', 'Sub',
        'Tanh', 'Where',
      ].sort(),
    );
    expect(ACTUATOR_INPUTS).toEqual(['pos', 'vel', 'pos_target', 'vel_target', 'effort_target']);
  });
});

describe('ActuatorGraph: refusals', () => {
  const ADD = { op: 'Add', inputs: ['pos', 'vel'], outputs: ['torque'] };

  it.each<[string, ModelSpec, RegExp]>([
    ['an input it does not feed', { inputs: ['pos', 'force'], nodes: [ADD] }, /reads "force", which is none of pos, vel/],
    [
      'a constant neither scalar nor per joint',
      { initializers: [{ name: 'kp', dims: [2], values: [1, 2] }], nodes: [ADD] },
      /initializer "kp" has shape \[2\], neither a scalar nor one value per joint \(3\)/,
    ],
    [
      'a column of values',
      { initializers: [{ name: 'kp', dims: [3, 1], values: [1, 2, 3] }], nodes: [ADD] },
      /shape \[3, 1\]/,
    ],
    [
      'a matrix product',
      { nodes: [{ op: 'MatMul', inputs: ['pos', 'vel'], outputs: ['torque'] }] },
      /uses MatMul, which an actuator law may not: it runs elementwise operators only/,
    ],
    ['another domain', { nodes: [{ ...ADD, domain: 'com.microsoft' }] }, /operator domain "com.microsoft"/],
    ['a second output', { outputs: ['torque', 'extra'], nodes: [ADD] }, /writes \["torque","extra"\], not one "torque"/],
    ['an output of another name', { outputs: ['effort'], nodes: [{ ...ADD, outputs: ['effort'] }] }, /not one "torque"/],
    ['an output nothing writes', { nodes: [{ ...ADD, outputs: ['sum'] }] }, /never writes "torque"/],
    [
      'a value read before it is written',
      { nodes: [{ op: 'Add', inputs: ['pos', 'later'], outputs: ['torque'] }] },
      /reads "later" before anything writes it/,
    ],
    [
      'a value written twice',
      { nodes: [ADD, ADD] },
      /writes "torque" twice/,
    ],
    ['an input overwritten', { nodes: [{ ...ADD, outputs: ['pos'] }] }, /writes "pos" twice/],
    [
      'a node of two outputs',
      { nodes: [{ op: 'Add', inputs: ['pos', 'vel'], outputs: ['torque', 'x'] }] },
      /Add node has 2 outputs/,
    ],
    [
      'a binary operator given one input',
      { nodes: [{ op: 'Sub', inputs: ['pos'], outputs: ['torque'] }] },
      /Sub node reads \["pos"\], where Sub takes 2/,
    ],
    [
      'Clip given four inputs',
      { nodes: [{ op: 'Clip', inputs: ['pos', 'vel', 'vel', 'vel'], outputs: ['torque'] }] },
      /where Clip takes 1 to 3/,
    ],
    [
      'a cast to int64',
      { nodes: [{ op: 'Cast', inputs: ['pos'], outputs: ['torque'], attributes: [{ name: 'to', int: DATA_TYPE.INT64 }] }] },
      /casts to data type 7/,
    ],
    [
      'a Constant of a string',
      { nodes: [{ op: 'Constant', inputs: [], outputs: ['torque'], attributes: [{ name: 'value_string', string: 'x' }] }] },
      /a Constant of a kind other than a float or bool tensor/,
    ],
  ])('%s', (_, spec, message) => {
    expect(() => law(spec)).toThrow(message);
  });

  it('names the law in every error, a reader error included', () => {
    expect(() => ActuatorGraph.fromOnnx(new Uint8Array([0x08, 0x01]), 3, 'DcMotorActuator_0')).toThrow(
      /^Actuator "DcMotorActuator_0": ONNX model holds no graph\.$/,
    );
    expect(() => law({ nodes: [{ op: 'MatMul', inputs: ['pos', 'vel'], outputs: ['torque'] }] })).toThrow(/^Actuator "law": /);
  });

  it('refuses a law over no joints', () => {
    expect(() => law({ nodes: [ADD] }, 0)).toThrow(/drives 0 joints/);
  });
});
