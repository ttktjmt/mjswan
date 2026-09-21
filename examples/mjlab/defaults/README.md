# mjlab Default Tasks

Source: https://github.com/mujocolab/mjlab

Demo: https://mjswan-mjlab.pages.dev/

Every MDP term here reaches the browser as a traced ONNX graph — no hand-written
TypeScript, so the build reports `uses_custom_js: false`. Two things still need
Python, and both now live in mjswan rather than beside this script:

- `mjswan.mjlab.register_custom_terminations` injects the terrain generator's
  `limit_x`/`limit_y`/`half_x`/`half_y` into mjlab's own termination params, because
  those constants live on the generator rather than on the function.
- `mjswan.mjlab.bindings` registers what each mjlab command cfg traces as — a
  debug-vis marker for `LiftingCommandCfg` and the RSI reset graph for
  `MotionCommandCfg` (whose clip lookup stays native). Importing it is deliberate:
  it is the one module under `mjswan.mjlab` that needs mjlab and torch at import
  time, which is what keeps the rest of the package's mjlab dependency soft.
