# Operator overrides of MDP terms

> Status: **Proposed**. Adds `engine.mdp` to the headless engine's API (ADR 0004):
> terminations switched off and given a delay or a stop, and events switched off or
> triggered by hand, by whoever watches the simulation, with what each term did reported
> back. mjswan Cloud's player is the first host, under its Settings; the SPA's panel does
> not draw it.

## Context

A browser episode ends the step a termination fires, as mjlab's does, and the reset puts
the robot back before anyone sees why. A policy that falls in a second resets every second.
Someone who pushed the robot over to see how it recovers watches it snap back instead.
mjlab's own viewer has the same behaviour, but there the episode is a training artefact; in
the browser it is the thing on show.

The engine already lets an operator drive two kinds of event term: fire a `manual` one,
arm or disarm an `interval` one (`engine.events`). Nothing reaches a termination, a reset
event, or a delay, and nothing says when a term fired.

## Decision

### 1. Termination: Off, and Delay or Stop. Event: Off, Trigger

`engine.mdp.terminations.set(name, { enabled?, onFire? })` and
`engine.mdp.events.set(name, { enabled })` / `engine.mdp.events.trigger(name)`:

| | Off | What follows the firing (`onFire`) | Trigger |
|---|---|---|---|
| termination | still evaluated and reported, never ends the episode | `'reset'`: at once, mjlab's. `{ delay: d }` (up to 60 s): `d` after the condition turned true, whatever it does meanwhile. `'stop'`: the simulation pauses on that step | |
| reset event | no write at the reset; its `min_step_count_between_reset` gate is left as if it had not fired | | the write lands now, mid-episode |
| interval event | the schedule stops: the same switch as `engine.events.setArmed` | | the write lands now; the timer is left alone |
| manual event | no switch: nothing but a press fires it | | the same as `engine.events.fire` |

Startup events are not listed: they have run before an operator could reach them.

A termination takes a delay or a stop, never both: each answers "let me see what made it
fire", and together they would ask the viewer to reason about a countdown that a pause
interrupts. `onFire` is one value, so the exclusion holds by construction.

A held termination is latched at the step it fires; a condition that clears does not cancel
it, turning the term off does. A new delay re-times what is held from its firing. A stop
keeps the frame the term fired on and lands the reset it owes on `play`. Off and a stop
combine: the simulation pauses where the term fires, and plays on without a reset.

Event delays were considered and dropped: a reset write held into the episode, or an
interval write held past its timer, answers no question a viewer has.

### 2. One clock: episode time

Every time is the episode's, `episode_length_buf × timestep × decimation`, the counter the
traced graphs read. It stands still while paused and runs slower than the wall clock when
the browser falls behind, so a delay means the same simulated span whatever the machine.

### 3. Overrides are the viewer's, kept by name

An override lives in the engine, keyed by kind and term name, for the engine's life:
across resets, policy switches and model rebuilds, so turning `bad_orientation` off once
holds for every checkpoint of a run. A name the loaded MDP lacks is kept for the next one
that has it. `engine.mdp.clear()` returns to the MDP as built. Nothing reaches the document,
the URL or the manifest: the simulation published is the simulation mjlab defined.

Overrides change the trajectory, and an event switched off draws nothing from the term
PRNG, so a recorded `termSeed` replays a session only with the same overrides.

### 4. What the terms did, in the snapshot

`MjswanEngineState.mdp` carries, per term, the override and this episode's record:
a termination's `active` (its condition at the last step), `firedAt` (when it last turned
true) and `resetAt` (when the reset it holds lands); an event's `firedAt`. `pendingReset`
is the soonest held reset, `stoppedBy` the term whose stop paused the simulation, and
`activity` the last 32 things any term did, each with its episode, episode time and a
sequence number to key on.

The runtime notifies at most every 100 ms and never inside a step, and while a reset is
held it notifies every step so a host's countdown moves.

## Consequences

- A host can show which term ended an episode, keep an episode going past it, or freeze the
  frame it ended on. The SPA's panel keeps its Events section as it was; `setArmed` now
  persists across policy switches like any other override.
- The default is unchanged: with no override every verdict and every write lands when
  mjlab's would, which the rollout parity test holds.

## Other kinds, considered

mjlab has eight managers: observation, action, command, event and termination run in the
browser; reward, curriculum and metrics are training-time only and are left out.

`engine.mdp` stays with terminations and events. Two neighbours were weighed and belong
elsewhere, as features of mjswan rather than operations on a running MDP:

- **Observation noise and delay**, which the build accepts and drops today. Whether the
  browser should apply them as mjlab does (compute, noise, clip, scale, delay, history) is
  undecided.
- **A viewer-set actuator command delay**, over the one the browser already runs from the
  actuator config (`commandDelay.ts`, physics substeps).

A tracking command's motion seek, speed and stop were considered and rejected. Turning the
policy off is not an operation here either: deselecting it leaves position actuators on
keyframe 0's targets and a motor robot without torque.
