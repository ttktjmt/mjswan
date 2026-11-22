import type { ObservationConstructor } from './policy/PolicyRunner';
import type { TerminationConstructor } from './termination/terminations';
import type { EventConstructor } from './event/EventBase';
import type { CommandTermConstructor } from './command/types';

/**
 * Custom MDP term constructors handed to a pinned engine at load, registered
 * instance-scoped (no module globals). Trusted contexts only: mjswan Cloud rejects author
 * code. Observations, terminations and commands come from the policy; events from the
 * scene and the policy, the policy's winning a name both carry. See context/frontend.md.
 */
export interface EnginePlugins {
  observations?: Record<string, ObservationConstructor>;
  terminations?: Record<string, TerminationConstructor>;
  events?: Record<string, EventConstructor>;
  commands?: Record<string, CommandTermConstructor>;
}
