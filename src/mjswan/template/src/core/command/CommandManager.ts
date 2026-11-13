import { TrackingCommand } from './TrackingCommand';
import { OnnxCommand, type OnnxCommandConfig } from './OnnxCommand';
import {
  getCommandInputId,
  type CheckboxCommandConfig,
  type CommandConfigEntry,
  type CommandDefinition,
  type CommandEvent,
  type CommandEventListener,
  type CommandInputConfig,
  type CommandTerm,
  type CommandTermConstructor,
  type CommandTermContext,
  type CommandsConfig,
  type SliderCommandConfig,
} from './types';

/** True for a config entry naming the shared `OnnxCommand` handler. */
function isOnnxCommandConfig(entry: CommandConfigEntry): entry is OnnxCommandConfig {
  return entry.name === 'OnnxCommand';
}

type ValueCommandConfig = SliderCommandConfig | CheckboxCommandConfig;

class UiCommand implements CommandTerm {
  private readonly inputs: CommandInputConfig[];
  private readonly values: Map<string, number>;

  constructor(
    _termName: string,
    config: CommandConfigEntry,
    // _context: CommandTermContext
  ) {
    this.inputs = Array.isArray(config.ui?.inputs) ? config.ui.inputs : [];
    this.values = new Map();
    for (const input of this.inputs) {
      if (input.type === 'slider') {
        this.values.set(input.name, input.default);
      } else if (input.type === 'checkbox') {
        this.values.set(input.name, input.default ? 1.0 : 0.0);
      }
    }
  }

  getCommand(): Float32Array {
    const valueInputs = this.inputs.filter(
      (input): input is ValueCommandConfig => input.type === 'slider' || input.type === 'checkbox'
    );
    const values = new Float32Array(valueInputs.length);
    for (let i = 0; i < valueInputs.length; i++) {
      const input = valueInputs[i];
      const fallback = input.type === 'checkbox' ? (input.default ? 1.0 : 0.0) : input.default;
      values[i] = this.values.get(input.name) ?? fallback ?? 0.0;
    }
    return values;
  }

  getUiConfig() {
    return { inputs: this.inputs };
  }

  /** The UI value as a `{command, field: 'command'}` slot; browser-only, so it binds here. */
  getStateField(field: string): Float32Array | null {
    return field === 'command' ? this.getCommand() : null;
  }

  reset(): void {
    for (const input of this.inputs) {
      if (input.type === 'slider') {
        this.values.set(input.name, input.default);
      } else if (input.type === 'checkbox') {
        this.values.set(input.name, input.default ? 1.0 : 0.0);
      }
    }
  }

  getUiValue(inputName: string): number | undefined {
    return this.values.get(inputName);
  }

  setValue(inputName: string, value: number): number {
    const input = this.inputs.find(
      (entry): entry is SliderCommandConfig | CheckboxCommandConfig =>
        (entry.type === 'slider' || entry.type === 'checkbox') && entry.name === inputName
    );
    if (!input) {
      return 0.0;
    }
    if (input.type === 'checkbox') {
      const normalized = value >= 0.5 ? 1.0 : 0.0;
      this.values.set(input.name, normalized);
      return normalized;
    }
    const clamped = Math.max(input.min, Math.min(input.max, value));
    this.values.set(input.name, clamped);
    return clamped;
  }

  /** `zero` resets every slider, as mjlab's own Zero button does. */
  triggerButton(inputName: string): boolean {
    if (inputName !== 'zero') return false;
    for (const input of this.inputs) {
      if (input.type === 'slider') this.values.set(input.name, 0);
    }
    return true;
  }
}

const BuiltinCommandTerms: Record<string, CommandTermConstructor> = {
  UiCommand,
  TrackingCommand,
};

export class CommandManager {
  private terms: Map<string, CommandTerm> = new Map();
  private commands: Map<string, CommandDefinition> = new Map();
  private commandGroups: Map<string, string[]> = new Map();
  private values: Map<string, number> = new Map();
  private listeners: Set<CommandEventListener> = new Set();
  private context: CommandTermContext | null = null;
  /** Buttons already reported as unhandled, so a repeated press is not a repeated log. */
  private warnedButtons: Set<string> = new Set();

  initialize(
    commandsConfig: CommandsConfig,
    context: CommandTermContext,
    pluginCommands?: Record<string, CommandTermConstructor>
  ): void {
    this.clear();
    this.context = context;
    const registry: Record<string, CommandTermConstructor> = {
      ...BuiltinCommandTerms,
      ...pluginCommands,
    };

    for (const [groupName, entry] of Object.entries(commandsConfig)) {
      if (isOnnxCommandConfig(entry)) {
        const term = this.buildOnnxCommand(groupName, entry, context);
        if (!term) continue;
        this.terms.set(groupName, term);
        this.registerUi(groupName, term);
        continue;
      }
      const Term = registry[entry.name];
      if (!Term) {
        throw new Error(`Unknown command term: ${entry.name}`);
      }
      const term = new Term(groupName, entry, context);
      this.terms.set(groupName, term);
      this.registerUi(groupName, term);
    }
  }

  /**
   * `OnnxCommand` bypasses the class registry: one shared handler needing a session and
   * rng that `new Term(name, config, context)` has no room for. Warns and skips, so one
   * missing session spares the others — and the skipped term leaves `termNames()`, so an
   * observation reading it fails to bind rather than reading zeros.
   */
  private buildOnnxCommand(
    groupName: string,
    entry: OnnxCommandConfig,
    context: CommandTermContext
  ): OnnxCommand | null {
    const session = context.onnxSessions?.get(entry.onnx);
    if (!session || !context.rng) {
      console.warn(
        `[CommandManager] OnnxCommand "${groupName}" needs onnxSessions/rng in context; skipping.`
      );
      return null;
    }
    return new OnnxCommand(groupName, entry, context, {
      session,
      rng: context.rng,
      readSlot: context.readOnnxSlot,
    });
  }

  update(dt: number, afterReset = false): void {
    for (const term of this.terms.values()) {
      term.update?.(dt, afterReset);
    }
  }

  /** Resolves once every term's last `update` has landed. */
  async settled(): Promise<void> {
    await Promise.all([...this.terms.values()].map((term) => term.settled?.()));
  }

  /** `shown` is the viewer's Debug Viz "Enabled". */
  updateDebugVisuals(shown = true): void {
    for (const term of this.terms.values()) {
      term.updateDebugVisuals?.(shown);
    }
  }

  /** The terms offering a debug drawing, as mjlab's `create_debug_vis_gui` lists them. */
  getDebugVisTerms(): Array<{ name: string; enabled: boolean }> {
    const out: Array<{ name: string; enabled: boolean }> = [];
    for (const [name, term] of this.terms) {
      const enabled = term.debugVisEnabled?.();
      if (enabled != null) out.push({ name, enabled });
    }
    return out;
  }

  setDebugVisEnabled(name: string, enabled: boolean): void {
    const term = this.terms.get(name);
    if (!term?.setDebugVisEnabled) return;
    term.setDebugVisEnabled(enabled);
    this.emit({ type: 'debug_vis', commandId: name, groupName: name });
  }

  /**
   * Reset every term **in config order**, awaiting each: a reset is a resample that may
   * write to the sim, and overlaps must resolve last-writer-wins as mjlab's do.
   */
  async resetTerms(): Promise<void> {
    for (const term of this.terms.values()) {
      await term.reset?.();
    }
    this.syncValuesFromTerms();
    this.emit({ type: 'reset', commandId: '*' });
  }

  getCommandGroups(): string[] {
    return Array.from(this.commandGroups.keys());
  }

  getCommandsInGroup(groupName: string): CommandDefinition[] {
    const ids = this.commandGroups.get(groupName) ?? [];
    return ids.map(id => this.commands.get(id)!).filter(Boolean);
  }

  getCommands(): CommandDefinition[] {
    return Array.from(this.commands.values());
  }

  getCommandById(id: string): CommandDefinition | undefined {
    return this.commands.get(id);
  }

  getValue(id: string): number {
    return this.values.get(id) ?? 0;
  }

  getValues(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [id, value] of this.values) {
      result[id] = value;
    }
    return result;
  }

  /**
   * The named term's current vector, or an empty one. A consumer whose *config* names a
   * term must validate it against `termNames()` at construction: an unvalidated miss is
   * zero-padded, handing the policy a block of zeros it was never trained on.
   */
  getCommand(groupName: string): Float32Array {
    const term = this.terms.get(groupName);
    return term ? term.getCommand() : new Float32Array(0);
  }

  /** Names in registration order, for binding-time validation by a consumer. */
  termNames(): string[] {
    return Array.from(this.terms.keys());
  }

  getTerm(groupName: string): CommandTerm | undefined {
    return this.terms.get(groupName);
  }

  getContext(): CommandTermContext | null {
    return this.context;
  }

  setValue(id: string, value: number): void {
    const command = this.commands.get(id);
    if (!command || (command.config.type !== 'slider' && command.config.type !== 'checkbox')) {
      return;
    }
    const term = this.terms.get(command.groupName);
    const inputName = command.config.name;
    const clamped = term?.setValue ? term.setValue(inputName, value) : undefined;
    const nextValue = typeof clamped === 'number' ? clamped : value;
    this.values.set(id, nextValue);
    this.emit({
      type: 'change',
      commandId: id,
      groupName: command.groupName,
      value: nextValue,
    });
  }

  triggerButton(id: string): void {
    const command = this.commands.get(id);
    if (!command || command.config.type !== 'button') {
      return;
    }

    const term = this.terms.get(command.groupName);
    const handled = term?.triggerButton?.(command.config.name);
    // mjlab's Zero moves the term's own sliders, and the panel reads them from here.
    this.syncValuesFromTerms();
    if (handled === false && !this.warnedButtons.has(id)) {
      this.warnedButtons.add(id);
      console.warn(
        `[CommandManager] "${command.groupName}" has no action for button ` +
          `"${command.config.name}".`,
      );
    }

    this.emit({
      type: 'button',
      commandId: id,
      groupName: command.groupName,
    });
  }

  resetToDefaults(): Promise<void> {
    return this.resetTerms();
  }

  addEventListener(listener: CommandEventListener): void {
    this.listeners.add(listener);
  }

  removeEventListener(listener: CommandEventListener): void {
    this.listeners.delete(listener);
  }

  clear(): void {
    for (const term of this.terms.values()) {
      term.dispose?.();
    }
    this.terms.clear();
    this.commands.clear();
    this.commandGroups.clear();
    this.values.clear();
    this.warnedButtons.clear();
    this.context = null;
    this.emit({ type: 'clear', commandId: '' });
  }

  hasCommands(): boolean {
    return this.commands.size > 0;
  }

  dispose(): void {
    for (const term of this.terms.values()) {
      term.dispose?.();
    }
    this.terms.clear();
    this.commands.clear();
    this.commandGroups.clear();
    this.values.clear();
    this.listeners.clear();
    this.context = null;
  }

  private registerUi(groupName: string, term: CommandTerm): void {
    const ui = term.getUiConfig?.();
    const inputs = Array.isArray(ui?.inputs) ? ui.inputs : [];
    if (inputs.length === 0) {
      return;
    }
    this.commandGroups.set(groupName, []);
    for (const input of inputs) {
      const id = getCommandInputId(groupName, input.name);
      this.commands.set(id, { id, groupName, config: input });
      this.commandGroups.get(groupName)!.push(id);
      if (input.type === 'slider' || input.type === 'checkbox') {
        const fallback = input.type === 'checkbox' ? (input.default ? 1.0 : 0.0) : input.default;
        this.values.set(id, term.getUiValue?.(input.name) ?? fallback);
      }
    }
    this.emit({
      type: 'group_registered',
      commandId: groupName,
      groupName,
    });
  }

  /**
   * Re-read what the panel shows, after a reset moved the terms.
   *
   * By input name: a term's command vector is not its UI vector (an `OnnxCommand`'s is
   * the policy's), so pairing them by position showed one input's value under another.
   */
  private syncValuesFromTerms(): void {
    for (const [id, command] of this.commands) {
      if (command.config.type !== 'slider' && command.config.type !== 'checkbox') {
        continue;
      }
      const value = this.terms.get(command.groupName)?.getUiValue?.(command.config.name);
      if (value !== undefined) this.values.set(id, value);
    }
  }

  private emit(event: CommandEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.warn('[CommandManager] Listener error:', error);
      }
    }
  }
}
