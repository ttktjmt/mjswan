/**
 * mjlab's Debug Viz folder: "Enabled" over one switch per drawing (command terms, then
 * raycast sensors, then reward terms). A drawing shows only while both are on. The
 * viewer's own switches follow, under "Enabled" alike.
 */

import type * as THREE from 'three';

import type { CommandManager } from '../command/CommandManager';
import { RaycastSensor } from '../onnx/raycast';
import { type DebugVisConfig, type Drawing, RaycastDrawing, UprightDrawing } from './drawings';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

export interface DebugVizEntry {
  /** `command:<term>`, `sensor:<name>`, `reward:<term>`, or a viewer switch's id. */
  id: string;
  /** As mjlab's checkbox reads: a command term capitalized, others as named. */
  label: string;
  enabled: boolean;
}

/**
 * A switch the viewer adds for something of its own. `apply` gets whether it shows, its
 * own switch and "Enabled" together, whenever either changes.
 */
export interface ViewerSwitch {
  label: string;
  enabled: boolean;
  apply(visible: boolean): void;
}

/** Python's `str.capitalize`, which mjlab labels a command term's checkbox with. */
export function capitalize(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
}

export class DebugViz {
  /** mjlab's "Enabled"; a viewer setting, so policy loads keep it. */
  private shown = true;
  private readonly drawings = new Map<string, { label: string; drawing: Drawing }>();
  private readonly casters: RaycastSensor[] = [];
  private readonly switches = new Map<string, ViewerSwitch>();

  constructor(
    private readonly commands: CommandManager,
    private readonly simulation: () => { mjModel: MjModel | null; mjData: MjData | null },
  ) {}

  /** Build the policy's sensor and reward drawings, replacing the previous policy's. */
  load(config: DebugVisConfig | undefined, mujoco: MainModule, root: THREE.Object3D): void {
    this.clear();
    if (!config) return;
    const casters = new Map<string, RaycastSensor>();
    for (const [name, sensor] of Object.entries(config.sensors ?? {})) {
      const caster = new RaycastSensor(mujoco, sensor);
      casters.set(name, caster);
      this.casters.push(caster);
      if (sensor.debug_vis) {
        this.drawings.set(`sensor:${name}`, {
          label: name,
          drawing: new RaycastDrawing(name, sensor, caster, config.meansize, root),
        });
      }
    }
    for (const [name, reward] of Object.entries(config.rewards ?? {})) {
      const sensors = (reward.terrain_sensors ?? []).flatMap((s) => casters.get(s) ?? []);
      this.drawings.set(`reward:${name}`, {
        label: name,
        drawing: new UprightDrawing(name, reward, sensors, root),
      });
    }
  }

  /** List `id` after the drawings until `removeSwitch`; a policy load keeps it. */
  addSwitch(id: string, entry: ViewerSwitch): void {
    this.switches.set(id, entry);
    entry.apply(this.shown && entry.enabled);
  }

  removeSwitch(id: string): void {
    this.switches.delete(id);
  }

  clear(): void {
    for (const { drawing } of this.drawings.values()) drawing.dispose();
    this.drawings.clear();
    for (const caster of this.casters) caster.dispose();
    this.casters.length = 0;
  }

  entries(): DebugVizEntry[] {
    const commands = this.commands.getDebugVisTerms().map(({ name, enabled }) => ({
      id: `command:${name}`,
      label: capitalize(name),
      enabled,
    }));
    const others = [...this.drawings].map(([id, { label, drawing }]) => ({
      id,
      label,
      enabled: drawing.enabled,
    }));
    const switches = [...this.switches].map(([id, entry]) => ({
      id,
      label: entry.label,
      enabled: entry.enabled,
    }));
    return [...commands, ...others, ...switches];
  }

  set(id: string, enabled: boolean): void {
    const viewerSwitch = this.switches.get(id);
    if (viewerSwitch) {
      viewerSwitch.enabled = enabled;
      viewerSwitch.apply(this.shown && enabled);
      return;
    }
    if (id.startsWith('command:')) {
      this.commands.setDebugVisEnabled(id.slice('command:'.length), enabled);
    } else {
      const entry = this.drawings.get(id);
      if (!entry) return;
      entry.drawing.enabled = enabled;
    }
    // The render loop would do this, but not in a hidden tab or a headless host.
    this.update();
  }

  isShown(): boolean {
    return this.shown;
  }

  setShown(shown: boolean): void {
    this.shown = shown;
    for (const entry of this.switches.values()) entry.apply(shown && entry.enabled);
    this.update();
  }

  /** Redraw everything from the current state, as mjlab's viewer does each frame. */
  update(): void {
    this.commands.updateDebugVisuals(this.shown);
    const { mjModel, mjData } = this.simulation();
    if (!mjModel || !mjData) return;
    for (const { drawing } of this.drawings.values()) {
      drawing.update(this.shown && drawing.enabled, mjModel, mjData);
    }
  }
}
