import { useCallback, useEffect, useState } from 'react';
import {
  ActionIcon,
  Anchor,
  Box,
  Button,
  Checkbox,
  Divider,
  Image,
  Menu,
  Modal,
  Select,
  Stack,
  Text,
  Tooltip,
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconChevronDown, IconRefresh, IconSquareX, IconX } from '@tabler/icons-react';

/** Icons an mjlab GUI can ask for; the rest of tabler is unbundled, so goes undrawn. */
const COMMAND_BUTTON_ICONS: Record<string, typeof IconSquareX> = {
  'square-x': IconSquareX,
};
import type { SplatConfig } from '../core/scene/splat';
import { MJSWAN_VERSION, GITHUB_CONTRIBUTORS, type Contributor } from '../Version';
import FloatingPanel from './FloatingPanel';
import { LabeledInput } from './LabeledInput';
import { CommandSection } from './CommandSection';
import { InteractionSection } from './InteractionSection';
import { SliderRow } from './SliderRow';
import { SplatSection } from './SplatSection';
import type {
  CommandDescriptor,
  DebugVisDescriptor,
  EventDescriptor,
  InteractionModeDescriptor,
  InteractionModeId,
} from '../engine';

export interface SelectOption {
  value: string;
  label: string;
}

interface ControlPanelProps {
  visible: boolean;
  onVisibleChange: (visible: boolean) => void;
  projects: SelectOption[];
  projectValue: string | null;
  projectLabel: string;
  onProjectChange: (value: string | null) => void;
  scenes: SelectOption[];
  sceneValue: string | null;
  onSceneChange: (value: string | null) => void;
  splats: SelectOption[];
  splatSection?: boolean;
  splatValue: string | null;
  onSplatChange: (value: string | null) => void;
  /** Splat config from the current scene (null if no splat), used for dev-mode calibration. */
  splatConfig?: SplatConfig | null;
  /** Dev-mode: update splat calibration (scale, x/y/z offsets, roll/pitch/yaw) live. */
  onCalibrateSplat?: (scale: number, xOffset: number, yOffset: number, zOffset: number, roll: number, pitch: number, yaw: number) => void;
  /** Load a splat from an arbitrary .spz URL. Returns true on success, false on failure. */
  onSplatUrlLoad?: (url: string) => Promise<boolean>;
  policies: SelectOption[];
  policyValue: string | null;
  onPolicyChange: (value: string | null) => void;
  motions: SelectOption[];
  motionValue: string | null;
  onMotionChange: (value: string | null) => void;
  showReferenceMotion: boolean;
  onShowReferenceMotionChange: (value: boolean) => void;
  /** Whether command controls are enabled */
  commandsEnabled?: boolean;
  /** Command descriptors from the engine state snapshot. */
  commands: CommandDescriptor[];
  /** Current command values keyed by descriptor id. */
  commandValues: Record<string, number>;
  /** Write a slider/checkbox command value (engine.commands.set). */
  onCommandChange: (id: string, value: number) => void;
  /** Press a button command (engine.commands.trigger). */
  onCommandTrigger?: (id: string) => void;
  /** Event terms the operator can drive: manual buttons, interval schedules. */
  events?: EventDescriptor[];
  /** Fire a `mode="manual"` event term (engine.events.fire). */
  onEventFire?: (name: string) => void;
  /** Start or stop a `mode="interval"` term's schedule (engine.events.setArmed). */
  onEventArmedChange?: (name: string, armed: boolean) => void;
  /** Command terms with a debug drawing to toggle. */
  debugVis?: DebugVisDescriptor[];
  /** Show or hide one term's debug drawing (engine.debugVis.set). */
  onDebugVisChange?: (term: string, enabled: boolean) => void;
  /** Reset the simulation (engine.reset). */
  onReset?: () => void;
  /** Pointer modes the viewer offers, as the engine reports them. */
  interactions?: InteractionModeDescriptor[];
  interactionMode?: InteractionModeId;
  interactionParams?: Readonly<Record<string, Readonly<Record<string, number>>>>;
  onInteractionModeChange?: (mode: InteractionModeId) => void;
  onInteractionParamChange?: (mode: InteractionModeId, name: string, value: number) => void;
}

function isEditableElement(element: Element | null): boolean {
  if (!(element instanceof HTMLElement)) {
    return false;
  }

  if (element.isContentEditable) {
    return true;
  }

  return (
    element.closest(
      'input, textarea, select, [contenteditable], [role="textbox"], [role="searchbox"], [role="combobox"]'
    ) !== null
  );
}

/**
 * Format group name for display (e.g., "velocity" -> "Velocity")
 */
function formatGroupName(groupName: string): string {
  return groupName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/** One command's slider, preceded by its "Max" companion — the order mjlab declares. */
function SliderControl({
  command,
  value,
  onChange,
  disabled,
  enabledWhenValue,
}: {
  command: CommandDescriptor;
  value: number;
  onChange: (id: string, value: number) => void;
  disabled?: boolean;
  enabledWhenValue?: number;
}) {
  const dependencyDisabled = command.enabledWhen !== undefined && (enabledWhenValue ?? 0) < 0.5;
  const isDisabled = disabled || dependencyDisabled;

  // A companion "Max <label>" slider, when the build asked for one (brief §3a).
  // Presentational only: it rescales how far *this* slider can be dragged and is
  // never sent to the engine, matching mjlab's play GUI. Symmetric around zero.
  const range = command.adjustableRange;
  const [reach, setReach] = useState(range?.default ?? 0);
  // `min`/`max` are declared optional because they are slider-only; a slider has them.
  const min = range ? -reach : (command.min ?? 0);
  const max = range ? reach : (command.max ?? 1);

  // Narrowing the reach past the current value would leave the thumb outside the
  // track, so bring the command with it rather than showing a stale position.
  useEffect(() => {
    if (!range) return;
    const clamped = Math.max(-reach, Math.min(reach, value));
    if (clamped !== value) onChange(command.id, clamped);
  }, [range, reach, value, command.id, onChange]);

  return (
    <>
      {range && (
        <SliderRow
          id={`${command.id}:max`}
          label={range.label ?? `Max ${command.label}`}
          value={reach}
          min={range.min}
          max={range.max}
          step={range.step}
          onChange={setReach}
          disabled={isDisabled}
        />
      )}
      <SliderRow
        id={command.id}
        label={command.label}
        value={value}
        min={min}
        max={max}
        step={command.step}
        onChange={(next) => onChange(command.id, next)}
        disabled={isDisabled}
      />
    </>
  );
}

function CheckboxControl({
  command,
  value,
  onChange,
  disabled,
}: {
  command: CommandDescriptor;
  value: number;
  onChange: (id: string, value: number) => void;
  disabled?: boolean;
}) {
  return (
    <LabeledInput id={command.id} label={command.label}>
      <Checkbox
        id={command.id}
        checked={value >= 0.5}
        onChange={(event) => onChange(command.id, event.currentTarget.checked ? 1.0 : 0.0)}
        size="xs"
        disabled={disabled}
      />
    </LabeledInput>
  );
}

function ControlPanel(props: ControlPanelProps) {
  const {
    visible,
    onVisibleChange,
    projects,
    projectValue,
    projectLabel,
    onProjectChange,
    scenes,
    sceneValue,
    onSceneChange,
    splats,
    splatSection = false,
    splatValue,
    onSplatChange,
    splatConfig,
    onCalibrateSplat,
    onSplatUrlLoad,
    policies,
    policyValue,
    onPolicyChange,
    motions,
    motionValue,
    onMotionChange,
    showReferenceMotion,
    onShowReferenceMotionChange,
    commandsEnabled = false,
    commands,
    commandValues,
    onCommandChange,
    onCommandTrigger,
    events = [],
    onEventFire,
    onEventArmedChange,
    debugVis = [],
    onDebugVisChange,
    onReset,
    interactions = [],
    interactionMode = 'pull',
    interactionParams = {},
    onInteractionModeChange,
    onInteractionParamChange,
  } = props;

  const [aboutModalOpened, { open: openAbout, close: closeAbout }] = useDisclosure(false);
  const [splatSearchValue, setSplatSearchValue] = useState('');
  const [splatUrlError, setSplatUrlError] = useState<string | null>(null);
  const [customSplatActive, setCustomSplatActive] = useState(false);

  const handleSplatKeyDown = useCallback(async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== 'Enter' || !onSplatUrlLoad) return;
    const trimmed = splatSearchValue.trim();
    if (splats.some(s => s.label === trimmed || s.value === trimmed)) return;
    if (!trimmed.toLowerCase().endsWith('.spz')) {
      setSplatUrlError('URL must end with .spz');
      return;
    }
    const ok = await onSplatUrlLoad(trimmed);
    if (ok) {
      setSplatUrlError(null);
      setCustomSplatActive(true);
    } else {
      setSplatUrlError('File not found at the specified URL');
    }
  }, [onSplatUrlLoad, splatSearchValue, splats]);

  // Command groups derived from the engine-supplied descriptors.
  const commandGroups = Array.from(new Set(commands.map((c) => c.group)));

  const handleReset = useCallback(() => {
    onReset?.();
  }, [onReset]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.repeat ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey
      ) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key !== 'c' && key !== 'r') {
        return;
      }

      const target = event.target instanceof Element ? event.target : document.activeElement;
      if (isEditableElement(target)) {
        return;
      }

      event.preventDefault();
      if (key === 'c') {
        onVisibleChange(!visible);
      } else {
        handleReset();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [visible, onVisibleChange, handleReset]);

  const getCommandsForGroup = (groupName: string): CommandDescriptor[] => {
    // Declaration order, so a `Zero` button lands under the sliders it zeroes.
    return commands.filter(
      (cmd) =>
        cmd.group === groupName &&
        (cmd.type === 'slider' || cmd.type === 'checkbox' || cmd.type === 'button')
    );
  };

  // Only show panel if we have data to display
  if (!projects.length && !scenes.length && !policies.length) {
    return null;
  }

  return (
    <>
    <Modal
      opened={aboutModalOpened}
      onClose={closeAbout}
      size="lg"
      title={null}
      centered
      styles={{ body: { textAlign: 'center' } }}
    >
      <Stack gap="md" align="center">
        <Image src={`${import.meta.env.BASE_URL || '/'}logo.svg`} style={{ width: '8em', height: 'auto' }} />
        <Text size="xl" fw={700}>powered by mjswan</Text>
        <Text size="sm" c="dimmed">version {MJSWAN_VERSION}</Text>
        <Text size="sm" c="dimmed">MuJoco Simulation on Web Assembly with Neural netwroks</Text>
        <Divider w="100%" />
        <Box>
          <Anchor href="https://github.com/ttktjmt/mjswan" target="_blank" style={{ fontWeight: '600' }}>
            GitHub
          </Anchor>
          &nbsp;&nbsp;&bull;&nbsp;&nbsp;
          <Anchor href="https://mjswan.readthedocs.io" target="_blank" style={{ fontWeight: '600' }}>
            Documentation
          </Anchor>
        </Box>
        <Divider w="100%" />
        <Box
          style={{
            textAlign: 'left',
            maxHeight: '120px',
            overflowY: 'auto',
            lineHeight: '1',
            fontSize: '0.8rem',
            opacity: '0.75',
          }}
          px="md"
        >
          Thanks to our contributors! <br />
          {GITHUB_CONTRIBUTORS.map((contributor: Contributor, index: number) => (
            <span key={contributor.login}>
              <Anchor
                href={contributor.html_url}
                target="_blank"
                style={{ textDecoration: 'none', fontSize: '0.75rem' }}
              >
                {contributor.login}
              </Anchor>
              {index < GITHUB_CONTRIBUTORS.length - 1 && ', '}
            </span>
          ))}
        </Box>
      </Stack>
    </Modal>
    <FloatingPanel
      width="20em"
      visible={visible}
      onVisibleChange={onVisibleChange}
      hiddenButtonTooltip="Show controls (C)"
    >
      <FloatingPanel.Handle>
        <Tooltip label={`mjswan ${MJSWAN_VERSION}`}>
          <Box
            component="a"
            onClick={(e) => { e.stopPropagation(); openAbout(); }}
            onMouseDown={(e) => e.stopPropagation()}
            onTouchStart={(e) => e.stopPropagation()}
            style={{ position: "absolute", cursor: "pointer", display: "flex", top: "0.8em", left: "0.9em" }}
          >
            <Image src={`${import.meta.env.BASE_URL || '/'}logo.svg`} style={{ width: "1.2em", height: "auto" }} />
          </Box>
        </Tooltip>
        <div style={{ width: "1.1em" }} />
        <FloatingPanel.HideWhenCollapsed>
          <Box
            px="xs"
            style={{
              flexGrow: 1,
              letterSpacing: "-0.5px",
              display: "flex",
              alignItems: "center",
              gap: "0.5em",
            }}
            pt="0.1em"
          >
            <span
              style={{
                minWidth: 0,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {projectLabel}
            </span>
            {projects.length > 1 && (
              <Menu position="bottom-start" offset={5}>
                <Menu.Target>
                  <Box
                    onClick={(e) => e.stopPropagation()}
                    onMouseDown={(e) => e.stopPropagation()}
                    onTouchStart={(e) => e.stopPropagation()}
                    style={{
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      flexShrink: 0,
                    }}
                  >
                    <IconChevronDown size={16} />
                  </Box>
                </Menu.Target>
                <Menu.Dropdown onClick={(e) => e.stopPropagation()}>
                  {projects.map((project) => (
                    <Menu.Item
                      key={project.value}
                      onClick={(e) => {
                        e.stopPropagation();
                        onProjectChange(project.value);
                      }}
                      style={{
                        fontWeight: project.value === projectValue ? 600 : 400,
                        backgroundColor:
                          project.value === projectValue
                            ? "rgba(34, 139, 230, 0.1)"
                            : undefined,
                      }}
                    >
                      {project.label}
                    </Menu.Item>
                  ))}
                </Menu.Dropdown>
              </Menu>
            )}
          </Box>
        </FloatingPanel.HideWhenCollapsed>
        <FloatingPanel.HideWhenExpanded>
          <Box px="xs" style={{ flexGrow: 1, letterSpacing: "-0.5px" }} pt="0.1em">
            {projectLabel}
          </Box>
        </FloatingPanel.HideWhenExpanded>
        <Tooltip label="Hide controls (C)">
          <ActionIcon
            variant="subtle"
            color="gray"
            size="sm"
            aria-label="Hide controls"
            onClick={(event) => {
              event.stopPropagation();
              onVisibleChange(false);
            }}
            onMouseDown={(event) => event.stopPropagation()}
            onTouchStart={(event) => event.stopPropagation()}
          >
            <IconX size={14} />
          </ActionIcon>
        </Tooltip>
      </FloatingPanel.Handle>
      <FloatingPanel.Contents>
        <Box pt="0.375em">
          {scenes.length > 0 && (
            <LabeledInput id="scene-select" label="Scene">
              <Select
                id="scene-select"
                placeholder="Select scene"
                data={scenes}
                value={sceneValue}
                onChange={onSceneChange}
                size="xs"
                radius="xs"
                searchable
                clearable={false}
                styles={{
                  input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em' },
                }}
                comboboxProps={{ zIndex: 1000 }}
              />
            </LabeledInput>
          )}

          {(splats.length > 0 || splatSection) && (
            <LabeledInput id="splat-select" label="Splat">
              <Tooltip label={splatUrlError ?? ''} color="red" position="bottom" opened={splatUrlError !== null} withArrow>
                <Select
                  id="splat-select"
                  placeholder={onSplatUrlLoad !== undefined ? 'Select splat or paste .spz URL' : 'Select splat'}
                  data={splats}
                  value={splatValue}
                  onChange={(val) => { onSplatChange(val); setSplatUrlError(null); setCustomSplatActive(false); }}
                  searchable={onSplatUrlLoad !== undefined}
                  searchValue={splatSearchValue}
                  onSearchChange={(val) => { setSplatSearchValue(val); if (val) setSplatUrlError(null); }}
                  onKeyDown={handleSplatKeyDown}
                  size="xs"
                  radius="xs"
                  clearable
                  styles={{
                    input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em' },
                  }}
                  comboboxProps={{ zIndex: 1000 }}
                />
              </Tooltip>
            </LabeledInput>
          )}

          {/* Splat controls — when splat.control === true and splat is selected, or a custom URL splat is active */}
          {((splatConfig?.control && splatValue !== null) || customSplatActive) && onCalibrateSplat && (
            <SplatSection
              scale={splatConfig?.scale ?? 1.0}
              xOffset={splatConfig?.xOffset ?? 0.0}
              yOffset={splatConfig?.yOffset ?? 0.0}
              zOffset={splatConfig?.zOffset ?? 0.0}
              roll={splatConfig?.roll ?? 0.0}
              pitch={splatConfig?.pitch ?? 0.0}
              yaw={splatConfig?.yaw ?? 0.0}
              onCalibrate={onCalibrateSplat}
            />
          )}

          {policies.length > 0 && (
            <LabeledInput id="policy-select" label="Policy">
              <Select
                id="policy-select"
                placeholder="Select policy"
                data={policies}
                value={policyValue}
                onChange={onPolicyChange}
                size="xs"
                radius="xs"
                searchable
                clearable
                styles={{
                  input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em' },
                }}
                comboboxProps={{ zIndex: 1000 }}
              />
            </LabeledInput>
          )}

          {motions.length > 0 && (
            <>
              <LabeledInput id="motion-select" label="Motion">
                <Select
                  id="motion-select"
                  placeholder="Select motion"
                  data={motions}
                  value={motionValue}
                  onChange={onMotionChange}
                  size="xs"
                  radius="xs"
                  searchable
                  clearable
                  styles={{
                    input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em' },
                  }}
                  comboboxProps={{ zIndex: 1000 }}
                />
              </LabeledInput>
              <Box pb="0.5em" px="xs">
                <Checkbox
                  label="Show reference"
                  checked={showReferenceMotion}
                  onChange={(event) => onShowReferenceMotionChange(event.currentTarget.checked)}
                  size="xs"
                />
              </Box>
            </>
          )}

          {/* Command Groups - only show if there are commands */}
          {commandGroups.length > 0 && commands.some(cmd => cmd.type === 'slider' || cmd.type === 'checkbox' || cmd.type === 'button') && (
            <CommandSection label="Commands" expandByDefault={true}>
              {commandGroups.map((groupName) => {
                const groupCommands = getCommandsForGroup(groupName);
                if (groupCommands.length === 0) return null;

                return (
                  <CommandSection
                    key={groupName}
                    label={formatGroupName(groupName)}
                    expandByDefault={true}
                  >
                    {groupCommands.map((command) => {
                      if (command.type === 'checkbox') {
                        return (
                          <CheckboxControl
                            key={command.id}
                            command={command}
                            value={commandValues[command.id] ?? 0}
                            onChange={onCommandChange}
                            disabled={!commandsEnabled}
                          />
                        );
                      }
                      if (command.type === 'button') {
                        const Icon = command.icon ? COMMAND_BUTTON_ICONS[command.icon] : undefined;
                        return (
                          <Box key={command.id} px="xs" pb="0.5em">
                            <Button
                              size="sm"
                              fullWidth
                              style={{ height: '2em' }}
                              leftSection={Icon ? <Icon size="1em" /> : undefined}
                              onClick={() => onCommandTrigger?.(command.id)}
                              disabled={!commandsEnabled || !onCommandTrigger}
                            >
                              {command.label}
                            </Button>
                          </Box>
                        );
                      }
                      if (command.type !== 'slider') {
                        return null;
                      }
                      return (
                        <SliderControl
                          key={command.id}
                          command={command}
                          value={commandValues[command.id] ?? 0}
                          onChange={onCommandChange}
                          disabled={!commandsEnabled}
                          enabledWhenValue={
                            command.enabledWhen
                              ? commandValues[`${command.group}:${command.enabledWhen}`]
                              : undefined
                          }
                        />
                      );
                    })}
                  </CommandSection>
                );
              })}
            </CommandSection>
          )}

          {/* Events: a button fires one, a checkbox lets its schedule run — scene-level,
              so no policy has to be loaded for either. */}
          {events.length > 0 && (
            <CommandSection label="Events" expandByDefault={true}>
              {events
                .filter((event) => event.kind === 'manual')
                .map((event) => (
                  <Box key={event.name} px="xs" pb="0.5em">
                    <Button
                      size="sm"
                      fullWidth
                      style={{ height: '2em' }}
                      onClick={() => onEventFire?.(event.name)}
                      disabled={!onEventFire || !event.armed}
                    >
                      {event.label}
                    </Button>
                  </Box>
                ))}
              {events
                .filter((event) => event.kind === 'interval')
                .map((event) => (
                  <LabeledInput key={event.name} id={`event:${event.name}`} label={event.label}>
                    <Checkbox
                      id={`event:${event.name}`}
                      checked={event.armed}
                      onChange={(changed) =>
                        onEventArmedChange?.(event.name, changed.currentTarget.checked)
                      }
                      disabled={!onEventArmedChange}
                      size="xs"
                    />
                  </LabeledInput>
                ))}
            </CommandSection>
          )}

          {/* Debug Viz — mjlab's own folder, one checkbox per drawing term. */}
          {debugVis.length > 0 && onDebugVisChange && (
            <CommandSection label="Debug Viz" expandByDefault={true}>
              {debugVis.map((term) => (
                <LabeledInput
                  key={term.term}
                  id={`debugvis:${term.term}`}
                  // The section names what is toggled; the term only tells several apart.
                  label={debugVis.length > 1 ? `Enable ${formatGroupName(term.term)}` : 'Enable'}
                >
                  <Checkbox
                    id={`debugvis:${term.term}`}
                    checked={term.enabled}
                    onChange={(event) => onDebugVisChange(term.term, event.currentTarget.checked)}
                    size="xs"
                  />
                </LabeledInput>
              ))}
            </CommandSection>
          )}

          {/* What the pointer does. Scene-level, so it sits below the policy's own
              controls and above the reset. */}
          {onInteractionModeChange && onInteractionParamChange && (
            <InteractionSection
              modes={interactions}
              mode={interactionMode}
              params={interactionParams}
              onModeChange={onInteractionModeChange}
              onParamChange={onInteractionParamChange}
            />
          )}

          {/* Reset Button - always at bottom */}
          <Divider mb="xs" mx="xs" />
          <Box px="xs" pb="xs">
            <Button
              variant="light"
              color="red"
              size="sm"
              fullWidth
              style={{ height: '2em' }}
              leftSection={<IconRefresh size={14} />}
              onClick={handleReset}
            >
              Reset
            </Button>
          </Box>
        </Box>
      </FloatingPanel.Contents>
    </FloatingPanel>
    </>
  );
}

export default ControlPanel;
