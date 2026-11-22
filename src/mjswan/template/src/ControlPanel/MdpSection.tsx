import { ActionIcon, Box, Button, Checkbox, Flex, Group, Paper, SegmentedControl, Select, Text, Tooltip } from '@mantine/core';
import { IconBolt, IconChevronDown, IconClock, IconPlayerPause, IconRefresh } from '@tabler/icons-react';
import type { EventTermDescriptor, MdpState, OnFire, TerminationTermDescriptor } from '../engine';
import { CommandSection } from './CommandSection';

export interface MdpHandlers {
  onTermination(name: string, patch: { enabled?: boolean; onFire?: OnFire }): void;
  onEvent(name: string, enabled: boolean): void;
  onTrigger(name: string): void;
  onClear(): void;
}

type OnFireKind = 'reset' | 'delay' | 'stop';

const titleCase = (name: string) =>
  name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
const labelOf = (term: { name: string; label?: string }) =>
  term.label && term.label !== term.name ? term.label : titleCase(term.name);
const seconds = (value: number) => `${Number(value.toFixed(2))} s`;
const kindOf = (onFire: OnFire): OnFireKind => (typeof onFire === 'object' ? 'delay' : onFire);

const DELAYS = [0.5, 1, 2, 5];
const DEFAULT_DELAY = 1;
// Wide enough for the delay box, so switching to Delay moves nothing else on the row.
const SLOT_WIDTH = 64;
const BOX_HEIGHT = 22;

const delayOptions = (current: number) =>
  (DELAYS.includes(current) ? DELAYS : [...DELAYS, current].sort((a, b) => a - b)).map((d) => ({
    value: String(d),
    label: seconds(d),
  }));

// The delay a term last had, so Reset or Stop and back to Delay brings it back.
const lastDelay: Record<string, number> = {};

function choose(handlers: MdpHandlers, term: TerminationTermDescriptor, kind: OnFireKind) {
  if (typeof term.onFire === 'object') lastDelay[term.name] = term.onFire.delay;
  handlers.onTermination(term.name, { onFire: kind === 'delay' ? { delay: lastDelay[term.name] ?? DEFAULT_DELAY } : kind });
}

const ON_FIRE = [
  { value: 'reset', label: 'Reset', Icon: IconRefresh },
  { value: 'delay', label: 'Delay', Icon: IconClock },
  { value: 'stop', label: 'Stop', Icon: IconPlayerPause },
] as const;

const nameStyle = { fontSize: '0.875em', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } as const;
const mono = { fontFamily: 'var(--mantine-font-family-monospace)' } as const;
const monoStyle = { ...mono, fontSize: '0.8em' } as const;

function Lamp({ lit }: { lit: boolean }) {
  return (
    <Box
      title={lit ? 'Condition holds' : 'Condition does not hold'}
      style={{
        width: 8,
        height: 8,
        flexShrink: 0,
        borderRadius: '50%',
        border: '1px solid',
        borderColor: lit ? 'var(--mantine-color-text)' : 'var(--mantine-color-default-border)',
        background: lit ? 'var(--mantine-color-text)' : 'transparent',
      }}
    />
  );
}

function Fired({ at, width }: { at: number | null; width?: number }) {
  return (
    <Text c="dimmed" title="Last fired this episode" style={{ ...monoStyle, flexShrink: 0, width, minWidth: '3.2em', textAlign: 'right' }}>
      {at === null ? '–' : seconds(at)}
    </Text>
  );
}

/** The delay, its box filling left to right while the reset it holds counts down. */
function DelayBox({ term, delay, time, handlers }: { term: TerminationTermDescriptor; delay: number; time: number; handlers: MdpHandlers }) {
  const held = term.resetAt !== null;
  const progress = held ? Math.min(1, Math.max(0, 1 - (term.resetAt! - time) / delay)) : 0;
  return (
    <Box pos="relative" style={{ width: SLOT_WIDTH, flexShrink: 0 }} title={held ? `Reset in ${Math.max(0, term.resetAt! - time).toFixed(1)} s` : undefined}>
      <Select
        size="xs"
        radius="xs"
        aria-label={`${labelOf(term)} delay`}
        data={delayOptions(delay)}
        value={String(delay)}
        onChange={(value) => value && handlers.onTermination(term.name, { onFire: { delay: Number(value) } })}
        allowDeselect={false}
        withCheckIcon={false}
        rightSection={<IconChevronDown size={12} />}
        rightSectionWidth={18}
        rightSectionPointerEvents="none"
        comboboxProps={{ zIndex: 1000 }}
        styles={{
          input: {
            ...monoStyle,
            height: BOX_HEIGHT,
            minHeight: BOX_HEIGHT,
            lineHeight: `${BOX_HEIGHT - 2}px`,
            paddingInlineStart: 6,
          },
        }}
      />
      {/* Over the input, not under it: the tint is light enough to read the value through. */}
      <Box
        aria-hidden
        style={{ position: 'absolute', inset: 1, borderRadius: 1, overflow: 'hidden', pointerEvents: 'none' }}
      >
        <Box
          style={{
            height: '100%',
            background: 'color-mix(in srgb, var(--mantine-primary-color-filled) 20%, transparent)',
            transformOrigin: 'left',
            transform: `scaleX(${progress})`,
            // The snapshot moves every 100 ms; the transition fills between, and a reset empties it at once.
            transition: held ? 'transform 100ms linear' : 'none',
          }}
        />
      </Box>
    </Box>
  );
}

function Subhead({ children }: { children: string }) {
  return (
    <Text c="dimmed" px="xs" pt="0.3em" pb="0.3em" style={{ fontSize: '0.8em', fontWeight: 500 }}>
      {children}
    </Text>
  );
}

function TerminationRow({ term, time, handlers }: { term: TerminationTermDescriptor; time: number; handlers: MdpHandlers }) {
  return (
    <Box px="xs" pb="0.6em">
      <Flex align="center" gap={8} style={{ minHeight: BOX_HEIGHT }}>
        <Checkbox
          size="xs"
          checked={term.enabled}
          aria-label={labelOf(term)}
          onChange={(e) => handlers.onTermination(term.name, { enabled: e.currentTarget.checked })}
        />
        <Text style={{ ...nameStyle, flex: 1 }} c={term.enabled ? undefined : 'dimmed'} title={term.name}>
          {labelOf(term)}
        </Text>
        <Lamp lit={term.active} />
        {typeof term.onFire === 'object' ? (
          <DelayBox term={term} delay={term.onFire.delay} time={time} handlers={handlers} />
        ) : (
          <Fired at={term.firedAt} width={SLOT_WIDTH} />
        )}
      </Flex>
      <Box pl={24} pt={6}>
        <SegmentedControl
          size="xs"
          radius="xs"
          fullWidth
          aria-label={`What follows ${labelOf(term)}`}
          value={kindOf(term.onFire)}
          onChange={(value) => choose(handlers, term, value as OnFireKind)}
          data={ON_FIRE.map(({ value, label, Icon }) => ({
            value,
            label: (
              <Group gap={4} wrap="nowrap" justify="center">
                <Icon size={12} />
                <span>{label}</span>
              </Group>
            ),
          }))}
        />
      </Box>
    </Box>
  );
}

function EventRow({ term, gated, handlers }: { term: EventTermDescriptor; gated: boolean; handlers: MdpHandlers }) {
  return (
    <Flex px="xs" pb="0.4em" align="center" gap={8}>
      {term.mode === 'manual' ? (
        <Box style={{ width: 16, flexShrink: 0 }} />
      ) : (
        <Checkbox
          size="xs"
          checked={term.enabled}
          aria-label={labelOf(term)}
          onChange={(e) => handlers.onEvent(term.name, e.currentTarget.checked)}
        />
      )}
      <Text style={{ ...nameStyle, flex: 1 }} c={term.enabled ? undefined : 'dimmed'} title={term.name}>
        {labelOf(term)}
      </Text>
      <Fired at={term.firedAt} />
      <Tooltip label="Trigger now" withArrow openDelay={300}>
        <ActionIcon
          variant="default"
          size="sm"
          aria-label={`Trigger ${labelOf(term)}`}
          disabled={gated}
          onClick={() => handlers.onTrigger(term.name)}
        >
          <IconBolt size={14} />
        </ActionIcon>
      </Tooltip>
    </Flex>
  );
}

/**
 * The MDP's terminations and events, as engine.mdp drives them. `gated` names the manual
 * terms whose `disabled_when` schedule is running, which a trigger would not fire.
 */
export function MdpSection({ mdp, gated, handlers }: { mdp: MdpState; gated: ReadonlySet<string>; handlers: MdpHandlers }) {
  return (
    <CommandSection label="MDP">
      {mdp.terminations.length > 0 && <Subhead>Terminations</Subhead>}
      {mdp.terminations.map((term) => (
        <TerminationRow key={term.name} term={term} time={mdp.time} handlers={handlers} />
      ))}
      {mdp.events.length > 0 && <Subhead>Events</Subhead>}
      {mdp.events.map((term) => (
        <EventRow key={term.name} term={term} gated={gated.has(term.name)} handlers={handlers} />
      ))}
      <Box px="xs" pt="0.2em">
        <Button variant="default" size="xs" fullWidth leftSection={<IconRefresh size={12} />} onClick={handlers.onClear}>
          Reset MDP
        </Button>
      </Box>
    </CommandSection>
  );
}

/** The term that stopped the simulation, or the reset a delayed termination holds, over the viewer. */
export function MdpStageNotice({ mdp }: { mdp: MdpState }) {
  const { stoppedBy, pendingReset } = mdp;
  if (!stoppedBy && !pendingReset) return null;
  return (
    <Paper
      shadow="sm"
      style={{
        position: 'absolute',
        top: 12,
        left: '50%',
        transform: 'translateX(-50%)',
        padding: '4px 12px',
        zIndex: 10,
        pointerEvents: 'none',
        background: 'rgba(0, 0, 0, 0.75)',
        color: '#fff',
        fontSize: '0.85em',
      }}
    >
      <Group gap={8} wrap="nowrap">
        {stoppedBy ? (
          <>
            <IconPlayerPause size={14} />
            <span>Stopped</span>
            <span>{titleCase(stoppedBy.term)}</span>
            <span style={mono}>{seconds(stoppedBy.at)}</span>
          </>
        ) : (
          <>
            <IconClock size={14} />
            <span>
              Reset in <span style={mono}>{Math.max(0, pendingReset!.at - mdp.time).toFixed(1)} s</span>
            </span>
            <span>{titleCase(pendingReset!.term)}</span>
          </>
        )}
      </Group>
    </Paper>
  );
}
