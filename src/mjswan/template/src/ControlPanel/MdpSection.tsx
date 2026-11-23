import { useEffect, useRef, useState } from 'react';
import { ActionIcon, Box, Button, Checkbox, Flex, Group, SegmentedControl, Text, TextInput, Tooltip } from '@mantine/core';
import { IconBolt, IconClock, IconPlayerPause, IconRefresh } from '@tabler/icons-react';
import type { EventTermDescriptor, MdpState, OnFire, TerminationTermDescriptor } from '../engine';
import { CommandSection } from './CommandSection';
import { heldProgress, parseDelayMs, toMs } from './mdpDelay';

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
const kindOf = (onFire: OnFire): OnFireKind => (typeof onFire === 'object' ? 'delay' : onFire);

const DEFAULT_DELAY = 1;
const BOX_WIDTH = 80;
const BOX_HEIGHT = 22;

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
const monoStyle = { fontFamily: 'var(--mantine-font-family-monospace)', fontSize: '0.8em' } as const;

/** Whether the termination's condition held at the last control step. */
function Lamp({ lit }: { lit: boolean }) {
  return (
    <Tooltip label={lit ? 'Active' : 'Inactive'} withArrow openDelay={300}>
      <Box
        role="img"
        aria-label={lit ? 'Active' : 'Inactive'}
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
    </Tooltip>
  );
}

/**
 * The delay in milliseconds, typed and kept on Enter or blur (Escape drops the edit), its box
 * filling left to right while the reset it holds counts down. 0 is the reset at once.
 */
function DelayBox({ term, delay, time, handlers }: { term: TerminationTermDescriptor; delay: number; time: number; handlers: MdpHandlers }) {
  const ms = toMs(delay);
  const [draft, setDraft] = useState(String(ms));
  const editing = useRef(false);
  const dropped = useRef(false);
  useEffect(() => {
    if (!editing.current) setDraft(String(ms));
  }, [ms]);
  const keep = () => {
    const next = parseDelayMs(draft);
    setDraft(String(next ?? ms));
    if (next !== null && next !== ms) handlers.onTermination(term.name, { onFire: { delay: next / 1000 } });
  };
  const held = term.resetAt !== null;
  return (
    <Box pos="relative" style={{ width: BOX_WIDTH, flexShrink: 0 }}>
      <TextInput
        size="xs"
        radius="xs"
        inputMode="numeric"
        aria-label={`${labelOf(term)} delay (ms)`}
        value={draft}
        onChange={(e) => setDraft(e.currentTarget.value)}
        onFocus={() => {
          editing.current = true;
        }}
        onBlur={() => {
          editing.current = false;
          if (dropped.current) dropped.current = false;
          else keep();
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
          if (e.key === 'Escape') {
            dropped.current = true;
            setDraft(String(ms));
            e.currentTarget.blur();
          }
        }}
        rightSection={<Text c="dimmed" style={monoStyle}>ms</Text>}
        rightSectionWidth={24}
        rightSectionPointerEvents="none"
        styles={{
          input: {
            ...monoStyle,
            height: BOX_HEIGHT,
            minHeight: BOX_HEIGHT,
            lineHeight: `${BOX_HEIGHT - 2}px`,
            paddingInlineStart: 6,
            textAlign: 'right',
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
            width: `${heldProgress(term, time) * 100}%`,
            background: 'color-mix(in srgb, var(--mantine-primary-color-filled) 20%, transparent)',
            // The snapshot moves every 100 ms; the transition fills between, and a reset empties it at once.
            transition: held ? 'width 100ms linear' : 'none',
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
        <Text style={nameStyle} c={term.enabled ? undefined : 'dimmed'} title={term.name}>
          {labelOf(term)}
        </Text>
        <Lamp lit={term.active} />
        <Box style={{ flex: 1 }} />
        {typeof term.onFire === 'object' && <DelayBox term={term} delay={term.onFire.delay} time={time} handlers={handlers} />}
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
      <Tooltip label="Trigger" withArrow openDelay={300}>
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
    <CommandSection label="MDP" expandByDefault={false}>
      {mdp.terminations.length > 0 && <Subhead>Terminations</Subhead>}
      {mdp.terminations.map((term) => (
        <TerminationRow key={term.name} term={term} time={mdp.time} handlers={handlers} />
      ))}
      {mdp.events.length > 0 && <Subhead>Events</Subhead>}
      {mdp.events.map((term) => (
        <EventRow key={term.name} term={term} gated={gated.has(term.name)} handlers={handlers} />
      ))}
      <Box px="xs" pt="0.2em" pb="0.5em">
        <Button variant="default" size="xs" fullWidth leftSection={<IconRefresh size={12} />} onClick={handlers.onClear}>
          Reset MDP
        </Button>
      </Box>
    </CommandSection>
  );
}
