import { useEffect, useRef, useState } from 'react';
import { ActionIcon, Box, Button, Checkbox, Flex, Group, SegmentedControl, Text, TextInput, Tooltip } from '@mantine/core';
import { IconBolt, IconClock, IconRefresh, IconSnowflake } from '@tabler/icons-react';
import type { EventTermDescriptor, MdpState, OnFire, TerminationTermDescriptor } from '../engine';
import { CommandSection } from './CommandSection';
import { freezeProgress, heldProgress, parseDelayMs, toMs } from './mdpDelay';

export interface MdpHandlers {
  onTermination(name: string, patch: { enabled?: boolean; onFire?: OnFire }): void;
  onEvent(name: string, enabled: boolean): void;
  onTrigger(name: string): void;
  onClear(): void;
}

type OnFireKind = 'reset' | 'delay' | 'freeze';

const titleCase = (name: string) =>
  name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
const labelOf = (term: { name: string; label?: string }) =>
  term.label && term.label !== term.name ? term.label : titleCase(term.name);
const kindOf = (onFire: OnFire): OnFireKind =>
  typeof onFire === 'object' ? ('delay' in onFire ? 'delay' : 'freeze') : onFire;

const BOX_WIDTH = 80;
const BOX_HEIGHT = 22;

// What a term last had under Delay and under Freeze, so leaving and coming back brings it
// back; a first Delay or Freeze is a second.
const last: Record<'delay' | 'freeze', Record<string, OnFire>> = { delay: {}, freeze: {} };
const FIRST: Record<'delay' | 'freeze', OnFire> = { delay: { delay: 1 }, freeze: { freeze: 1 } };

function choose(handlers: MdpHandlers, term: TerminationTermDescriptor, kind: OnFireKind) {
  const current = kindOf(term.onFire);
  if (current !== 'reset') last[current][term.name] = term.onFire;
  handlers.onTermination(term.name, { onFire: kind === 'reset' ? 'reset' : (last[kind][term.name] ?? FIRST[kind]) });
}

const ON_FIRE = [
  { value: 'reset', label: 'Reset', Icon: IconRefresh },
  { value: 'delay', label: 'Delay', Icon: IconClock },
  { value: 'freeze', label: 'Freeze', Icon: IconSnowflake },
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

/** `Date.now()` every frame while `on`: the engine sends nothing while frozen. */
function useWallClock(on: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!on) return;
    let frame = requestAnimationFrame(function tick() {
      setNow(Date.now());
      frame = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(frame);
  }, [on]);
  return now;
}

/**
 * The wait in milliseconds, typed and kept on Enter or blur (Escape drops the edit), its box
 * filling left to right as the wait runs: a delay's in episode time, a freeze's on the wall
 * clock. 0 is the reset at once. An empty Freeze waits for a reset (∞); an empty Delay keeps
 * the delay it had.
 */
function WaitBox({
  term,
  kind,
  seconds,
  progress,
  handlers,
}: {
  term: TerminationTermDescriptor;
  kind: 'delay' | 'freeze';
  seconds: number | null;
  progress: number;
  handlers: MdpHandlers;
}) {
  const shown = seconds === null ? '' : String(toMs(seconds));
  const [draft, setDraft] = useState(shown);
  const editing = useRef(false);
  const dropped = useRef(false);
  useEffect(() => {
    if (!editing.current) setDraft(shown);
  }, [shown]);
  const keep = () => {
    if (draft.trim() === '' && kind === 'freeze') {
      setDraft('');
      if (seconds !== null) handlers.onTermination(term.name, { onFire: 'freeze' });
      return;
    }
    const next = parseDelayMs(draft);
    setDraft(next === null ? shown : String(next));
    if (next === null || (seconds !== null && next === toMs(seconds))) return;
    handlers.onTermination(term.name, { onFire: kind === 'delay' ? { delay: next / 1000 } : { freeze: next / 1000 } });
  };
  return (
    <Box pos="relative" style={{ width: BOX_WIDTH, flexShrink: 0 }} title={kind === 'freeze' && seconds === null ? 'Until reset' : undefined}>
      <TextInput
        size="xs"
        radius="xs"
        inputMode="numeric"
        aria-label={`${labelOf(term)} ${kind} (ms)`}
        placeholder={kind === 'freeze' ? '∞' : undefined}
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
            setDraft(shown);
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
            width: `${progress * 100}%`,
            background: 'color-mix(in srgb, var(--mantine-primary-color-filled) 20%, transparent)',
            // A delay's snapshot moves every 100 ms: the transition fills between. A freeze's is per frame.
            transition: kind === 'delay' && progress > 0 ? 'width 100ms linear' : 'none',
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

function TerminationRow({
  term,
  time,
  frozenBy,
  handlers,
}: {
  term: TerminationTermDescriptor;
  time: number;
  frozenBy: MdpState['frozenBy'];
  handlers: MdpHandlers;
}) {
  const kind = kindOf(term.onFire);
  const counting = frozenBy?.term === term.name && frozenBy.resumesAt !== null;
  const now = useWallClock(counting);
  const onFire = term.onFire;
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
        {kind === 'delay' && typeof onFire === 'object' && 'delay' in onFire && (
          <WaitBox term={term} kind="delay" seconds={onFire.delay} progress={heldProgress(term, time)} handlers={handlers} />
        )}
        {kind === 'freeze' && (
          <WaitBox
            term={term}
            kind="freeze"
            seconds={typeof onFire === 'object' && 'freeze' in onFire ? onFire.freeze : null}
            progress={freezeProgress(frozenBy, term.name, onFire, now)}
            handlers={handlers}
          />
        )}
      </Flex>
      <Box pl={24} pt={6}>
        <SegmentedControl
          size="xs"
          radius="xs"
          fullWidth
          aria-label={`What follows ${labelOf(term)}`}
          value={kind}
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
        <TerminationRow key={term.name} term={term} time={mdp.time} frozenBy={mdp.frozenBy} handlers={handlers} />
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
