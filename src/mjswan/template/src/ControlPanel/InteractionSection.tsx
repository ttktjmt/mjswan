import { Box, SegmentedControl, Text } from '@mantine/core';

import type { InteractionModeDescriptor, InteractionModeId } from '../engine';
import { CommandSection } from './CommandSection';
import { SliderRow } from './SliderRow';

interface InteractionSectionProps {
  modes: InteractionModeDescriptor[];
  mode: InteractionModeId;
  /** Values for every mode, keyed as the engine reports them. */
  params: Readonly<Record<string, Readonly<Record<string, number>>>>;
  onModeChange: (mode: InteractionModeId) => void;
  onParamChange: (mode: InteractionModeId, name: string, value: number) => void;
}

/**
 * What the pointer does, and the numbers behind it.
 *
 * Only the active mode's parameters are drawn: three modes' worth at once would be most
 * of the panel, and a number you cannot currently exercise is noise. Which gesture drives
 * which mode is not configurable and so has no control here; the one rule, that a press
 * hitting something is the mode's and a press that misses is the camera's, is in the hint.
 */
export function InteractionSection({
  modes,
  mode,
  params,
  onModeChange,
  onParamChange,
}: InteractionSectionProps) {
  if (modes.length === 0) {
    return null;
  }
  const active = modes.find((m) => m.id === mode);
  return (
    <CommandSection label="Interaction">
      <Box px="xs" pb="0.5em">
        <SegmentedControl
          fullWidth
          size="xs"
          radius="xs"
          value={mode}
          onChange={(next) => onModeChange(next as InteractionModeId)}
          data={modes.map((m) => ({ value: m.id, label: m.label, disabled: !m.available }))}
        />
      </Box>
      {active && (
        <Box px="xs" pb="0.5em">
          <Text c="dimmed" style={{ fontSize: '0.8em', lineHeight: 1.4 }}>
            {active.available ? active.hint : (active.reason ?? 'Not available in this scene.')}
          </Text>
        </Box>
      )}
      {active?.available &&
        active.params.map((param) => (
          <SliderRow
            key={param.name}
            id={`interaction:${mode}:${param.name}`}
            label={param.unit ? `${param.label} (${param.unit})` : param.label}
            value={params[mode]?.[param.name] ?? param.default}
            min={param.min}
            max={param.max}
            step={param.step}
            onChange={(value) => onParamChange(mode, param.name, value)}
          />
        ))}
    </CommandSection>
  );
}
