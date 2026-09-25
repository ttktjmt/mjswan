import { Box, Checkbox, SegmentedControl, Tooltip } from '@mantine/core';

import type { InteractionModeDescriptor, InteractionModeId } from '../engine';
import { CommandSection } from './CommandSection';
import { LabeledInput } from './LabeledInput';
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
 * Only the active mode's parameters are drawn, in a folder of their own: one level down,
 * as a command group is, so the label, slider and number columns line up with the
 * commands above. A mode the scene cannot run says why on hover.
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
    <CommandSection label="Interact" expandByDefault={false}>
      <Box px="xs" pb="0.5em">
        <SegmentedControl
          fullWidth
          size="xs"
          radius="xs"
          // The panel's buttons print at `sm`; `xs` would set the mode names smaller than
          // the Reset below them.
          styles={{ label: { fontSize: 'var(--mantine-font-size-sm)' } }}
          value={mode}
          onChange={(next) => onModeChange(next as InteractionModeId)}
          data={modes.map((m) => ({
            value: m.id,
            disabled: !m.available,
            label: m.available ? (
              m.label
            ) : (
              <Tooltip label={m.reason ?? 'Not available in this scene.'} withArrow>
                <span style={{ pointerEvents: 'auto' }}>{m.label}</span>
              </Tooltip>
            ),
          }))}
        />
      </Box>
      {active?.available && (
        <CommandSection label={active.label}>
          {active.params.map((param) => {
            const id = `interaction:${mode}:${param.name}`;
            const label = param.unit ? `${param.label} (${param.unit})` : param.label;
            const value = params[mode]?.[param.name] ?? param.default;
            const set = (next: number) => onParamChange(mode, param.name, next);
            if (param.type === 'checkbox') {
              return (
                <LabeledInput key={param.name} id={id} label={label}>
                  <Checkbox
                    id={id}
                    checked={value >= 0.5}
                    onChange={(event) => set(event.currentTarget.checked ? 1 : 0)}
                    size="xs"
                  />
                </LabeledInput>
              );
            }
            return (
              <SliderRow
                key={param.name}
                id={id}
                label={label}
                value={value}
                min={param.softMin ?? param.min}
                max={param.softMax ?? param.max}
                inputMin={param.min}
                inputMax={param.max}
                step={param.step}
                onChange={set}
              />
            );
          })}
        </CommandSection>
      )}
    </CommandSection>
  );
}
