import { Flex, NumberInput, Slider } from '@mantine/core';

import { LabeledInput } from './LabeledInput';

/** The bound as printed under a track end, without float noise. */
export function formatBound(value: number): string {
  return String(Number(value.toFixed(3)));
}

/** A labelled slider with a number box: the panel's one numeric row. */
export function SliderRow({
  id,
  label,
  value,
  min,
  max,
  inputMin = min,
  inputMax = max,
  step,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  /**
   * How far the number box reaches, when a typed value may go past the slider. The thumb
   * then pins at the end of the track.
   */
  inputMin?: number;
  inputMax?: number;
  /** A slider descriptor always carries one; Mantine's own default stands in if not. */
  step?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <LabeledInput id={id} label={label}>
      <Flex justify="space-between">
        <Slider
          id={id}
          value={value}
          onChange={onChange}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          marks={[
            { value: min, label: formatBound(min) },
            { value: max, label: formatBound(max) },
          ]}
          style={{ flexGrow: 1 }}
        />
        <NumberInput
          value={value}
          onChange={(next) => {
            const parsed = typeof next === 'number' ? next : Number(next);
            if (Number.isFinite(parsed)) onChange(parsed);
          }}
          min={inputMin}
          max={inputMax}
          step={step}
          size="xs"
          hideControls
          clampBehavior="strict"
          decimalScale={3}
          disabled={disabled}
          style={{ width: '3rem', marginLeft: 'var(--mantine-spacing-xs)' }}
        />
      </Flex>
    </LabeledInput>
  );
}
