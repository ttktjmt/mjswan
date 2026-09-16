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
  step,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
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
          min={min}
          max={max}
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
