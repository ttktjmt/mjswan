import { Box, Button, Checkbox, Group, SegmentedControl, Select } from '@mantine/core';
import { useClipboard } from '@mantine/hooks';
import { IconCopy, IconRefresh } from '@tabler/icons-react';

import {
  TONE_MAPPINGS,
  type GraphicsAvailable,
  type GraphicsInput,
  type GraphicsSettings,
  type ColorManagement,
  type ToneMapping,
} from '../engine';
import { CommandSection } from './CommandSection';
import { LabeledInput } from './LabeledInput';
import { SliderRow } from './SliderRow';

const TONE_MAPPING_LABELS: Record<ToneMapping, string> = {
  none: 'None',
  linear: 'Linear',
  reinhard: 'Reinhard',
  cineon: 'Cineon',
  aces_filmic: 'ACES Filmic',
  agx: 'AgX',
  neutral: 'Neutral',
  aces: 'ACES',
  filmic: 'Filmic',
};

/** Each setting's `ViewerConfig` field, and whether only Filament reads it. */
const PYTHON_FIELDS: ReadonlyArray<[keyof GraphicsSettings, string, boolean]> = [
  ['colorManagement', 'color_management', false],
  ['toneMapping', 'tone_mapping', true],
  ['exposure', 'exposure', true],
  ['outputColorSpace', 'output_color_space', true],
  ['baseColorType', 'base_color_type', true],
  ['shadows', 'enable_shadows', false],
  ['reflections', 'enable_reflections', false],
  ['haze', 'enable_haze', false],
];

function pythonValue(value: string | number | boolean): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return String(Number(value.toFixed(3)));
  return `"${value}"`;
}

/**
 * The settings drawn now that differ from the scene's own, as the call that builds them
 * in. Back to classic, the Filament settings are cleared: a classic `ViewerConfig` refuses them.
 */
export function graphicsAsPython(current: GraphicsSettings, scene: GraphicsSettings): string {
  const classic = current.colorManagement === 'classic';
  const args = PYTHON_FIELDS.flatMap(([key, field, filamentOnly]) => {
    if (filamentOnly && classic) return scene.colorManagement === 'classic' ? [] : [`${field}=None`];
    return current[key] === scene[key] ? [] : [`${field}=${pythonValue(current[key])}`];
  });
  return `from dataclasses import replace\n\nscene.set_viewer(replace(scene.viewer${args.map((arg) => `, ${arg}`).join('')}))\n`;
}

/**
 * How the scene is drawn, mjswan's own: mjlab's viewer has no such controls. Collapsed
 * by default, apart from the Scene section recorded from mjlab.
 */
export function GraphicsSection({
  graphics,
  sceneGraphics,
  available,
  onChange,
  onReset,
}: {
  graphics: GraphicsSettings;
  /** The scene's own settings, which Reset returns to and Copy as Python compares with. */
  sceneGraphics: GraphicsSettings;
  /** The switches that change anything in this model; the others are offered disabled. */
  available?: GraphicsAvailable;
  onChange: (input: GraphicsInput) => void;
  onReset: () => void;
}) {
  const clipboard = useClipboard({ timeout: 1500 });
  const checkbox = (key: 'shadows' | 'reflections' | 'haze', label: string) => (
    <LabeledInput id={`graphics:${key}`} label={label}>
      <Checkbox
        id={`graphics:${key}`}
        checked={graphics[key]}
        disabled={available ? !available[key] : false}
        onChange={(event) => onChange({ [key]: event.currentTarget.checked })}
        size="xs"
      />
    </LabeledInput>
  );
  return (
    <CommandSection label="Graphics" expandByDefault={false}>
      <LabeledInput label="Color management">
        <SegmentedControl
          fullWidth
          size="xs"
          radius="xs"
          aria-label="Color management"
          styles={{ label: { fontSize: 'var(--mantine-font-size-sm)' } }}
          value={graphics.colorManagement}
          onChange={(value) => onChange({ colorManagement: value as ColorManagement })}
          data={[
            { value: 'classic', label: 'Classic' },
            { value: 'filament', label: 'Filament' },
          ]}
        />
      </LabeledInput>
      {graphics.colorManagement === 'filament' && (
        <>
          <LabeledInput id="graphics:tone-mapping" label="Tone mapping">
            <Select
              id="graphics:tone-mapping"
              data={TONE_MAPPINGS.map((value) => ({ value, label: TONE_MAPPING_LABELS[value] }))}
              value={graphics.toneMapping}
              onChange={(toneMapping) => toneMapping && onChange({ toneMapping: toneMapping as ToneMapping })}
              allowDeselect={false}
              size="xs"
              radius="xs"
              styles={{ input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em' } }}
              comboboxProps={{ zIndex: 1000 }}
            />
          </LabeledInput>
          <SliderRow
            id="graphics:exposure"
            label="Exposure"
            value={graphics.exposure}
            min={-3}
            max={3}
            inputMin={-10}
            inputMax={10}
            step={0.1}
            onChange={(exposure) => onChange({ exposure })}
          />
        </>
      )}
      {checkbox('shadows', 'Shadows')}
      {checkbox('reflections', 'Reflections')}
      {checkbox('haze', 'Haze')}
      <Box px="xs" pb="0.5em">
        <Group gap="xs" wrap="nowrap">
          <Button
            variant="light"
            size="sm"
            px="xs"
            style={{ height: '2em', flexShrink: 0 }}
            leftSection={<IconRefresh size={14} />}
            onClick={onReset}
          >
            Reset
          </Button>
          <Button
            variant="light"
            size="sm"
            px="xs"
            style={{ height: '2em', flexGrow: 1 }}
            leftSection={<IconCopy size={14} />}
            onClick={() => clipboard.copy(graphicsAsPython(graphics, sceneGraphics))}
          >
            {clipboard.copied ? 'Copied' : 'Copy as Python'}
          </Button>
        </Group>
      </Box>
    </CommandSection>
  );
}
