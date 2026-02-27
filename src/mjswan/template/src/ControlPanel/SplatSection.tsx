import { useCallback, useState } from 'react';
import { Box, Button, Slider, Stack, Text, TextInput } from '@mantine/core';
import { CommandSection } from './CommandSection';

interface SplatSectionProps {
  onLoad?: (url: string, scale: number, groundOffset: number) => void;
  onClear?: () => void;
  loaded?: boolean;
}

export function SplatSection({ onLoad, onClear, loaded = false }: SplatSectionProps) {
  const [url, setUrl] = useState('');
  const [scale, setScale] = useState(1.0);
  const [groundOffset, setGroundOffset] = useState(0.0);

  const handleLoad = useCallback(() => {
    if (url.trim()) onLoad?.(url.trim(), scale, groundOffset);
  }, [url, scale, groundOffset, onLoad]);

  return (
    <CommandSection label="Splat" expandByDefault={false}>
      <Box px="xs" pb="xs">
        <TextInput
          placeholder="https://...scene.spz"
          value={url}
          onChange={(e) => setUrl(e.currentTarget.value)}
          size="xs"
          radius="xs"
          styles={{ input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em', fontSize: '0.75em' } }}
          mb="xs"
        />
        <Box pb="xs">
          <Text c="dimmed" style={{ fontSize: '0.8em', marginBottom: '0.25em' }}>
            Scale: {scale.toFixed(2)}
          </Text>
          <Slider
            value={scale}
            onChange={setScale}
            min={0.1}
            max={5.0}
            step={0.01}
            size="xs"
            styles={{ root: { padding: '0' }, track: { height: 4 }, thumb: { width: 12, height: 12 } }}
          />
        </Box>
        <Box pb="xs">
          <Text c="dimmed" style={{ fontSize: '0.8em', marginBottom: '0.25em' }}>
            Ground offset: {groundOffset.toFixed(2)}
          </Text>
          <Slider
            value={groundOffset}
            onChange={setGroundOffset}
            min={-3.0}
            max={3.0}
            step={0.01}
            size="xs"
            styles={{ root: { padding: '0' }, track: { height: 4 }, thumb: { width: 12, height: 12 } }}
          />
        </Box>
        <Stack gap="xs">
          <Button
            variant="light"
            size="xs"
            fullWidth
            disabled={!url.trim()}
            onClick={handleLoad}
          >
            Load Splat
          </Button>
          {loaded && (
            <Button variant="subtle" color="gray" size="xs" fullWidth onClick={onClear}>
              Clear
            </Button>
          )}
        </Stack>
      </Box>
    </CommandSection>
  );
}
