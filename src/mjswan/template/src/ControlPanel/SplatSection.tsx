import { useEffect, useState } from 'react';
import { Box, Slider, Text } from '@mantine/core';
import { CommandSection } from './CommandSection';

interface SplatSectionProps {
  /** Initial scale value from config. */
  scale: number;
  /** Initial ground offset value from config. */
  groundOffset: number;
  /** Called whenever scale or ground offset is adjusted. */
  onCalibrate: (scale: number, groundOffset: number) => void;
}

/** Dev-mode calibration controls for a Gaussian Splat background. */
export function SplatSection({ scale: initialScale, groundOffset: initialGroundOffset, onCalibrate }: SplatSectionProps) {
  const [scale, setScale] = useState(initialScale);
  const [groundOffset, setGroundOffset] = useState(initialGroundOffset);

  // Sync when the config changes (e.g. switching scenes)
  useEffect(() => { setScale(initialScale); }, [initialScale]);
  useEffect(() => { setGroundOffset(initialGroundOffset); }, [initialGroundOffset]);

  const handleScaleChange = (val: number) => {
    setScale(val);
    onCalibrate(val, groundOffset);
  };

  const handleGroundOffsetChange = (val: number) => {
    setGroundOffset(val);
    onCalibrate(scale, val);
  };

  return (
    <CommandSection label="Splat Control" expandByDefault={true}>
      <Box px="xs" pb="xs">
        <Box pb="xs">
          <Text c="dimmed" style={{ fontSize: '0.8em', marginBottom: '0.25em' }}>
            Scale: {scale.toFixed(3)}
          </Text>
          <Slider
            value={scale}
            onChange={handleScaleChange}
            min={0.1}
            max={5.0}
            step={0.001}
            size="xs"
            styles={{ root: { padding: '0' }, track: { height: 4 }, thumb: { width: 12, height: 12 } }}
          />
        </Box>
        <Box pb="xs">
          <Text c="dimmed" style={{ fontSize: '0.8em', marginBottom: '0.25em' }}>
            Ground offset: {groundOffset.toFixed(3)}
          </Text>
          <Slider
            value={groundOffset}
            onChange={handleGroundOffsetChange}
            min={-5.0}
            max={5.0}
            step={0.001}
            size="xs"
            styles={{ root: { padding: '0' }, track: { height: 4 }, thumb: { width: 12, height: 12 } }}
          />
        </Box>
      </Box>
    </CommandSection>
  );
}
