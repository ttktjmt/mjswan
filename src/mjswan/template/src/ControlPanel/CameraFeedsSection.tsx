import { Box, Checkbox, Text } from '@mantine/core';
import { useEffect, useRef } from 'react';

import type { CameraFeedDescriptor } from '../engine';
import { CommandSection } from './CommandSection';
import { LabeledInput } from './LabeledInput';

/**
 * mjlab's "Camera Feeds" folder (`viewer/viser/viewer.py`, `camera_viewer.py`): for each
 * camera sensor, its image under the label mjlab gives it, then its "Frustum" switch.
 */
export function CameraFeedsSection({
  feeds,
  onAttach,
  onFrustumChange,
}: {
  feeds: CameraFeedDescriptor[];
  onAttach: (id: string, canvas: HTMLCanvasElement) => () => void;
  onFrustumChange: (id: string, shown: boolean) => void;
}) {
  return (
    <CommandSection label="Camera Feeds" expandByDefault={true}>
      {feeds.map((feed) => (
        <CameraFeed
          key={feed.id}
          feed={feed}
          onAttach={onAttach}
          onFrustumChange={onFrustumChange}
        />
      ))}
    </CommandSection>
  );
}

function CameraFeed({
  feed,
  onAttach,
  onFrustumChange,
}: {
  feed: CameraFeedDescriptor;
  onAttach: (id: string, canvas: HTMLCanvasElement) => () => void;
  onFrustumChange: (id: string, shown: boolean) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => (canvas.current ? onAttach(feed.id, canvas.current) : undefined), [feed.id, onAttach]);
  const frustumId = `camerafeed:${feed.id}:frustum`;
  return (
    <>
      <Box px="xs" pb="0.5em">
        <Text size="sm" style={{ lineHeight: '1.375em', overflowWrap: 'anywhere' }}>
          {feed.label}
        </Text>
        <canvas
          ref={canvas}
          width={feed.width}
          height={feed.height}
          aria-label={feed.label}
          style={{
            display: 'block',
            width: `${feed.width * feed.scale}px`,
            height: `${feed.height * feed.scale}px`,
            maxWidth: '100%',
            imageRendering: 'pixelated',
          }}
        />
      </Box>
      <LabeledInput id={frustumId} label="Frustum">
        <Checkbox
          id={frustumId}
          checked={feed.frustum}
          onChange={(event) => onFrustumChange(feed.id, event.currentTarget.checked)}
          size="xs"
        />
      </LabeledInput>
    </>
  );
}
