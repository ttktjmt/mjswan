import { Box, Button, Checkbox, Group, Tooltip } from '@mantine/core';

import type { HandTrackingDescriptor, XrSessionDescriptor, XrSessionId } from '../engine';
import { CommandSection } from './CommandSection';
import { LabeledInput } from './LabeledInput';

interface WebXrSectionProps {
  sessions: XrSessionDescriptor[];
  handTracking: HandTrackingDescriptor | null;
  /** Enter the session, or leave it while it runs. */
  onSessionPress: (id: XrSessionId, active: boolean) => void;
  onHandTrackingChange: (enabled: boolean) => void;
}

/**
 * The way into VR and AR, shown only where the device can start a session. The hand switch
 * waits for the next model build, which entering makes when the model disagrees with it.
 */
export function WebXrSection({ sessions, handTracking, onSessionPress, onHandTrackingChange }: WebXrSectionProps) {
  if (sessions.length === 0) {
    return null;
  }
  return (
    <CommandSection label="WebXR" expandByDefault={false}>
      {handTracking && (
        <LabeledInput id="webxr:hands" label="Hand tracking">
          <Tooltip
            label={handTracking.reason ?? 'Not available in this scene.'}
            disabled={handTracking.available}
            withArrow
          >
            {/* A disabled input takes no pointer events, so the tip hangs off its box. */}
            <span style={{ display: 'inline-flex' }}>
              <Checkbox
                id="webxr:hands"
                checked={handTracking.enabled}
                disabled={!handTracking.available}
                onChange={(event) => onHandTrackingChange(event.currentTarget.checked)}
                size="xs"
              />
            </span>
          </Tooltip>
        </LabeledInput>
      )}
      <Box px="xs" pb="0.5em">
        <Group grow gap="xs">
          {sessions.map((session) => (
            <Button
              key={session.id}
              size="sm"
              style={{ height: '2em' }}
              onClick={() => onSessionPress(session.id, session.active)}
            >
              {session.label}
            </Button>
          ))}
        </Group>
      </Box>
    </CommandSection>
  );
}
