import { useEffect, useRef, useState } from 'react';
import { Box, Slider, Text } from '@mantine/core';
import { CommandSection } from './CommandSection';

interface SplatSectionProps {
  scale: number;
  xOffset: number;
  yOffset: number;
  zOffset: number;
  /** Roll in degrees. */
  roll: number;
  /** Pitch in degrees. */
  pitch: number;
  /** Yaw in degrees. */
  yaw: number;
  onCalibrate: (scale: number, xOffset: number, yOffset: number, zOffset: number, roll: number, pitch: number, yaw: number) => void;
}

const SECTION_LABEL_STYLE = { fontSize: '0.875em', fontWeight: 450, lineHeight: '1.375em', letterSpacing: '-0.75px', width: '4.5em', flexShrink: 0 } as const;
const SLIDER_STYLES = { root: { padding: '0' }, track: { height: 4 }, thumb: { width: 12, height: 12 } } as const;

const INPUT_STYLE: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  height: '1.75em',
  fontSize: '0.8em',
  fontFamily: 'inherit',
  border: '1px solid var(--mantine-color-default-border)',
  borderRadius: 'var(--mantine-radius-sm)',
  background: 'var(--mantine-color-default)',
  color: 'var(--mantine-color-text)',
  textAlign: 'right',
  padding: '0 0.4em',
  outline: 'none',
  cursor: 'ns-resize',
  touchAction: 'none',
};

// --- ScrubInput ----------------------------------------------------------
// Blender-style: drag up/down to scrub, click to type directly in the field.

interface ScrubInputProps {
  axis: string;
  value: number;
  min?: number;
  max?: number;
  step: number;
  onChange: (v: number) => void;
}

function ScrubInput({ axis, value, min = -Infinity, max = Infinity, step, onChange }: ScrubInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragRef = useRef<{ startY: number; startValue: number; moved: boolean } | null>(null);

  // Keep displayed value in sync with prop unless the field has focus
  useEffect(() => {
    const el = inputRef.current;
    if (el && document.activeElement !== el) {
      el.value = value.toFixed(2);
    }
  }, [value]);

  const clamp = (v: number) =>
    Math.max(min, Math.min(max, parseFloat((Math.round(v / step) * step).toFixed(10))));

  const onPointerDown = (e: React.PointerEvent<HTMLInputElement>) => {
    // Prevent default so the input doesn't receive focus on drag start.
    // We'll focus it manually on click (no-drag) in onPointerUp.
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { startY: e.clientY, startValue: value, moved: false };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLInputElement>) => {
    if (!dragRef.current || !(e.buttons & 1)) return;
    const deltaY = dragRef.current.startY - e.clientY;
    if (Math.abs(deltaY) > 3) dragRef.current.moved = true;
    if (!dragRef.current.moved) return;
    const newVal = clamp(dragRef.current.startValue + deltaY * step);
    onChange(newVal);
    // Update display immediately without React re-render lag
    if (inputRef.current) inputRef.current.value = newVal.toFixed(2);
  };

  const onPointerUp = () => {
    if (dragRef.current && !dragRef.current.moved) {
      // Plain click — enter text-edit mode
      const el = inputRef.current;
      if (el) { el.focus(); el.select(); }
    }
    dragRef.current = null;
  };

  const onBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    const n = parseFloat(e.target.value);
    if (!isNaN(n)) onChange(clamp(n));
    // Reset to canonical value
    e.target.value = value.toFixed(2);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') e.currentTarget.blur();
    if (e.key === 'Escape') {
      if (inputRef.current) inputRef.current.value = value.toFixed(2);
      inputRef.current?.blur();
    }
  };

  return (
    <Box style={{ display: 'flex', alignItems: 'center', gap: '0.25em', flex: 1 }}>
      <Text c="dimmed" style={{ fontSize: '0.875em', fontWeight: 450, letterSpacing: '-0.75px', flexShrink: 0 }}>{axis}</Text>
      <input
        ref={inputRef}
        size={1}
        defaultValue={value.toFixed(2)}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onBlur={onBlur}
        onKeyDown={onKeyDown}
        onFocus={(e) => e.currentTarget.style.outline = '2px solid var(--mantine-color-blue-5)'}
        onBlurCapture={(e) => e.currentTarget.style.outline = 'none'}
        style={INPUT_STYLE}
      />
    </Box>
  );
}

// --- VectorRow ----------------------------------------------------------

interface VectorRowProps {
  label: string;
  x: number; y: number; z: number;
  step: number;
  xLabel?: string; yLabel?: string; zLabel?: string;
  onX: (v: number) => void;
  onY: (v: number) => void;
  onZ: (v: number) => void;
}

function VectorRow({ label, x, y, z, step, xLabel = 'X', yLabel = 'Y', zLabel = 'Z', onX, onY, onZ }: VectorRowProps) {
  return (
    <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center', gap: '0.5em' }}>
      <Text c="dimmed" style={SECTION_LABEL_STYLE}>{label}</Text>
      <ScrubInput axis={xLabel} value={x} step={step} onChange={onX} />
      <ScrubInput axis={yLabel} value={y} step={step} onChange={onY} />
      <ScrubInput axis={zLabel} value={z} step={step} onChange={onZ} />
    </Box>
  );
}

// --- SplatSection -------------------------------------------------------

/** Dev-mode calibration controls for a Gaussian Splat background. */
export function SplatSection({ scale: initialScale, xOffset: initialXOffset, yOffset: initialYOffset, zOffset: initialZOffset, roll: initialRoll, pitch: initialPitch, yaw: initialYaw, onCalibrate }: SplatSectionProps) {
  const [scale, setScale] = useState(initialScale);
  const [xOffset, setXOffset] = useState(initialXOffset);
  const [yOffset, setYOffset] = useState(initialYOffset);
  const [zOffset, setZOffset] = useState(initialZOffset);
  const [roll, setRoll] = useState(initialRoll);
  const [pitch, setPitch] = useState(initialPitch);
  const [yaw, setYaw] = useState(initialYaw);

  useEffect(() => { setScale(initialScale); }, [initialScale]);
  useEffect(() => { setXOffset(initialXOffset); }, [initialXOffset]);
  useEffect(() => { setYOffset(initialYOffset); }, [initialYOffset]);
  useEffect(() => { setZOffset(initialZOffset); }, [initialZOffset]);
  useEffect(() => { setRoll(initialRoll); }, [initialRoll]);
  useEffect(() => { setPitch(initialPitch); }, [initialPitch]);
  useEffect(() => { setYaw(initialYaw); }, [initialYaw]);

  const call = (s: number, x: number, y: number, z: number, r: number, p: number, w: number) => onCalibrate(s, x, y, z, r, p, w);

  return (
    <CommandSection label="Control" expandByDefault={true}>
      <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center', gap: '0.5em' }}>
        <Text c="dimmed" style={SECTION_LABEL_STYLE}>Scale</Text>
        <Box style={{ flex: 1 }}>
          <Slider value={scale} onChange={(val) => { setScale(val); call(val, xOffset, yOffset, zOffset, roll, pitch, yaw); }} min={0.1} max={5.0} step={0.05} size="xs" label={(val) => val.toFixed(2)} styles={SLIDER_STYLES} />
        </Box>
      </Box>
      <VectorRow
        label="Position" x={xOffset} y={yOffset} z={zOffset} step={0.05}
        onX={(val) => { setXOffset(val); call(scale, val, yOffset, zOffset, roll, pitch, yaw); }}
        onY={(val) => { setYOffset(val); call(scale, xOffset, val, zOffset, roll, pitch, yaw); }}
        onZ={(val) => { setZOffset(val); call(scale, xOffset, yOffset, val, roll, pitch, yaw); }}
      />
      <VectorRow
        label="Rotation" x={roll} y={pitch} z={yaw} step={0.5}
        xLabel="R" yLabel="P" zLabel="Y"
        onX={(val) => { setRoll(val);  call(scale, xOffset, yOffset, zOffset, val, pitch, yaw); }}
        onY={(val) => { setPitch(val); call(scale, xOffset, yOffset, zOffset, roll, val, yaw); }}
        onZ={(val) => { setYaw(val);   call(scale, xOffset, yOffset, zOffset, roll, pitch, val); }}
      />
    </CommandSection>
  );
}
