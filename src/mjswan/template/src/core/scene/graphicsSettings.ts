/**
 * A scene's graphics settings, its manifest's `graphics` (#232), as values: what the
 * manifest parser and a host share with the engine, with nothing of three.js in it.
 */

export type ColorManagement = 'classic' | 'filament';
/** three.js's operators, and MuJoCo's two that three.js has no name for. */
export type ToneMapping =
  | 'none'
  | 'linear'
  | 'reinhard'
  | 'cineon'
  | 'aces_filmic'
  | 'agx'
  | 'neutral'
  | 'aces'
  | 'filmic';
/** A {@link ToneMapping}, or MuJoCo's name for one: `pbr_neutral`, `aces_legacy`. */
export type ToneMappingName = ToneMapping | 'pbr_neutral' | 'aces_legacy';
export type ColorSpaceName = 'srgb' | 'linear';

export interface GraphicsSettings {
  /** How colours become pixels, as MuJoCo's classic renderer or its Filament renderer; three.js draws both. */
  colorManagement: ColorManagement;
  /** Filament only. */
  toneMapping: ToneMapping;
  /** Filament only: in stops, before tone mapping, as MuJoCo's `filament.cg.exposure`. */
  exposure: number;
  /** Filament only: `srgb` encodes the output, `linear` leaves it. */
  outputColorSpace: ColorSpaceName;
  /** Filament only: `srgb` decodes an `rgba`, `linear` takes it as written. */
  baseColorType: ColorSpaceName;
  shadows: boolean;
  reflections: boolean;
  /** Drawn only with a skybox. */
  haze: boolean;
}

/** Which of the three switches change anything in the loaded model, by MuJoCo's rules. */
export interface GraphicsAvailable {
  /** A light casts shadows. */
  shadows: boolean;
  /** An opaque plane or box has a `reflectance`. */
  reflections: boolean;
  /** An infinite plane meets a skybox. */
  haze: boolean;
}

/** Before a model loads, nothing is drawn. */
export const NO_GRAPHICS_AVAILABLE: Readonly<GraphicsAvailable> = Object.freeze({
  shadows: false,
  reflections: false,
  haze: false,
});

/** What a host may pass: any of the settings, a tone mapping by any of its names. */
export type GraphicsInput = Partial<Omit<GraphicsSettings, 'toneMapping'> & { toneMapping: ToneMappingName }>;

/** MuJoCo's: what a document without `graphics` is drawn with. */
export const DEFAULT_GRAPHICS: Readonly<GraphicsSettings> = Object.freeze({
  colorManagement: 'classic',
  toneMapping: 'neutral',
  exposure: 0,
  outputColorSpace: 'srgb',
  baseColorType: 'srgb',
  shadows: true,
  reflections: true,
  haze: true,
});

/** Every tone mapping, in the order a menu lists them. */
export const TONE_MAPPINGS: readonly ToneMapping[] = [
  'none',
  'linear',
  'reinhard',
  'cineon',
  'aces_filmic',
  'agx',
  'neutral',
  'aces',
  'filmic',
];

const SYNONYMS: Readonly<Record<string, ToneMapping>> = { pbr_neutral: 'neutral', aces_legacy: 'aces_filmic' };

/** A tone mapping by any of its names, or null for one nobody draws. */
export function toneMappingNamed(name: unknown): ToneMapping | null {
  if (typeof name !== 'string') return null;
  if ((TONE_MAPPINGS as readonly string[]).includes(name)) return name as ToneMapping;
  return Object.prototype.hasOwnProperty.call(SYNONYMS, name) ? SYNONYMS[name] : null;
}

const colorSpace = (value: unknown) => (value === 'srgb' || value === 'linear' ? value : null);
const flag = (value: unknown) => (typeof value === 'boolean' ? value : null);
const CHECKS: { [K in keyof GraphicsSettings]: (value: unknown) => GraphicsSettings[K] | null } = {
  colorManagement: (value) => (value === 'classic' || value === 'filament' ? value : null),
  toneMapping: toneMappingNamed,
  exposure: (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null),
  outputColorSpace: colorSpace,
  baseColorType: colorSpace,
  shadows: flag,
  reflections: flag,
  haze: flag,
};

/** `input` over `base`; a value this engine cannot draw keeps `base`'s, with a warning. */
export function mergeGraphics(base: GraphicsSettings, input: GraphicsInput | null | undefined): GraphicsSettings {
  const next: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(input ?? {})) {
    if (value === undefined || !Object.prototype.hasOwnProperty.call(CHECKS, key)) continue;
    const checked = CHECKS[key as keyof GraphicsSettings](value);
    if (checked === null) {
      console.warn(`[mjswan] graphics: ${key} ${JSON.stringify(value)} cannot be drawn; keeping ${JSON.stringify(next[key])}.`);
      continue;
    }
    next[key] = checked;
  }
  return next as unknown as GraphicsSettings;
}

/** Whether an `rgba` is decoded from sRGB: Filament reads it so by default; classic takes it as written. */
export function decodesBaseColor(settings: GraphicsSettings): boolean {
  return settings.colorManagement === 'filament' && settings.baseColorType === 'srgb';
}
