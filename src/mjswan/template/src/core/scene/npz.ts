export type NpzEntry = {
  shape: number[];
  data: Float32Array;
  strings?: string[];
};

export type NpzData = Record<string, NpzEntry>;

async function inflateRaw(compressed: Uint8Array, maxBytes = Infinity): Promise<Uint8Array> {
  const ds = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  const reader = ds.readable.getReader();
  // These reject only when the read below is cancelled, which throws its own error.
  writer.write(compressed as Uint8Array<ArrayBuffer>).catch(() => {});
  writer.close().catch(() => {});
  const chunks: Uint8Array[] = [];
  let totalLen = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    totalLen += value.byteLength;
    if (totalLen > maxBytes) {
      await reader.cancel();
      throw new RangeError(`npz: an entry inflates past the ${maxBytes} bytes this host allows`);
    }
  }
  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function parseNpyBuffer(buffer: Uint8Array): NpzEntry {
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
  for (let i = 0; i < magic.length; i++) {
    if (buffer[i] !== magic[i]) {
      throw new Error('Invalid .npy magic number');
    }
  }

  const major = buffer[6];
  let headerLen: number;
  let headerStart: number;
  if (major === 1) {
    headerLen = buffer[8] | (buffer[9] << 8);
    headerStart = 10;
  } else {
    headerLen =
      buffer[8] | (buffer[9] << 8) | (buffer[10] << 16) | (buffer[11] << 24);
    headerStart = 12;
  }
  // The header regexes are quadratic in its length.
  if (!(headerLen >= 0 && headerLen <= MAX_HEADER_LEN) || headerStart + headerLen > buffer.length) {
    throw new Error(`npz: a ${headerLen}-byte .npy header (at most ${MAX_HEADER_LEN})`);
  }

  const headerBytes = buffer.slice(headerStart, headerStart + headerLen);
  const header = new TextDecoder().decode(headerBytes);

  const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  const orderMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);

  if (!descrMatch || !shapeMatch) {
    throw new Error(`Cannot parse .npy header: ${header}`);
  }

  const descr = descrMatch[1];
  const fortranOrder = orderMatch ? orderMatch[1] === 'True' : false;
  if (fortranOrder) {
    throw new Error('Fortran-ordered arrays not supported');
  }

  const shapeParts = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const shape = shapeParts.map((s) => parseInt(s, 10));
  // Before any branch: an Infinity dimension would match an Infinity count from a zero width.
  if (!shape.every((d) => Number.isSafeInteger(d) && d >= 0)) {
    throw new Error(`npz: shape (${shapeMatch[1]}) is not a list of sizes`);
  }

  const dataOffset = headerStart + headerLen;
  const rawData = buffer.slice(dataOffset);

  const byteOrder = descr[0];
  const dtype = descr.slice(1);
  const needsSwap =
    byteOrder === '>' || (byteOrder === '=' && !isLittleEndian());

  let float32Data: Float32Array;
  if (dtype === 'f4') {
    float32Data = new Float32Array(
      rawData.buffer,
      rawData.byteOffset,
      rawData.byteLength / 4,
    );
    if (needsSwap) float32Data = swapEndian32(float32Data);
  } else if (dtype === 'f8') {
    let f64 = new Float64Array(
      rawData.buffer as ArrayBuffer,
      rawData.byteOffset,
      rawData.byteLength / 8,
    );
    if (needsSwap) f64 = swapEndian64(f64);
    float32Data = new Float32Array(f64.length);
    for (let i = 0; i < f64.length; i++) float32Data[i] = f64[i];
  } else if (dtype === 'i4') {
    let i32 = new Int32Array(
      rawData.buffer as ArrayBuffer,
      rawData.byteOffset,
      rawData.byteLength / 4,
    );
    if (needsSwap) i32 = swapEndianI32(i32);
    float32Data = new Float32Array(i32.length);
    for (let i = 0; i < i32.length; i++) float32Data[i] = i32[i];
  } else if (dtype === 'i8') {
    // int64 → float32: lossy beyond 2^24, but fits typical index/count fields (numpy default int on 64-bit).
    let bi64 = new BigInt64Array(
      rawData.buffer as ArrayBuffer,
      rawData.byteOffset,
      rawData.byteLength / 8,
    );
    if (needsSwap) bi64 = swapEndianBI64(bi64);
    float32Data = new Float32Array(bi64.length);
    for (let i = 0; i < bi64.length; i++) float32Data[i] = Number(bi64[i]);
  } else {
    // Handle fixed-width byte-string dtype: |S<n>
    const strMatch = descr.match(/^S(\d+)$/);
    if (strMatch) {
      const width = parseInt(strMatch[1], 10);
      if (!(width >= 1)) throw new Error(`npz: byte-string width ${strMatch[1]}`);
      const count = rawData.length / width;
      const expectedCount = shape.reduce((a, b) => a * b, 1);
      if (count !== expectedCount) {
        throw new Error(
          `npz: byte-string array size mismatch (shape ${shape} expects ${expectedCount} elements, got ${count})`
        );
      }
      // UTF-8 assumed; numpy U<n> (UCS-4) strings are not supported.
      const decoder = new TextDecoder('utf-8');
      const strings: string[] = [];
      for (let i = 0; i < count; i++) {
        const chunk = rawData.slice(i * width, (i + 1) * width);
        const end = chunk.indexOf(0);
        strings.push(decoder.decode(end >= 0 ? chunk.slice(0, end) : chunk));
      }
      return { shape, data: new Float32Array(0), strings };
    }
    // Handle NumPy Unicode dtype: <U<n> / >U<n> / =U<n> (UCS-4 codepoints).
    const unicodeMatch = dtype.match(/^U(\d+)$/);
    if (unicodeMatch) {
      const width = parseInt(unicodeMatch[1], 10);
      if (!(width >= 1)) throw new Error(`npz: unicode width ${unicodeMatch[1]}`);
      const bytesPerItem = width * 4;
      const count = rawData.length / bytesPerItem;
      const expectedCount = shape.reduce((a, b) => a * b, 1);
      if (count !== expectedCount) {
        throw new Error(
          `npz: unicode array size mismatch (shape ${shape} expects ${expectedCount} elements, got ${count})`
        );
      }
      const view = new DataView(rawData.buffer, rawData.byteOffset, rawData.byteLength);
      const littleEndian = !needsSwap;
      const strings: string[] = [];
      for (let i = 0; i < count; i++) {
        const codepoints: number[] = [];
        for (let j = 0; j < width; j++) {
          const codepoint = view.getUint32(i * bytesPerItem + j * 4, littleEndian);
          if (codepoint === 0) {
            break;
          }
          codepoints.push(codepoint);
        }
        strings.push(String.fromCodePoint(...codepoints));
      }
      return { shape, data: new Float32Array(0), strings };
    }
    if (dtype === 'O') {
      // numpy object dtype (pickled Python objects) cannot be deserialised in the browser; treated as a skip.
      throw new Error(PICKLED_OBJECT_DTYPE_MESSAGE);
    }
    throw new Error(`Unsupported numpy dtype: ${descr}`);
  }

  // The numeric branches size the data from the bytes, so the header could claim anything.
  const expected = shape.reduce((a, b) => a * b, 1);
  if (float32Data.length !== expected) {
    throw new Error(`npz: shape (${shape}) expects ${expected} values, the data holds ${float32Data.length}`);
  }
  return { shape, data: float32Data };
}

/** numpy's `_MAX_HEADER_SIZE`. */
const MAX_HEADER_LEN = 10_000;

const PICKLED_OBJECT_DTYPE_MESSAGE =
  'npz: pickled python objects (|O dtype) cannot be loaded in the browser';

function isLittleEndian(): boolean {
  const buf = new ArrayBuffer(2);
  new DataView(buf).setInt16(0, 1, true);
  return new Int16Array(buf)[0] === 1;
}

function swapEndian32(arr: Float32Array): Float32Array {
  const view = new DataView(arr.buffer, arr.byteOffset, arr.byteLength);
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = view.getFloat32(i * 4, true);
  }
  return out;
}

function swapEndian64(arr: Float64Array<ArrayBufferLike>): Float64Array<ArrayBuffer> {
  const view = new DataView(arr.buffer, arr.byteOffset, arr.byteLength);
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = view.getFloat64(i * 8, true);
  }
  return out;
}

function swapEndianI32(arr: Int32Array<ArrayBufferLike>): Int32Array<ArrayBuffer> {
  const view = new DataView(arr.buffer, arr.byteOffset, arr.byteLength);
  const out = new Int32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = view.getInt32(i * 4, true);
  }
  return out;
}

function swapEndianBI64(
  arr: BigInt64Array<ArrayBufferLike>,
): BigInt64Array<ArrayBuffer> {
  const view = new DataView(arr.buffer, arr.byteOffset, arr.byteLength);
  const out = new BigInt64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = view.getBigInt64(i * 8, true);
  }
  return out;
}

/**
 * Parse an NPZ archive (raw bytes) into Float32Arrays for each contained .npy, inflating
 * no more than `maxInflatedBytes` across its entries.
 */
export async function loadNpz(arrayBuffer: ArrayBuffer, maxInflatedBytes = Infinity): Promise<NpzData> {
  const bytes = new Uint8Array(arrayBuffer);
  let inflated = 0;

  const result: NpzData = {};
  let offset = 0;

  while (offset + 30 <= bytes.length) {
    // ZIP local file header signature: PK\x03\x04
    if (
      bytes[offset] !== 0x50 ||
      bytes[offset + 1] !== 0x4b ||
      bytes[offset + 2] !== 0x03 ||
      bytes[offset + 3] !== 0x04
    ) {
      break;
    }

    const compressionMethod = bytes[offset + 8] | (bytes[offset + 9] << 8);
    const uncompressedSizeRaw =
      (bytes[offset + 22] |
      (bytes[offset + 23] << 8) |
      (bytes[offset + 24] << 16) |
      (bytes[offset + 25] << 24)) >>> 0;
    let compressedSize =
      (bytes[offset + 18] |
      (bytes[offset + 19] << 8) |
      (bytes[offset + 20] << 16) |
      (bytes[offset + 21] << 24)) >>> 0;
    const fileNameLen = bytes[offset + 26] | (bytes[offset + 27] << 8);
    const extraLen = bytes[offset + 28] | (bytes[offset + 29] << 8);

    const fileNameBytes = bytes.slice(offset + 30, offset + 30 + fileNameLen);
    let fileName = new TextDecoder().decode(fileNameBytes);

    // Handle ZIP64: sentinel value 0xFFFFFFFF means actual size is in extra field
    if (compressedSize === 0xffffffff || uncompressedSizeRaw === 0xffffffff) {
      const extraStart = offset + 30 + fileNameLen;
      let ep = extraStart;
      while (ep + 4 <= extraStart + extraLen) {
        const tag = bytes[ep] | (bytes[ep + 1] << 8);
        const sz = bytes[ep + 2] | (bytes[ep + 3] << 8);
        if (tag === 0x0001) {
          let fp = ep + 4;
          if (uncompressedSizeRaw === 0xffffffff && fp + 8 <= ep + 4 + sz) {
            fp += 8; // skip uncompressed size (not needed)
          }
          if (compressedSize === 0xffffffff && fp + 8 <= ep + 4 + sz) {
            const lo = (bytes[fp] | (bytes[fp+1]<<8) | (bytes[fp+2]<<16) | (bytes[fp+3]<<24)) >>> 0;
            const hi = (bytes[fp+4] | (bytes[fp+5]<<8) | (bytes[fp+6]<<16) | (bytes[fp+7]<<24)) >>> 0;
            compressedSize = hi * 0x100000000 + lo;
          }
          break;
        }
        ep += 4 + sz;
      }
    }

    const dataStart = offset + 30 + fileNameLen + extraLen;
    const rawEntry = bytes.slice(dataStart, dataStart + compressedSize);

    offset = dataStart + compressedSize;

    // Skip directories
    if (fileName.endsWith('/')) continue;
    // Strip .npy extension for key name
    if (fileName.endsWith('.npy')) {
      fileName = fileName.slice(0, -4);
    }

    let npyBytes: Uint8Array;
    if (compressionMethod === 0) {
      npyBytes = rawEntry;
    } else if (compressionMethod === 8) {
      npyBytes = await inflateRaw(rawEntry, maxInflatedBytes - inflated);
      inflated += npyBytes.byteLength;
    } else {
      console.warn(`[NPZ] Unsupported compression method ${compressionMethod} for ${fileName}`);
      continue;
    }

    try {
      result[fileName] = parseNpyBuffer(npyBytes);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === PICKLED_OBJECT_DTYPE_MESSAGE) {
        console.debug(`[NPZ] Skipping pickled object field "${fileName}"`);
      } else {
        console.warn(`[NPZ] Failed to parse ${fileName}:`, err);
      }
    }
  }

  return result;
}
