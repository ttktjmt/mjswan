/**
 * What a policy's camera sensors last saw, in a corner of the viewer: the image a traced
 * observation read, not a separate render, so it shows exactly what the policy got.
 */

/** On-screen height of each image; the pixels are scaled up without smoothing. */
const FEED_HEIGHT_PX = 128;

export class CameraFeed {
  private readonly root: HTMLDivElement;
  private readonly canvases = new Map<number, HTMLCanvasElement>();
  private visible = false;

  constructor(container: HTMLElement) {
    const root = document.createElement('div');
    root.dataset.mjswanCameraFeed = '';
    Object.assign(root.style, {
      position: 'absolute',
      left: '12px',
      bottom: '12px',
      display: 'none',
      gap: '8px',
      pointerEvents: 'none',
    });
    container.appendChild(root);
    this.root = root;
  }

  get shown(): boolean {
    return this.visible;
  }

  show(visible: boolean): void {
    this.visible = visible;
    this.root.style.display = visible ? 'flex' : 'none';
  }

  /** Draw camera `cameraId`'s RGBA image, rows bottom first as WebGL reads them. */
  update(cameraId: number, pixels: Uint8Array, width: number, height: number): void {
    if (!this.visible) return;
    const canvas = this.canvas(cameraId, width, height);
    const context = canvas.getContext('2d');
    if (!context) return;
    const image = context.createImageData(width, height);
    const rowBytes = width * 4;
    for (let row = 0; row < height; row++) {
      const src = (height - 1 - row) * rowBytes;
      image.data.set(pixels.subarray(src, src + rowBytes), row * rowBytes);
      for (let col = 0; col < width; col++) image.data[row * rowBytes + col * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  /** Drop the images of a policy no longer loaded. */
  clear(): void {
    for (const canvas of this.canvases.values()) canvas.remove();
    this.canvases.clear();
  }

  dispose(): void {
    this.clear();
    this.root.remove();
  }

  private canvas(cameraId: number, width: number, height: number): HTMLCanvasElement {
    let canvas = this.canvases.get(cameraId);
    if (!canvas || canvas.width !== width || canvas.height !== height) {
      canvas?.remove();
      canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      Object.assign(canvas.style, {
        height: `${FEED_HEIGHT_PX}px`,
        width: `${(FEED_HEIGHT_PX * width) / height}px`,
        imageRendering: 'pixelated',
        border: '1px solid rgba(255, 255, 255, 0.7)',
        borderRadius: '4px',
      });
      this.root.appendChild(canvas);
      this.canvases.set(cameraId, canvas);
    }
    return canvas;
  }
}
