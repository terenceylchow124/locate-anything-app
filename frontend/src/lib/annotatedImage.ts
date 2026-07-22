import type { Detection } from "../types";

type ColoredDetection = Detection & { color?: string };

/**
 * Composite the base image plus boxes onto an offscreen canvas at natural
 * resolution and download as PNG. Each detection may carry its own color
 * (overlay); otherwise `strokeColor` is used (result-grid cards). Pass the
 * already-thresholded detections so the saved image matches what's on screen.
 */
export async function downloadAnnotatedImage(
  imageUrl: string,
  detections: ColoredDetection[],
  strokeColor: string,
  strokeWidth: number,
  filename: string,
): Promise<void> {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = imageUrl;
  await img.decode();

  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.drawImage(img, 0, 0);
  const fontSize = Math.max(16, Math.round(canvas.height / 60));
  const halo = cssVar("--bg") ?? "#ffffff";

  for (const d of detections) {
    const [x1, y1, x2, y2] = d.box;
    const color = d.color ?? strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = d.score != null ? `${d.label} ${(d.score * 100).toFixed(0)}%` : d.label;
    const labelY = Math.max(fontSize, y1 - 4);
    ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
    ctx.textBaseline = "alphabetic";
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(3, fontSize / 5);
    ctx.strokeStyle = halo;
    ctx.strokeText(label, x1, labelY);
    ctx.fillStyle = color;
    ctx.fillText(label, x1, labelY);
  }

  await new Promise<void>((resolve) => {
    canvas.toBlob((blob) => {
      if (blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      }
      resolve();
    }, "image/png");
  });
}

function cssVar(name: string): string | undefined {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || undefined;
}
