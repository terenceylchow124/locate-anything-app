import type { DetectResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// Decode mode is fixed to hybrid for this ticket, not user-selectable (see
// spec's Implementation Decisions, ticket #04).
const MODE = "hybrid";

export async function detect(image: Blob, prompt: string): Promise<DetectResponse> {
  const formData = new FormData();
  formData.append("file", image, "image");
  formData.append("prompt", prompt);
  formData.append("mode", MODE);

  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `detect failed with status ${response.status}`);
  }

  return response.json();
}
