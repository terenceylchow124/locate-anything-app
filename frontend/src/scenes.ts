import type { Scene } from "./types";

// Scene data lives in public/scene-config/scenes.json, fetched at runtime
// (not imported at build time) so it can be bind-mounted into the running
// container (docker-compose.yml) -- edit it + drop an image under
// public/scene-config/images/ and refresh, no rebuild needed. Both live
// under one scene-config/ directory, mounted as a directory (covers the
// whole thing in one mount) rather than mounting individual files -- a
// single-file bind mount breaks the moment the
// file is saved via the common "write a new file, rename over the old one"
// pattern most editors/tools use (the mount tracks the original inode, which
// the rename orphans -- edits stop showing up in the container even though
// `cat`ing the file on the host looks correct). Mounting the containing
// directory instead sidesteps that: lookups inside a mounted directory always
// resolve through the current directory entries. One data note that doesn't
// fit in JSON (no comments): hardware-screws' expected_count_range was
// widened 80-110 -> 80-125 after a real calibration run measured 120 -- see
// backend/test_detect.py's CALIBRATION_CASES docstring for the full
// rationale (same range, kept in sync by hand on both sides).
export async function loadScenes(): Promise<Scene[]> {
  const response = await fetch("/scene-config/scenes.json");
  if (!response.ok) {
    throw new Error(`failed to load /scene-config/scenes.json: ${response.status}`);
  }
  const scenes = (await response.json()) as Scene[];
  if (!Array.isArray(scenes) || scenes.length === 0) {
    throw new Error("/scene-config/scenes.json did not contain a non-empty array of scenes");
  }
  return scenes;
}

// Which scene loads by default -- VITE_DEFAULT_SCENE_ID (see .env.example),
// baked in at frontend build time like VITE_API_BASE_URL (frontend/Dockerfile).
// Falls back to the first scene in the (runtime-loaded) list if unset or if
// it names a scene that doesn't exist (typo, or a scene removed from
// scenes.json since the image was built).
export function resolveDefaultScene(scenes: Scene[]): Scene {
  const requestedDefaultId = import.meta.env.VITE_DEFAULT_SCENE_ID;
  return scenes.find((s) => s.id === requestedDefaultId) ?? scenes[0];
}
