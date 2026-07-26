import scenesData from "./scenes.json";
import type { Scene } from "./types";

// Scene data lives in scenes.json (add a 6th scene there + one image under
// public/scenes/ -- no code changes needed). JSON has no comments, so one
// data note lives here instead: hardware-screws' expected_count_range was
// widened 80-110 -> 80-125 after a real calibration run measured 120 -- see
// backend/test_detect.py's CALIBRATION_CASES docstring for the full
// rationale (same range, kept in sync by hand on both sides).
export const SCENES: Scene[] = scenesData as Scene[];

// Which scene loads by default -- VITE_DEFAULT_SCENE_ID (see .env.example),
// baked in at frontend build time like VITE_API_BASE_URL (frontend/Dockerfile).
// Falls back to the first scene in the registry if unset or if it names a
// scene that doesn't exist (typo, or a scene that got removed from scenes.json).
const requestedDefaultId = import.meta.env.VITE_DEFAULT_SCENE_ID;
export const DEFAULT_SCENE: Scene =
  SCENES.find((s) => s.id === requestedDefaultId) ?? SCENES[0];
