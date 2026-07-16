import type { Scene } from "./types";

// One placeholder scene for ticket #04 -- ticket #05 expands this array to
// the full 5-scene registry (see docs/spec-locateanything-demo.md); no
// frontend code changes needed when it does.
export const SCENES: Scene[] = [
  {
    id: "hardware-screws",
    display_name: "Hardware / screws",
    category: "hardware",
    default_image: "/scenes/dense_screws.jpg",
    default_prompts: ["screw", "hex nut"],
    expected_count_range: [80, 125], // widened after ticket #03's calibration run measured 120
  },
];
