import type { Scene } from "./types";

// Full 5-scene registry (ticket #05). Adding a 6th scene means adding one
// entry here + one image under public/scenes/ -- no other code changes.
export const SCENES: Scene[] = [
  {
    id: "hardware-screws",
    display_name: "Hardware / screws",
    category: "hardware",
    default_image: "/scenes/dense_screws.jpg",
    default_prompts: ["screw", "hex nut"],
    expected_count_range: [80, 125], // widened after ticket #03's calibration run measured 120
  },
  {
    id: "pallets-shelves",
    display_name: "Pallets / shelves",
    category: "logistics",
    default_image: "/scenes/dense_pallets.jpg",
    default_prompts: ["pallet", "stack of boxes"],
    expected_count_range: [10, 16],
  },
  {
    id: "orchard-crops",
    display_name: "Orchard / crops",
    category: "agriculture",
    default_image: "/scenes/dense_apples.jpg",
    default_prompts: ["apple"],
    expected_count_range: [10, 18],
  },
  {
    id: "temple-offerings",
    display_name: "Temple offerings",
    category: "culture",
    default_image: "/scenes/dense_ema_plaques.jpg",
    default_prompts: ["wooden plaque", "prayer plaque"],
    expected_count_range: [35, 65],
  },
  {
    id: "night-market-crowd",
    display_name: "Night-market crowd",
    category: "crowd",
    default_image: "/scenes/dense_night_market.jpg",
    default_prompts: ["person"],
    expected_count_range: null, // uncountable by eye -- qualitative QA only
  },
];
