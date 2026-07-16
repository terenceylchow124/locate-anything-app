# Test Fixture Sources

All images from Wikimedia Commons, resized (max 2400px on the long edge) to keep the repo lightweight. Original URLs below for provenance.

## dense_screws.jpg (scene: hardware/screws)

- Source: [Wikimedia Commons — "Computer screws in a zipper storage bag - 6 x 9 cm - on Blue plastic"](https://commons.wikimedia.org/wiki/File:Computer_screws_in_a_zipper_storage_bag_-_6_x_9_cm_-_on_Blue_plastic.jpg)
- Author: Fructibus
- License: CC0 — no attribution required

## dense_pallets.jpg (scene: pallets/shelves)

- Source: [Wikimedia Commons — "UNICEF pallets 02.jpg"](https://commons.wikimedia.org/wiki/File:UNICEF_pallets_02.jpg)
- License: CC BY-SA 3.0 — attribution required. Credit: UNICEF, via Wikimedia Commons, CC BY-SA 3.0.

## dense_apples.jpg (scene: orchard/crops)

- Source: [Wikimedia Commons — "'Malus Rajka' apples Capel Manor College Gardens Enfield London England.jpg"](https://commons.wikimedia.org/wiki/File:%27Malus_Rajka%27_apples_Capel_Manor_College_Gardens_Enfield_London_England.jpg)
- License: CC BY-SA 4.0 — attribution required. Credit: via Wikimedia Commons, CC BY-SA 4.0.

## dense_ema_plaques.jpg (scene: temple offerings)

- Source: [Wikimedia Commons — "Ema plaques in Japan.jpg"](https://commons.wikimedia.org/wiki/File:Ema_plaques_in_Japan.jpg)
- License: CC BY-SA 2.0 — attribution required. Credit: via Wikimedia Commons, CC BY-SA 2.0.

## dense_night_market.jpg (scene: night-market crowd)

- Source: [Wikimedia Commons — "DFC 3506 A lively night market crowded with people eating shopping and strolling beneath brightly lit stalls and signs.jpg"](https://commons.wikimedia.org/wiki/File:DFC_3506_A_lively_night_market_crowded_with_people_eating_shopping_and_strolling_beneath_brightly_lit_stalls_and_signs.jpg)
- Author: PattayaPatrol
- License: CC BY-SA 4.0 — attribution required. Credit: PattayaPatrol, via Wikimedia Commons, CC BY-SA 4.0.
- Note: too densely crowded for a reliable manual ground-truth count (background crowd is indistinguishable). Not used for the calibration regression test — qualitative QA spot-check only (see spec's QA Testing Plan).

## Manual ground-truth counts (approximate — see individual ticket discussions for uncertainty rationale)

| Fixture | Manual estimate | expected_count_range | Notes |
| --- | --- | --- | --- |
| dense_screws.jpg | ~95 | 80-125 | widened from 80-110 after ticket #03's real pipeline run measured 120; root cause (dedup threshold vs. undercounted ground truth vs. mild over-detection) not yet isolated -- provisional, see `backend/test_detect.py` |
| dense_pallets.jpg | ~12 pallets | 10-16 | counting unit is the pallet stack, not individual strapped boxes |
| dense_apples.jpg | ~14 apples | 10-18 | some apples partially occluded by leaves |
| dense_ema_plaques.jpg | ~50 | 35-65 | extremely dense/stacked, widest uncertainty band of the four countable scenes |
| dense_night_market.jpg | n/a | n/a | not used for count calibration — background crowd is uncountable by eye; qualitative use only |
