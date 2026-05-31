# `Params` reference

Every knob in the pipeline lives in the `Params` dataclass (§8). One instance `P` is
passed into every stage — there are no hidden globals. Current defaults (tuned on
`data_1`) are shown below.

```python
P = Params()   # the live configuration; re-run the §8 cell after editing
```

## Polarity

| field | default | meaning |
|---|---|---|
| `bright_features` | `True` | `True` = scratches are **brighter** than the glass (specular highlights on dark glass) → white top-hat, `D = max(0, I−B)`, no inversion. `False` = scratches **darker** (transmission lighting) → black-hat, `D = max(0, B−I)`, invert before threshold. |

This is the single most important flag — it flips the polarity of every operator. Set it
wrong and nothing downstream works.

## Preprocessing

| field | default | controls | tuning |
|---|---|---|---|
| `gaussian_ksize` | `10` | Gaussian low-pass kernel (forced odd) | raise to denoise more; lowers faint-scratch contrast |
| `median_ksize` | `5` | median window (dust removal) | larger = more dust killed; keep < scratch length |
| `clahe_clip` | `3.0` | CLAHE contrast clip limit | **lower** if CLAHE amplifies texture/noise |
| `clahe_tile` | `8` | CLAHE tile grid (8×8) | smaller tiles = more local, more noise |
| `bg_subtract_ksize` | `51` | background-estimate median window | must be ≫ scratch width |
| `tophat_ksize` | `15` | top-hat structuring element | ≈ largest scratch width to suppress |

## Frangi ridge filter

| field | default | controls |
|---|---|---|
| `frangi_sigmas` | `(1, 2, 3, 4)` | scales (px); each ≈ half-width of a target ridge. Add larger σ for wider scratches |
| `frangi_alpha` | `0.5` | plate-vs-line sensitivity (2D: minor effect) |
| `frangi_beta` | `0.5` | blob-vs-line sensitivity ($R_B$ weight) |
| `frangi_gamma` | `5.0` | structureness ($S$) weight; lower = more sensitive to faint ridges |

## Thresholding ⭐

| field | default | controls |
|---|---|---|
| `threshold_method` | **`'ridge_hysteresis'`** | which detector (see [thresholding.md](thresholding.md)). Options: `fixed`, `otsu`, `adaptive`, `tophat_otsu`, `bgsub_otsu`, `ridge_otsu`, `ridge_hysteresis`, `hysteresis` |
| `fixed_thresh` | `30` | constant `T` for `fixed` |
| `adaptive_block` | `51` | local-mean window for `adaptive` (forced odd) |
| `adaptive_C` | `-5` | offset subtracted from local mean |
| `hyst_low` | `0.03` | hysteresis grow cutoff (fraction of per-image peak Frangi) |
| `hyst_high` | `0.12` | hysteresis seed cutoff (fraction of per-image peak Frangi) |

**Tuning the hysteresis** (only affects `ridge_hysteresis` / `ridge_otsu`-adjacent paths):

- Missing faint scratches → **lower** both (e.g. `0.02 / 0.08`).
- Too much speckle → **raise** both (e.g. `0.05 / 0.20`).
- Keep `hyst_low < hyst_high`. They are fractions of the per-image peak, so they transfer
  across images of different contrast.

## Morphology (binary clean-up)

| field | default | controls | tuning |
|---|---|---|---|
| `open_ksize` | `3` | opening SE (kills speckles ≤ size) | set `0` to skip — preserves 1-px faint scratches, relies on component filter |
| `close_ksize` | `3` | closing SE (bridges intra-scratch gaps) | larger reconnects more, may merge dust into scratches |

## Connected-component filter (geometry gate)

| field | default | controls | tuning |
|---|---|---|---|
| `min_area` | `30` | drop components smaller than this (px) | raise to kill more small dust |
| `max_area` | `200_000` | drop components larger than this (px) | safety cap for runaway floods |
| `min_length` | `8` | min bbox `max(w,h)` | raise to require longer scratches |
| `min_aspect_ratio` | `2.0` | bbox elongation proxy (OR branch) | lower to admit stubbier scratches |
| `min_elongation` | `1.5` | fitted-ellipse elongation (OR branch) | lower to admit noisier ellipse fits |
| `max_solidity` | `1.10` | `area / hull_area` cap (> 1 allows OpenCV rounding) | lower to require thinner curves |
| `min_slenderness` | `1.5` | **skeleton dust gate** = `skel_len² / area` (AND) | **lower to ~1.0** if it eats faint short scratches; **raise to ~2.0** to be stricter about dust |

The keep rule:

```python
shape_ok = (aspect >= min_aspect_ratio) OR (elong >= min_elongation)
keep =  (min_area <= area <= max_area) AND (length >= min_length)
        AND shape_ok
        AND (solidity <= max_solidity)
        AND (slenderness >= min_slenderness)
```

`min_slenderness` is the gate that drops compact dust grains which fake elongation via a
noisy ellipse fit — see [pipeline.md](pipeline.md#4-connected-component-filter--filter_components-52)
for the measured distributions that set the `1.5` default.

---

## Tuning workflow

1. **Restart & Run All** so `P` is fresh. The §12 QA cell prints the active
   `threshold_method` — confirm it is `ridge_hysteresis`.
2. Use the **§12 QA** (`show_mask_qa`) to eyeball 6 images (2 per glass). Adjust
   `hyst_low/hyst_high` for recall, `min_slenderness`/`min_area` for dust.
3. Use the per-component `DataFrame` returned by `filter_components` (columns include
   `aspect`, `elongation`, `solidity`, `skel_len`, `width`, `slenderness`, `kept`) to see
   exactly why each component was kept or dropped — drop `kept=False` rows into a scatter
   plot to find a decision boundary.
4. Use **§13** (`compare_illumination`) when onboarding a new glass with different optics,
   to confirm the detector choice still holds.

> All of `compare_illumination` / `show_mask_qa` read the **current** global `P` at call
> time (parameter default is `p=None`), so editing `P` and re-running §8 is enough — you
> do not need to re-run their definition cells.
