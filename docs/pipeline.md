# Pipeline — stage by stage

This document explains every stage of `build_mask(img, p)`. Each stage is a pure
function of the image and the `Params` instance `p`. Notebook section numbers (§) are
referenced throughout.

```python
def build_mask(img, p):
    stages    = preprocess(img, p)          # §3
    candidate = candidate_mask(stages, p)   # §4
    cleaned   = morphology_clean(candidate, p)   # §5.1
    final, df = filter_components(cleaned, p)    # §5.2
    return dict(stages=stages, candidate=candidate, cleaned=cleaned,
                final=final, components=df)
```

---

## 0. Loading — `load_gray()` (§2)

Every entry point goes through `load_gray(path)`, which does **red pen-mark removal**
before returning grayscale:

1. **HSV hue gate** (`_red_mask_bgr`). The samples have red/orange reference annotations
   at the borders. Red wraps around the hue circle, so two bands are unioned —
   `H ∈ [0, 20]` and `H ∈ [160, 180]` — both requiring moderate saturation
   (`S ≥ 60`, `V ≥ 40`) to skip dark neutral pixels.
2. **Connected-component area filter.** Scratches have chromatic aberration that leaks a
   few isolated red-tinted pixels; only blobs `≥ RED_MIN_COMPONENT_AREA` (100 px) are
   kept as real pen-marks, so faint scratches are not erased.
3. **Inpaint, don't zero.** The mask is dilated to cover the anti-aliased halo, then
   `cv.inpaint(..., INPAINT_TELEA)` fills the hole with surrounding glass texture. Zeroing
   would create a hard artificial edge that the Frangi ridge filter would later detect as
   a false scratch.

Output: single-channel `uint8` grayscale. Working in one channel is justified because a
scratch is an **intensity** signature (a thin bright or dark line), not a colour one.

---

## 1. Preprocessing — `preprocess()` (§3)

Goal: **enhance** thin line-like structures and **suppress** everything else (sensor
noise, dust grains, slow illumination drift) *before* thresholding. Returns a dict of
every intermediate so the debug overlays can show them. The ordering matters: median
runs **before** CLAHE so CLAHE does not amplify dust; `bgsub` / `top-hat` / `Frangi` all
operate on the CLAHE output.

| key | operator | purpose |
|---|---|---|
| `gray` | identity | input |
| `blur` | Gaussian | low-pass, denoise |
| `median` | median | remove isolated dust grains (edge-preserving) |
| `clahe` | CLAHE | local contrast boost for faint scratches |
| `bgsub` | background subtraction | remove slow illumination drift |
| `tophat` | morphological top-hat / black-hat | small-scale residual |
| `ridge` | **Frangi vesselness** | multi-scale thin-line detector |

### Gaussian blur — `gaussian_blur()`
Convolution with an isotropic Gaussian (low-pass):

$$
I_g(x,y) = \sum_{i,j} I(x+i,y+j)\,G_\sigma(i,j), \qquad
G_\sigma(i,j) = \frac{1}{2\pi\sigma^2}\exp\!\Big(-\tfrac{i^2+j^2}{2\sigma^2}\Big)
$$

`gaussian_ksize=10` (forced odd → 11). OpenCV derives σ from the kernel size.

### Median blur — `median_blur()`
Non-linear, robust to outliers — kills isolated dust grains without smearing edges:

$$
I_m(x,y) = \operatorname{median}\{\,I(x+i,y+j) : |i|,|j| \le k/2\,\}
$$

`median_ksize=5`. A median has a 50 % breakdown point, so a 5×5 window still has > half
its pixels along a 1–2 px scratch running through it — the scratch survives while dust
grains (which fill < half the window) are removed.

### CLAHE — `clahe()`
Contrast-Limited Adaptive Histogram Equalization: local histogram equalization per tile,
with bin-clipping to limit noise amplification.

$$
I_c(x,y) = \big\lfloor 255 \cdot \mathrm{CDF}_{\text{tile}}(I(x,y)) \big\rfloor
$$

`clahe_clip=3.0`, `clahe_tile=8` (8×8 grid). **Caveat:** on textured/mottled glass CLAHE
amplifies surface texture as well as scratches — one of the reasons global Otsu floods
(see [thresholding.md](thresholding.md)). The Frangi path is far less sensitive to this.

### Background subtraction — `background_subtract()`
A large median filter estimates the slowly-varying glass background; subtract it
(polarity-aware):

$$
B = \operatorname{median}_{k\times k}(I), \qquad
D = \begin{cases} \max(0,\,I-B) & \text{bright\_features=True} \\ \max(0,\,B-I) & \text{else} \end{cases}
$$

`bg_subtract_ksize=51` — much larger than scratch width, so the thin scratch is an
outlier of the median and survives as a positive residual. `cv.subtract` saturates
(clips negatives to 0), giving the `max(0, ·)` for free.

### Top-hat / Black-hat — `tophat()`
Polarity-aware morphological residual. With structuring element `S`:

$$
\gamma_S(I) = (I \ominus S)\oplus S, \qquad \phi_S(I) = (I \oplus S)\ominus S
$$
$$
\text{white top-hat} = I - \gamma_S(I) \;\;(\text{bright features}), \qquad
\text{black-hat} = \phi_S(I) - I \;\;(\text{dark features})
$$

`tophat_ksize=15`. Both flavours return a near-zero background with small-scale features
lifted to positive intensities (bright-on-dark by construction).

### Frangi ridge filter — `ridge_filter()` ⭐
Multi-scale Hessian-based line detector — **the most important operator for this
dataset.** At each scale σ, build the Hessian of the Gaussian-smoothed image and take its
eigenvalues $|\lambda_1| \le |\lambda_2|$. A ridge-like (line) pixel has
$|\lambda_1| \approx 0$ and $|\lambda_2| \gg 0$. Define

$$
R_B = \frac{|\lambda_1|}{|\lambda_2|}, \quad S = \sqrt{\lambda_1^2 + \lambda_2^2}, \quad
V_\sigma = \exp\!\Big(-\tfrac{R_B^2}{2\beta^2}\Big)\Big(1 - \exp\!\big(-\tfrac{S^2}{2\gamma^2}\big)\Big)
$$

Final response $V(x,y) = \max_\sigma V_\sigma(x,y) \in [0,1]$.

- `frangi_sigmas=(1,2,3,4)` — each scale ≈ half-width of a target ridge, so 1–4 px wide
  scratches all score high.
- `frangi_alpha=0.5`, `frangi_beta=0.5`, `frangi_gamma=5.0`.
- Polarity via `black_ridges = not bright_features`.

**Why it matters:** because $V$ is built from *ratios* of eigenvalues, it is largely
invariant to absolute brightness — it responds to the *shape* of a thin ridge, not to how
bright the region is. That is exactly what makes the ridge-based thresholds robust to the
uneven illumination that breaks Otsu.

---

## 2. Thresholding — `candidate_mask()` (§4)

Converts a preprocessed image into a binary candidate mask
$M(x,y) = 255\cdot\mathbf 1[I(x,y) > T(x,y)]$. Eight methods are available; the dispatch
reads `p.threshold_method`. **Default: `ridge_hysteresis`.**

This is covered in full in **[thresholding.md](thresholding.md)** — including why the
default was changed away from `otsu`.

---

## 3. Morphology clean-up — `morphology_clean()` (§5.1)

Binary opening then closing:

$$
\gamma_S(M) = (M \ominus S)\oplus S \;\text{(opening — removes speckles)}, \qquad
\phi_S(M) = (M \oplus S)\ominus S \;\text{(closing — bridges intra-scratch gaps)}
$$

Order is **open first** (kill isolated dust dots), **then close** (reconnect a scratch
broken into fragments). `open_ksize=3`, `close_ksize=3`. Setting `open_ksize=0` skips the
opening entirely — recommended when chasing very faint 1-px scratches that an erosion
would destroy; the component filter then carries the full dust-removal burden.

---

## 4. Connected-component filter — `filter_components()` (§5.2)

The candidate mask still contains dust and noise blobs. This stage labels 8-connected
components and keeps only those whose **geometry** is scratch-like. It returns the
filtered mask **and** a per-component `DataFrame` (all metrics, `kept` flag) for tuning.

For each component (bbox `w×h`, area `A`, largest contour `C`):

| metric | formula | scratch behaviour |
|---|---|---|
| length | $L = \max(w,h)$ | large |
| aspect | $\mathrm{AR} = \max(w,h)/\max(\min(w,h),1)$ | large (bbox proxy) |
| elongation | $e = a_\text{major}/a_\text{minor}$ of fitted ellipse | large (ellipse proxy) |
| solidity | $s = A/|\mathrm{ConvexHull}(C)|$ | small (thin curve) |
| **slenderness** | $\text{slen} = \text{skel\_len}^2 / A = \text{skel\_len}/\text{width}$ | **large** |

where `skel_len` is the pixel count of the 1-px skeleton (`skimage.skeletonize`) and
`width = A / skel_len` is the mean stroke width.

### The decision rule

```python
shape_ok = (aspect >= min_aspect_ratio) OR (elong >= min_elongation)
keep =  (min_area <= A <= max_area)
    AND (L >= min_length)
    AND shape_ok
    AND (solidity <= max_solidity)
    AND (slenderness >= min_slenderness)      # skeleton dust gate
```

Two design choices worth understanding:

1. **`aspect` OR `elongation` (not AND).** They are two proxies for "how line-like."
   When scratches cross or are bordered by dust, the bbox widens and `aspect` collapses,
   but the fitted ellipse is still elongated — and vice versa for short straight scratches
   where the ellipse fit is noisy but the bbox aspect is clean. Either passing is enough.

2. **`slenderness` as a hard AND — the skeleton dust gate.** Both `aspect` and
   `elongation` are bbox/ellipse proxies that a compact dust grain can *fake* — a noisy
   ellipse fit on a near-circular blob reports a spurious elongation above
   `min_elongation`, and the grain sneaks through the OR. The skeleton sees the actual
   centreline: a grain's collapses to a short stub (slenderness < ~1), while a real
   scratch's stays long relative to its width.

   Measured on the development image `img__0`: kept scratches had **median slenderness
   5.3** (5th-percentile 1.2), while rejected dust (area ≥ 30) had **median 0.4**. A floor
   of `min_slenderness=1.5` pruned the 8 chunkiest blobs (width 8–17 px, skeleton 9–19 px)
   that had slipped through on ellipse noise, while keeping 84 of the 92 genuine scratch
   components.

   > **Note on `width` alone:** on chunky masks a real scratch can be several px wide, so
   > `width` does *not* separate it from a dust grain (both are ~4–17 px). It is
   > *slenderness* (length relative to width) that discriminates — see the development
   > notes in the `_skeleton_metrics` docstring.

See [parameters.md](parameters.md) for how to tune each threshold.

---

## 5. Saving — `save_mask()` (§6) and batch drivers

`save_mask(mask, src_path, out_dir)` writes a strict `{0,255}` `uint8` PNG with the
**same filename and shape** as the source — the contract the U-Net dataset relies on.

- **`process_batch`** (§9) — single flat folder (`RAW_DIR` → `MASKS_DIR`), also writes a
  4-panel debug strip.
- **`process_split`** (§12) — walks the nested `dataset/{train,val}/images` trees and
  saves each mask to the **mirrored** path under `.../masks/`, preserving the
  `glass/img__*.png` structure so image↔mask pairing is exact.
