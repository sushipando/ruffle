# Performance changes to Ruffle

This branch (`perf`, based on upstream `2372a4e1`) makes Ruffle's desktop player much faster
for content that draws many objects with blend modes, cached bitmaps and filters, and
keeps memory and logic time from growing in content that keeps loading new movies.

Target setup used for all measurements: Apple M1, macOS, fullscreen on a
5120x1440 display, default (high, 4x MSAA) quality. The test content is a large AS3
movie whose heavy scenes have hundreds of blend layers, around a hundred filtered
cached bitmaps and thousands of shapes per frame.

| Scene | Upstream | This branch |
|---|---|---|
| Light scene | 24 fps | 24 fps |
| Medium scene | **0.5 fps** (2 s per render) | **24 fps** |
| Heaviest medium scene seen (~250 complex blends/frame) | not measured (<1 fps) | 15-19 fps before async submit (7); not re-measured since |
| Heavy scenes (~1400 complex blends/frame) | not measured | scene A: 3.4 → ~10 fps (12); scene B: 17 → ~21 fps (13) |

Later measurements on the laptop's own display (2880x1800 fullscreen, stage
magnified 3x), with everything up to 21:

| Scene | Result |
|---|---|
| Medium scenes | 24 fps |
| Heavy scenes, most of them | 19-24 fps |
| Heavy scene A | ~15 fps (GPU-bound) |
| Heavy scene B, with large blurs | 11.5 → 16.6 fps with 21 (2x only) |
| Heavy scene C, with large blurs | 17.1 → 22.6-23.9 fps with 21 (incl. 4x) |
| Logic time after replacing most of the content 7 times | was 4 → 23 ms; now stays 2-5 ms (15-18) |

From 24 on, the test content ran at 60 fps, so the target is 60. Laptop display,
heavy scenes, shown fps:

| Scene | Before 24 | Up to 31 | Up to 36 |
|---|---|---|---|
| Heavy scene, typical | ~30 | 57-60 | 60 |
| Heavy scene, ~110 filtered cached bitmaps | ~15-19 | ~32-47 | 60 (34) |
| Heavy scene, ~730 single-shape blend layers | not measured | ~38-48 | 60 (35) |
| Alternating 8 times between a heavy and a medium scene | not measured | heavy 21-41, medium 60 | 59.7-60 everywhere |

In-session comparisons for each change are in its section (24-36). Scenes differ
between sessions, so the columns compare similar scenes, not the same ones.

## Running

```sh
cargo build --release -p ruffle_desktop
./target/release/ruffle_desktop --tcp-connections allow --fullscreen \
    <movie URL>
```

Movies that use sockets need `--tcp-connections allow`.

## Why it was slow

Profiling showed the main thread spent ~80% of its time blocked inside
`CommandEncoder::finish`, waiting on Metal's command buffer semaphore. The
ActionScript side was cheap (~5 ms/frame). The cost was in the renderer:

* The scenes are built from many small parts, and ~130-250 per frame use
  complex blend modes (Overlay, Multiply), ~50-100 use Add/Screen, and ~70 are
  cached as bitmaps with Blur/Glow filters that change every frame (animations).
* Ruffle rendered **every blend into a full-viewport, multisampled offscreen
  texture** (5120x1440 x4 samples here): ~295 such targets per frame.
* Every complex blend split the main render pass. With MSAA, every pass ended
  with a **resolve of the whole target**, which was ~90% of the GPU time.
* wgpu (on Metal) creates up to 3 command buffers per render pass, and its
  per-pass bookkeeping scales with the total number of resources on the device.
  Hundreds of passes per frame made the per-pass CPU overhead dominant.

The fixes below go roughly in the order they were found. All of them except the
wgpu patches and the mouse-picking fixes can be toggled at runtime (see
[Tooling](#tooling)), which is how the A/B numbers were measured: same session,
same scene, flag on vs off.

## Optimizations

### 1. Bounded blend layers

*Files: `core/src/display_object.rs` (`blend_layer_bounds`), `render/src/commands.rs`
(`Command::BlendBounded`, `CommandHandler::blend_bounded`),
`render/wgpu/src/surface/commands.rs` (`bounded_blend_rect`, `blend_impl`),
`render/wgpu/src/surface.rs` (`Surface::with_origin`), `render/wgpu/src/surface/target.rs`.*

Core now computes conservative viewport-space bounds of a blended object
(including filter growth) and emits `BlendBounded` instead of `Blend`. The wgpu
backend renders the layer into a texture covering only those bounds (+2 px
padding, rounded up to multiples of 32 px so textures can be reused). The
surface's origin is applied as a translation to its draws. Complex blends
composite into just that region (viewport) using a backdrop copy of just that
region.

Bounds are not used (falls back to the full-size layer) when they can't be
trusted: `scrollRect` anywhere in the subtree, Alpha/Erase/Shader blends anywhere
in the subtree, or when the bounds cover most of the target anyway. Other backends
ignore the bounds (default trait method), so they behave as before.

Effect: logic 0.5 → ~14 fps. Shown frames were still GPU-bound at ~3/s.
Same-session A/B: GPU time per frame 1029 ms (off) vs 365 ms (on).

Flag: `bounded-blends`.

### 2. Lazy MSAA resolve

*Files: `render/wgpu/src/surface/target.rs` (`color_attachments_for_pass`,
`ensure_resolved`, `copy_region_for_blend`), `render/wgpu/src/msaa_resolve.rs`,
`render/wgpu/shaders/msaa_region_resolve.wgsl`.*

Experiments in one session showed GPU time per frame was 365 ms at high quality,
199 ms at medium and 18 ms at low. Skipping the per-pass resolve alone brought high
quality down to 27 ms. Multisampled targets now:

* draw passes without a resolve target, and mark the target as needing a resolve;
* resolve on the last pass of the target (folded into that pass), or with an
  empty resolve pass in `ensure_resolved` when a finished target is read;
* for complex blends, resolve only the needed region with a small shader that
  averages the samples (`MsaaRegionResolve`, a box filter like the hardware
  resolve). The source offset is packed into the instance index, so no buffer is
  needed.

Effect (same session): GPU 430 → 32 ms/frame, shown fps 2.2 → 14.5.

Flag: `lazy-resolve`.

### 3. Merging complex-blend composites into the next pass

*File: `render/wgpu/src/surface.rs` (`PendingBlend`, `draw_pending_blend`).*

A complex blend's composite (blend shader quad) used to get its own render pass,
followed by another pass for the draws after it. It's now drawn at the start of
the next draw pass on the same target (with the right stencil configuration), and
the last pass on a target also resolves it. Effect: ~22 fps shown at ~440 passes,
GPU ~18 ms/frame.

### 4. Layer atlas (blend layers and cached bitmaps)

*Files: `render/wgpu/src/surface/layer_atlas.rs`, `render/wgpu/src/backend.rs`
(cache entry loop), `render/wgpu/src/surface/commands.rs` (`chunk_leaf_layers`,
`DrawCommand::SetScissor`).*

Blend layers and cacheAsBitmap contents that only contain plain draws (no masks,
nested layers or reads of other cache textures in the same frame) don't depend on
anything else. They're shelf-packed into one atlas texture, drawn in a single
pass (each layer translated to its slot and scissored to it), and copied out into
their own textures. Cache entries are only atlased when their clear color is fully
transparent.

Effect (same session): blend layers 713 → 518 passes, render CPU 68 → 52 ms.
Adding cache entries: 657 → ~415 passes, render CPU 50 → 40 ms, shown 15.2 → 18.6 fps.

Flag: `layer-atlas`.

### 5. Filter results copied back with a blit

*File: `render/wgpu/src/backend.rs`.*

After applying filters to a cached bitmap, the result was copied back into the
cache texture with a render pipeline (one pass). When the size, format and sample
count match (always, in practice), it's now a `copy_texture_to_texture`. This
gives identical bytes and saves ~65 passes per frame in a medium scene.

### 6. Wave scheduling of complex blends

*Files: `render/wgpu/src/surface/wave_schedule.rs`, `render/wgpu/src/mesh.rs`
(mesh bounds), `render/wgpu/src/surface.rs` (batched backdrops),
`render/wgpu/src/surface/target.rs` (`copy_regions_for_blends`).*

Complex blends on different objects don't affect each other, but each one
still broke the main pass. Before chunking, a target's commands are reordered
into "waves":

* every command gets conservative bounds on a 32 px grid (meshes store exact
  vertex bounds; masked groups `PushMask..PopMask` are treated as one unit,
  since they restore the stencil buffer; anything unboundable is a barrier);
* each unit gets the earliest wave that keeps it after everything it overlaps
  (draws may share their dependency's wave, anything after a complex blend goes
  to the next wave);
* waves are output in order: draws (in original order), then that wave's complex
  blends.

Only non-overlapping commands change relative order, so the output is identical.
Consecutive non-overlapping complex blends then share one backdrop: all regions
are resolved in a single pass into an atlas (or copied, without MSAA), and their
composites are drawn at the start of the next pass.

Effect: 130 complex blends went from ~92 to 44 waves, with 122 of them batched;
passes 402 → 307 (same session). It costs ~0.5 ms per frame. Masks had initially
been treated as barriers, which serialized everything (fixed). A finer grid
(16 px) was tried and made no difference, so the remaining dependencies are real
overlaps within a single object.

Flag: `wave-schedule`.

### 7. Asynchronous frame submission

*Files: `render/wgpu/src/backend.rs` (`ActiveFrame::submit_for_target_async`,
`AsyncSubmitter`), `render/wgpu/src/target.rs` (`can_submit_asynchronously`),
`desktop/src/gui/movie.rs`.*

`CommandEncoder::finish` (where wgpu encodes the actual Metal commands) plus
`Queue::submit` was ~60% of render time. For targets whose `submit` is a plain
`queue.submit` (only the desktop `MovieView`), the finished frame's encoder is
handed to a dedicated thread, which finishes and submits it while the main thread
runs the next frame's scripts.

Ordering rules: before anything that must come after the previous frame on the
queue (the next frame's staging belt/arena flush, `submit_direct`, mid-frame
flushes, pixel readbacks), `wait_for_submission` waits for it. The staging belt is
recalled only after the submission. `submit_frame` waits first thing, since it
finishes the staging belt itself.

Effect (same session): main-thread render time 12.7 → 3.6 ms/frame.

Trade-offs:
* One extra frame of display latency, since the GUI composites the previous frame
  (no longer with 24).
* Texture writes made by scripts before the in-flight frame is submitted can
  apply to that frame (one frame early).
* `Context3D` submits directly and isn't synchronized with this (the test content
  doesn't use Stage3D).

Flag: `async-submit`.

### 8. Per-frame transforms arena

*Files: `render/wgpu/src/dynamic_transforms.rs`, `render/wgpu/src/layouts.rs`
(dynamic offset), `render/wgpu/src/buffer_builder.rs`.*

Every chunk copied its transforms and vertices into the same small buffer, right
before its pass. That's a staging copy per chunk, and on Metal a command buffer
per chunk. Chunks now get their own regions of 8 MiB / 4 MiB arenas, bound with a
dynamic offset. Everything is uploaded with one `Queue::write_buffer` before each
submission, which is ordered after earlier submissions, so the arena can be
reused every frame. If an arena fills up, chunks fall back to the old per-chunk
copy into a reserved region. `DynamicTransforms` moved into `ActiveFrame`, so
every submission path flushes it.

Effect: `copy_buffer_to_buffer` 471 → 106 samples per 10 s profile.

### 9. Rendering only the visible (letterboxed) region

*Files: `core/src/display_object/stage.rs` (`visible_region`), `core/src/player.rs`,
`render/src/backend.rs` (`RenderBackend::set_visible_region`),
`render/wgpu/src/backend.rs`, `render/wgpu/src/utils.rs`
(`run_copy_pipeline_into_region`).*

When the stage is letterboxed (ShowAll and letterbox enabled, as in fullscreen),
nothing outside the stage rectangle is visible. The main surface is then only the
stage rectangle (2514x1440 instead of 5120x1440 here), copied into the frame with
the rest cleared to black.

Effect (same session): GPU 24.5 → 20.6 ms/frame.

Flag: `visible-region`.

### 10. Texture pool eviction and size bucketing (memory leak fix)

*Files: `render/wgpu/src/buffer_pool.rs`, `render/wgpu/src/globals.rs`,
`render/wgpu/src/surface/commands.rs`.*

`TexturePool` never evicted anything, and bounded layers made texture sizes (and
`Globals`, which included the origin) different every frame. A build reached
**79 GB** before macOS killed it. Now:

* pools and globals unused for 4 frames are dropped (`TexturePool::end_frame`);
* `Globals` no longer depend on the origin (it's applied as a draw translation);
* layer and atlas sizes are rounded up (32 px / 256 px) so textures get reused.

The offscreen pool is no longer thrown away every frame. Memory is now stable at
~700 MB.

### 11. Mouse picking

*Files: `core/src/display_object.rs` (`global_matrix_cache`), `core/src/player.rs`
(`run_mouse_pick`), `core/src/display_object/movie_clip.rs`.*

Mouse picking (run twice per frame) was ~20% of the main thread:

* every visited object recomputed its global matrix by walking up to the root.
  During a pick (no scripts run, so the display list can't change), global
  matrices are now memoized per object and built from the parent's cached one.
  Flag: `pick-cache`.
* `MovieClip::mouse_pick_avm2` computed `world_bounds` (a walk over the clip's
  whole subtree) before checking whether the clip has a drawing at all; the
  checks were reordered. This is a pure reordering of side-effect-free checks.

Effect: picking 829 → 579 → 245 samples per 10 s profile.

### 12. Multiply as a fixed-function blend on opaque targets

*Files: `render/wgpu/src/blend.rs` (`TrivialBlend::MultiplyOpaque`),
`render/wgpu/src/surface/commands.rs` (`opaque`), `render/wgpu/src/surface.rs`,
`render/wgpu/src/surface/wave_schedule.rs`.*

On a fully opaque target (the main stage, cleared to the opaque stage background),
Flash's multiply `src*(1-dst.a) + dst*(1-src.a) + src*dst` reduces to
`src*dst + dst*(1-src.a)`. That's the blend state (src factor `Dst`, dst factor
`OneMinusSrcAlpha`), so such layers are composited like Add layers: no backdrop
copy, no pass break, and not part of any wave. Every other operation keeps an
opaque target opaque; a shader blend clears the flag.

Effect (same session, heavy scene A): complex blends 1380 → ~350 per
frame, passes 1377 → ~900, shown fps 3.4 → ~9.5. Flag: `opaque-multiply`.

### 13. Batched filters

*Files: `render/wgpu/src/surface/layer_atlas.rs` (`apply_filters_batched`,
`is_local_filter`), `render/wgpu/src/backend.rs`.*

Cached bitmaps up to 512x512 with identical "local" filters (Blur, Glow,
DropShadow, ColorMatrix) are filtered together. They're copied into an atlas with
transparent gutters wider than the filters' reach (from `calculate_dest_rect`),
filtered once, and copied back. Entries whose texture is drawn by another cache
entry are excluded, because their filtered result must exist first. Textures
that don't fit are filtered individually.

Effect (same session, heavy scene B): ~62 filters batched per frame,
passes 538 → 428, shown fps 17.0 → 20.5. Flag: `filter-batching`.

### 14. Asynchronous mid-frame flushes and per-submission staging belts

The periodic flush every 100 cache entries also goes to the submit thread when
one exists (~20% of main-thread time in a heavy scene). Each asynchronous submission
takes its own `StagingBelt`, which is recalled only after that submission;
recording continues on another belt. An earlier version recalled the shared belt
while it held chunks of not-yet-submitted work (`submit_frame` and the pixel
bender finish the belt mid-frame), which made wgpu fail with
"StagingBelt staging buffer is still mapped".

### 19. Bounded `Layer` blends around `Alpha`/`Erase` children

*File: `core/src/display_object.rs` (`blend_layer_bounds`).*

A common Flash masking pattern is a `Layer` parent with an `Alpha` (or `Erase`) child.
`blend_layer_bounds` gave up on bounding when any descendant had one of those blend
modes. So such layers covered the whole target, and so did their `Alpha` children,
whose target is the layer. One heavy-scene session had 12 of each at full screen every
frame (`Layer/full`, `Alpha/full` in `[gpu-named]`), at 91% GPU and ~10 fps.

The alpha and erase shaders discard where the child is transparent and otherwise
only scale the layer's alpha down, so they never make a layer cover more pixels.
For `Layer` blends, such descendants are now left out of the bounds. Other blend
modes still use the whole target. `visual/blend_modes/layer_alpha` and
`layer_erase` take the new path, and still match their reference images. The
effect on the test content is unmeasured: the object went away before it could be
compared.

### 20. Render passes share Metal command buffers

*Files: wgpu-core fork (`src/command/render.rs`, `can_share_command_buffer`;
`src/ruffle_patch.rs`), `render/wgpu/src/backend.rs`.*

Ruffle records a frame into one `CommandEncoder`, but wgpu-core gives every render
pass its own raw command buffer. A pass's barriers and fixups are only known after
recording it, so they go into another raw buffer that is submitted before the pass.
On Metal, that came to ~430 `MTLCommandBuffer`s per frame in a heavy scene (the empty barrier
buffers are already skipped, see the wgpu-hal patches).

When a pass can't need anything but barriers before it, wgpu-core now records it
into the current raw command buffer, and records its barriers (no-ops on Metal)
after it. That's the case when:
* nothing was discarded in the command buffer so far, and the pass discards nothing
* the pass has no queries
* it has no indirect draws or bundles

Other backends and other passes are unchanged. Ruffle always qualifies.

Effect (heavy scene A, same session): command buffers 11,000/s → 70/s,
main-thread render CPU 13 → 9 ms/frame. Shown fps was unchanged (14.4 vs 14.8,
within scene drift), because that scene is GPU-bound. Flag: `share-cmdbufs`. It
switches `wgpu_core::ruffle_patch::set_share_command_buffers`, so `render/wgpu`
now depends on `wgpu-core` directly (desktop only).

### 21. Large blurs at lower resolution when the stage is magnified

*Files: `render/wgpu/src/filters/blur.rs` (`apply_downscaled`, `downscale_factor`),
`render/src/filters.rs` (`set_view_scale`), `core/src/player.rs`,
`desktop/src/main.rs`.*

Profiling GPU time per pass type (`RUFFLE_GPU_LABELS`) in heavy scene B (~11 fps)
showed the blur filter taking ~60% of GPU time. Glow, drop shadow
and bevel all blur, too. Filter sizes are multiplied by the stage magnification
(`Filter::scale`). So on a 2880-wide display (3x the movie's stage), a blur costs
9x the pixels times 3x the taps per pixel.

Now, when the stage is magnified by at least 2 and both blur sizes are at least 8
device pixels, the blur's source is scaled down 2x, blurred with half the sizes,
and scaled back up:
* Blurs of at least 64 device pixels are scaled down 4x instead (in two 2x steps),
  so they're still at least 16 pixels wide. They have no detail to lose. A 2x factor
  never goes below the movie's own resolution.
* Scaling down and up uses the blur pipeline with a one-tap kernel and bilinear
  sampling. Sampling at the corners of 2x2 blocks averages them.
* Blur work drops ~8x (2x) or ~64x (4x).
* Core passes the magnification to the renderer through a thread-local
  (`ruffle_render::filters::set_view_scale`) while it submits the frame. Filters
  applied by scripts (`BitmapData.applyFilter`) see 1.0 and stay exact.

The result is visually identical, but not bit-exact:
`visual/filters/blur_scales_with_screen` differs by up to 15/255 on 31,804 pixels
when it's on. So the flag defaults to off in the library (and the tests), and the
desktop player turns it on at startup.

Effect (heavy scene B, same session, alternating, 2x only): blur work
~1450M → ~200M texture taps per frame, GPU 86 → 59 ms/frame, shown fps 11.5 → 16.6.
With 4x for very large blurs (heavy scene C): ~925M → 93-317M taps,
GPU 58.5 → 39-43 ms/frame, shown fps 17.1 → 22.6-23.9. Full-resolution crops of
glowing objects look the same with it on and off.
Flag: `blur-downscale`. `[gpu-named]` shows blur statistics with `RUFFLE_PERF`:
`blur-kpx` (pixels blurred per frame), `blur-mtaps` (texture taps, also by size),
and blur counts by size and quality.

### 24. The GUI presents on the submit thread

*Files: `desktop/src/gui/controller.rs` (`PendingPresents`, `render`),
`render/wgpu/src/backend.rs` (`SubmitJob`, `run_after_submissions`).*

At 60 fps, a Metal System Trace (Instruments, see Tooling) of a heavy scene showed
the GPU idle for about half of each frame, mostly in two gaps: ~11 ms before each frame's
first pass and ~5 ms before the GUI pass. The main thread was blocked in `Queue::present`
~30% of the time: `present` takes wgpu-core's snatch lock for writing, and the submit
thread holds it for reading while it finishes the previous frame's encoder (7). So the
main thread, the submit thread and the GPU took turns.

Now the GUI's frame (the movie texture plus egui) is submitted and presented by the
submit thread right after the movie's frame. Before acquiring the next surface texture
or reconfiguring the surface, the GUI waits for that present. This also removes the
extra frame of latency from 7: the GUI now shows the frame that was just rendered,
not the previous one. And `submit_frame` no longer waits for the previous submission
before recording (it waits before submitting), since recording uses another staging
belt and only buffers transforms on the CPU (8).

Effect: in the traces (debug-label builds, different scenes), the GPU's idle time went
from 47% to 26%. Shown fps was the
same in a heavy scene (~30 either way), because the GPU is the limit there once the binary
isn't slowed down by debug labels. It helps when the CPU is the limit. Flags:
`async-present`, `overlap-record`.

### 25. All passes of a blur in each direction at once

*Files: `render/wgpu/src/filters/blur.rs` (`merged_kernel`, `apply_merged`),
`render/wgpu/shaders/filter/blur.wgsl` (`kernel_fragment`), `render/src/filters.rs`
(`rendering_for_display`).*

A Flash blur of quality q is q box blurs in each direction, and Ruffle did each one as
a render pass, plus scaling down and up (21): 8 passes for a typical glow. In one
heavy-scene frame, 194 blur passes took 11 ms of GPU time for only ~100M texture samples.
Most of that is per-pass cost: setting up the pass, and the GPU idling 17-37 µs while
each pass waits for the one before it.

The q box blurs in one direction are now combined into one kernel on the CPU (their
convolution, at most 256 bilinear taps), and applied in one pass per direction. That's
the same number of texture samples, but 2 passes instead of 2q.

The result isn't bit-exact: it's rounded to 8 bits once per direction instead of once
per pass (up to 5/255 different in `visual/filters/blur_pass_scaling`), and the first
pass reads further outside of the source, which matters for bitmaps that are opaque
up to their edges. So it's only used while a frame is rendered for display, never for
`BitmapData.applyFilter` (`avm2/bitmapdata_applyfilter_blur` differed by up to 50/255
otherwise), and the flag is off in the library (and tests) and on in the desktop player.

Effect (heavy scene A, alternating): 14.8 → 17.1 fps, GPU 67.7 → 58.5
ms/frame, 594 → 506 passes. Flag: `blur-merge`.

### 26. Blurs scaled down separately in each direction, up to 8x

*File: `render/wgpu/src/filters/blur.rs` (`downscale_factors`, `apply_downscaled`).*

`RUFFLE_BLUR_DEBUG` (below) showed that most of the blur work in a heavy-scene frame
(389M of 445M texture samples) came from a few blurs like `blurX = 0, blurY = 80` at
quality 3. On a 3x magnified stage that's -2 by 238 device pixels (`scale_blur` maps 0
to -2, which is still no blur). 21 chose the scale from the smaller of the two sizes,
so these ran at full resolution: 300-550M samples each.

The factor is now chosen for each direction: 1 below 8 pixels (and without blur), 2 from
8, 4 from 64 and 8 from 128, so a blur is still at least 16 pixels wide after scaling
down by 4 or 8. Scaling down one direction at a time works like 2x2 blocks: sampling
between two texels averages them.

Effect (heavy scene B): blur work 445M → ~27M texture samples per frame.
Blur downscaling on vs off in the same session: 11.7 → 19.3 fps (the previous session
with the old rule was at 13.6 fps).

### 27. Complex blends read the backdrop from the color attachment

*Files: `render/wgpu/shaders/blend/fetch.metal`, `render/wgpu/src/shaders.rs`
(`fetch_blend_shaders`, `supports_fetch_blends`), `render/wgpu/src/pipelines.rs`
(`fetch_blends`), `render/wgpu/src/surface/commands.rs` (`DrawCommand::FetchBlend`),
`render/wgpu/src/backend.rs` (`request_device`).*

A complex blend (Overlay, Multiply onto a transparent target, HardLight, ...) needs the
pixels below it. So far that meant ending the target's render pass, resolving and copying
the region below the blend, and drawing the composite with a shader that reads the copy
(1-3, 6). Wave scheduling (6) batched them, but heavy scenes still had dozens of waves.
`complex-as-normal` (Tooling) showed that this cost ~19 ms of a 74 ms frame in heavy scene B.

Apple GPUs can read the current color attachment in a fragment shader (programmable
blending, `[[color(0)]]` in Metal). With an MSAA target, the shader then runs for each
sample. WGSL can't express that, but wgpu passes Metal source through
(`PASSTHROUGH_SHADERS`, requested only on Metal with an Apple GPU). The blend's
fragment shader is written in Metal (`fetch.metal`, with each mode's formula from the
existing WGSL blend shaders). Its vertex stage is the bitmap shader's (WGSL), so it
uses naga's names for that shader's outputs (`user(loc0)`), and wgpu-hal's slots for the
bitmap pipelines' bindings (the texture and sampler are the fragment stage's first).

Complex blends other than Alpha and Erase (whose backdrop is their layer, which may be
another target) are now drawn like trivial ones: a quad with the layer texture, in the
current pass. No pass break, resolve, copy or wave scheduling (which is skipped then).
The results are the same, except that the blend is now computed for each sample at
anti-aliased edges, instead of once from the averaged backdrop. All `visual/blend_modes`
tests pass with it (11 fail when the Metal shader is replaced with a solid color, so
they do use it).

Effect (heavy scene D, alternating): 33.9 → 46.5 fps, GPU 29.5 → 21.4
ms/frame, 203 → 127 passes. Flag: `fetch-blends`.

### 28. Chunks of draws share render passes

*Files: `render/wgpu/src/surface.rs` (`draw_commands`), `render/wgpu/src/surface/layer_atlas.rs`,
`render/wgpu/src/dynamic_transforms.rs` (`ChunkOffsets::fallback`).*

Draws are recorded in chunks of up to ~200 transforms (the size of one uniform
binding), and each chunk was drawn in its own render pass. With complex blends gone
from the main pass (27), that was most of the remaining passes on the multisampled
main target: ~20 per frame in heavy scene B, at ~0.45 ms of GPU time each. The layer atlas
(4) was split the same way.

Since 8, every chunk's transforms and vertices have their own region of an arena,
bound with a dynamic offset. So consecutive chunks (with the same stencil use) are
now drawn in one pass, rebinding the transforms at each chunk's offset. Only a chunk
in the arena's fallback region (used when the arena is full) ends the pass, since the
next such chunk would overwrite it.

Effect (heavy scene B, alternating): 41.7 → 57.5 fps, GPU 24.0 → 16.5
ms/frame. Flag: `merge-chunks`.

### 29. Cached bitmaps keep slightly larger textures

*Files: `core/src/display_object.rs` (`BitmapCache::update`, `BitmapCache::bitmap`).*

A CPU profile of `Player::render` in a heavy scene showed ~15% of it creating textures (Metal
resource setup and kernel calls), and ~40% of those textures came from
`create_empty_texture` in `render_base`. A `cacheAsBitmap` object (or one with filters)
got a new texture whenever its pixel size changed, which animated objects do every
frame. The rest came from the renderer's texture pool, whose entries are keyed by exact
size: filter passes and multisampled targets for those bitmaps had new sizes, too.

Now the cache's texture is rounded up to a multiple of 32 pixels, and kept while it's
at most 64 pixels too large in each direction. Only the used part is drawn (`bitmap()`
returns the used size, and draws use that region). The extra space is transparent, which
doesn't change the drawing (cached bitmaps are drawn pixel-snapped at scale 1), or what
blurs, glows, drop shadows and color matrices produce in the used part. Exact sizes are
kept for an `opaqueBackground` (it fills the whole texture) and other filters (a
displacement map depends on the texture's size: 3 `visual/filters/displacement_map*`
tests failed without that).

Effect (heavy scene A, alternating): render CPU 9.4 → 6.7 ms/frame, main
thread 47% → 38% busy. Flag: `cache-slack`.

### 30. No AVM1 display list search when nothing is pending removal

*Files: `core/src/avm1/runtime.rs` (`remove_pending`), `core/src/display_object/container.rs`
(`queue_removal`).*

`Player::run_frame` runs the AVM1 frame after the AVM2 phases, even for AVM2 movies,
and that started with a search of the whole display list for AVM1 objects pending
removal. In a heavy scene that's tens of thousands of objects, 60 times a second: ~0.4 ms per
frame (from a profile). Objects are only ever marked pending removal by `queue_removal`,
which now also sets a flag, and the search only runs when it's set.

### 31. Larger cached bitmaps share filter atlases

*File: `render/wgpu/src/surface/layer_atlas.rs` (`apply_filters_batched`,
`max_batched_filter_size`).*

Filter batching (13) applied identical filters to cached bitmaps of up to 512 pixels,
in atlases of up to 2048. Heavy scenes often show the same symbol several times, with large
filtered parts (for example, 8 bitmaps with three glows each), which were filtered one by
one. The limits are now 1024 and 4096 (an atlas is still only as large as its contents,
rounded up to 256).

Effect (heavy scene D, 86 filtered bitmaps per frame, alternating): passes
113 → 89, GPU 16.7 → 16.0 ms/frame. Flag: `batch-large-filters`.

### 32. No profiler scope per draw command

*Files: `render/wgpu/src/utils.rs` (`PROFILE_COMMANDS`), `render/wgpu/src/surface.rs`,
`render/wgpu/src/surface/layer_atlas.rs`.*

Every draw command was executed inside a `wgpu_profiler` scope, which allocates its label
and sends a message over a channel even when timer queries and debug groups are off.
With thousands of draw commands per frame, that was ~5% of `Player::render`. Scopes per
command are now only used in builds with `render_debug_labels` or tracy.

### 33. Blend layer textures keep their bind groups

*Files: `render/wgpu/src/buffer_pool.rs` (`PooledTexture`),
`render/wgpu/src/surface/commands.rs` (`texture_bind_group`).*

Each blend layer is drawn with a bind group for its texture, and ~300 were created per
frame (~10% of `Player::render` was `create_bind_group`). Layer textures come from the
texture pool and are reused every frame, so pooled textures now keep that bind group.

Effect (heavy scene C, alternating): render CPU 5.68 → 5.04 ms/frame.
Flag: `cache-bind-groups`.

### 34. Filters of all cached bitmaps applied together

*Files: `render/wgpu/src/filters/batch.rs`, `render/wgpu/src/backend.rs` (`submit_frame`),
`render/wgpu/src/filters/glow.rs` (`linear_pipeline`).*

In heavy scenes, cached bitmaps had ~110 filters per frame, in 20-30 different
combinations (`RUFFLE_FILTER_DEBUG=1` lists them every 120 frames). 13 and 31 batch
identical filters only, so each combination still took 4-6 passes per filter (scale
down, blur horizontally and vertically, scale up, glow), and filters were ~10 ms of a 21
ms frame (`skip-cache-filters`).

Now all cached bitmaps whose filters are blurs, glows and color matrices (and that no
other cached bitmap draws) are placed in one atlas, with transparent space around each
as wide as its filters spread, at positions that are multiples of 8. Each step of their
filter lists is done for all of them together, with one quad per bitmap:
* Blurs (and glows' blurs) are grouped by how much they're scaled down (26). Each group
  takes a pass per 2x step, then a horizontal and a vertical pass with each bitmap's
  merged kernel (25), passed with a dynamic uniform offset.
* A final pass draws each bitmap's result for the step into a new atlas: blurs scaled up
  (bilinear), glows reading the scaled-down blur directly (with a filtering sampler,
  hence `linear_pipeline`), color matrices, and copies of bitmaps without a filter in
  this step.

So a step costs about four passes for all bitmaps together. It's only used with
`blur-merge` (desktop), and gives the same results as 25 (all `visual/filters` tests pass
with both on, except the two that differ by 5/255 with 25 alone; 14 fail if the results
aren't copied back, so they do use it). Drop shadows (their offset can reach into a
neighbor), bevels and other filters use the previous paths.

Effect (heavy scene C, ~110 filtered bitmaps per frame, alternating): 56.7
→ 60.1 fps, GPU 17.5 → 13.8 ms/frame, passes 130 → 50. Flag: `filter-together`.

### 35. Blends of a single fill are drawn without a layer

*Files: `render/src/tessellator.rs` (`Draw::single_fill`), `render/wgpu/src/mesh.rs`,
`render/wgpu/src/surface/commands.rs` (`direct_blend_mode`), `render/wgpu/src/pipelines.rs`
(`blended_fills`).*

Counting blend layers by their number of commands (the `/1cmd` names in `[gpu-named]`)
showed that in a heavy scene, almost all of them (~730 per frame) contain a single command:
Multiply (on the opaque stage, so a trivial blend, see 12) ~430, Overlay ~190, Add ~110.
Each was drawn into the layer atlas, copied out into its own texture, and composited:
the copies alone were ~5.7 ms of GPU time per frame.

Blending a layer draws each of its pixels once with the blend mode. If the layer is a
single bitmap, or a shape with a single fill and no strokes, whose triangles don't overlap
each other (the tessellator now marks such draws: one fill path, which lyon tessellates
into non-overlapping triangles; several solid fills are merged into one draw and may
overlap), drawing it directly with the blend mode's blend state does the same. So for
trivial blend modes (including Multiply on opaque targets), such blends are now drawn
directly, with blend-mode variants of the color and gradient pipelines (created on first
use) and of the bitmap pipelines. The only difference is at anti-aliased edges, where
each sample is blended instead of the averaged layer.

With fetch blends (27), solid fills with a complex blend mode (other than Alpha and Erase)
are drawn directly too: the color shader's vertex stage already outputs the final
color, so a Metal fragment shader (`fetch.metal` with `VERTEX_COLOR`) blends it with the
pixel below it, without any bindings.

Effect (heavy scene D, alternating): 47.8 → 60.1 fps, GPU 20.9 → 15.9 ms/frame
(before the complex modes; in a lighter scene later, ~225 blends per frame were drawn
directly). All tests pass (5 `visual/blend_modes` tests fail when the direct path ignores
the blend mode, 2 when the solid fill's color is replaced, so they use it). Flag:
`direct-blends`.

### 36. Cached bitmaps drawn and filtered in levels

*File: `render/wgpu/src/backend.rs` (`submit_frame`, `log_cache_filters`).*

Batched filters (13, 31, 34) are applied after all cache entries are drawn, so entries
whose texture another entry draws (nested `cacheAsBitmap`, common in the test content) were
excluded and filtered one by one: in a heavy scene at 21-35 fps, ~50 filters per frame
(`RUFFLE_FILTER_DEBUG` now says why an entry isn't batched).

Entries are now processed in levels: an entry's level is one more than the highest level
of the entries it draws. Each level's entries are drawn, then their filters are applied
together, before the next level's entries (which draw them) are drawn.

Effect: in heavy scenes, all filtered bitmaps are batched (~50 → 0 filters
applied one by one), and alternating 8 times between a heavy and a medium scene stayed at
59.7-60 fps everywhere (different scenes than the 21-35 fps run, though).

### 37. Shape definitions parsed on demand

*Files: `core/src/display_object/graphic.rs` (`ShapeRecords`),
`core/src/display_object/movie_clip.rs` (`define_shape`), `core/src/perf_stats.rs`.*

Every `DefineShape` kept its parsed records and styles (`swf::Shape`) for the life of
its movie, although they're only needed to tessellate the shape and to hit test it.
`RUFFLE_PERF` now prints their approximate size (`parsed shapes`): 77 MB right after
startup, and growing with all new content loaded. Most of them are never drawn
(other frames, hidden parts of symbols).

The movie's uncompressed data stays in memory anyway, so a shape now remembers where its
tag is in it, and drops its parsed definition right after it's defined. It's parsed
again when it's needed (no disk access):
* to tessellate it, the first time it's drawn and whenever it's drawn at a scale more
  than 2x different from its cached tessellations (for example, after the window is
  resized); it's dropped again right after;
* to hit test it, after which it's kept until it's tessellated again (the mouse tends
  to stay over the same objects).

Effect (4 content changes each): parsed shapes 77 → 100 MB before, 0-1 MB after (~1 MB
after hovering over objects), and RSS ~80 MB lower at the start. Resizing the window
from fullscreen (3x) to 500x340 points and back rendered correctly at 60 fps. With
re-parsing broken on purpose, 40 mouse and hit test tests fail, so they cover it.
Flag: `release-shapes`.

The rest of the growth per content change (~65 MB) is elsewhere: GPU meshes, `MovieClip`
preload data, and classes that the content keeps.

### 38. Meshes of shapes that aren't drawn are freed

*Files: `core/src/tessellation_cache.rs` (`clock`, `evict_unused_since`),
`core/src/player.rs` (`evict_unused_tessellations`), `render/wgpu/src/mesh.rs`
(`MESH_BYTES`).*

`RUFFLE_PERF` now prints the number of meshes and the size of their buffers
(`[gpu-mem]`). Over 6 content changes, they went from 43 MB to 208 MB (6,000 →
21,000 meshes), about half of the memory growth: a shape keeps its tessellations for as
long as its movie lives, and the test content keeps every movie it has loaded.

Each cached tessellation now remembers when it was last drawn (in seconds, updated
once per rendered frame), and every 5 seconds the player drops the tessellations of
library shapes that weren't drawn for 2 minutes (30 seconds at first; raised, since
tessellating them again when they come back can stall a frame): content that's no
longer shown, frames of
animations that aren't shown, and scales that were only used during a resize. A mesh is
freed once nothing else holds it (frames being rendered do). A shape that's drawn again
is parsed and tessellated again (37), like the first time.

Effect (6 content changes): meshes 59 → 98 MB (10,500 meshes) instead of 43 → 208
MB, RSS +238 MB instead of +310 MB (different scenes), still 60 fps. Flag: `evict-meshes`.

### 39. Vtables share their superclass's method tables

*File: `core/src/avm2/vtable.rs` (`MethodTable`).*

Like the traits in 22, every vtable copied its superclass's method table (indexed by
`disp_id`) and then replaced its overrides and appended its new methods. A heap profile
with allocation stacks after 4 content changes (`MallocStackLogging=lite` and
`malloc_history -callTree`) put 69.5 MB under `init_vtable`, the largest single entry:
the test content defines thousands of MovieClip subclasses, with ~200 inherited entries each.

A `MethodTable` now holds the superclass's flattened table in an `Rc`, a sorted list of
this class's overrides, and this class's new methods. A lookup checks the overrides (if
there are any) before the shared table. A class's flattened table is only made when it
gets a subclass, and a class that neither overrides nor adds methods just passes its
superclass's table on. `replace_scopes_with` (only used for `Object`, at startup) only
changes the class's own entries and clears its flattened table.

Effect (4 content changes): `init_vtable` 69.5 → 4.3 MB of live memory, same frame
rates, all tests pass.

### 40. Frames presented at an even pace

*Files: wgpu-hal fork (`src/metal/mod.rs`, `set_present_min_duration`), wgpu-core fork
(`src/ruffle_patch.rs`), `render/wgpu/src/backend.rs`,
`desktop/src/app.rs` (`pace_present`).*

Much Flash content runs at 24 fps. On a 144 Hz
display, a Metal System Trace of 24 fps content showed frames presented 33-50 ms
apart instead of every 41.7 ms: frames were presented as soon as they were drawn, and
drawing times vary, so each frame stayed on screen for 5, 6 or 7 refreshes, which makes
smooth motion (such as a rotating part) look uneven.

Frames are now presented with `presentDrawable:afterMinimumDuration:` (patched
wgpu-hal), which keeps each frame on screen for at least that long after the previous
one appeared. The desktop player sets it from the display's refresh rate when the rate
frames are drawn at divides it (for example 24 fps on 144 Hz: 6 refreshes; it uses
half a refresh less than that, so rounding never adds one). Otherwise frames are
presented as soon as possible.

Effect (24 fps, 144 Hz, from the trace's displayed-surface intervals): 107 of 108 frames
on screen for exactly 41.7 ms (6 refreshes), before 33-50 ms. Flag: `paced-present`.

### 41. Frames drawn between movie frames (`--interpolate N`)

*Files: `core/src/display_object/interpolation.rs`, `core/src/display_object.rs`
(transform setters and getters, `BitmapCache::changed_tick`),
`core/src/display_object/container.rs`, `core/src/player.rs`, `desktop/src/app.rs`
(`schedule_subframes`, `render_fraction`, `adapt_interpolation`), `desktop/src/cli.rs`,
`render/wgpu/src/filters/batch.rs` (`ScratchBuffers`).*

`--interpolate N` draws N frames per movie frame (for example 6: a 24 fps movie at 144
fps), blending moving objects between movie frames. Flash animations move, rotate, scale
and fade objects by changing their transforms every frame, and draw other things by
replacing objects' contents (frame-by-frame animation). So:
* While interpolation is on, each display object records its transform (matrix and color
  transform) from before the current movie frame first changes it (in the base's
  transform setters). Objects created or placed in the current frame (instances clone a
  library symbol, so that state resets on clone, and placing a child resets it) aren't
  blended.
* While drawing frame j of N after a movie frame, the transform getters return the blend
  at (j + 1) / N: each axis's angle and length are blended (so rotations don't shrink
  halfway), and the translation and color transform linearly. Changes that are too large
  to be motion (moves over 100 pixels, turns over 90°, scaling by more than 4x, as when an
  animation jumps back to its start) aren't blended. The blend is cached per object and
  frame, since rendering reads transforms many times (without the cache, a render took
  9 ms instead of 3).
* Cached bitmaps whose contents changed in the current movie frame are redrawn for
  every frame in between (`changed_tick`), since what they contain is blended too.
* What objects contain still changes at movie frames, and frames show the previous movie
  frame blended toward the current one, so interpolation delays what's shown by up to
  one movie frame.

The desktop player wakes up at each frame's time (only ticking the player when a movie
frame or timer is due; ticking also runs mouse picking), presents frames paced at the
frame interval (40), and skips frames that are already late. When more than a tenth of
the frames are skipped for two seconds (not counting one-off stalls), it draws fewer:
the next divisor of N, so that frames stay evenly spaced (6 → 3 → 2 → 1), and tries more
again after a while if the GPU and main thread look like they have time for it: the
busier of the two (GPU time measured from Metal command buffers' start and end times,
main-thread time from ticking and drawing), scaled by the higher number over the current
one, must be at most 80%. Without that check, it kept trying 6 in heavy scenes and
dropping back, with uneven frames each time. `RUFFLE_FIXED_INTERPOLATION` turns the adapting off (for measuring).

The batched filters (34) used to create their uniform and vertex buffers every pass, which
was ~13% of render time at 144 fps. They now reuse buffers (`ScratchBuffers`), written
through the staging belt.

Effect (5120x1440 at 144 Hz):
* A light scene: 144 fps with `--interpolate 6` (503 of 507 frames on screen
  for exactly 1 refresh), main thread 25% busy.
* A heavy scene: the GPU manages ~75 frames per second, so it settles at 3
  (72 fps, ~3 ms of CPU per frame, GPU ~64% busy); 2 gives 48 fps (3 refreshes each).

What limits it in heavy scenes: much of the animated content is under filters (glows), so
~25-110 cached bitmaps are redrawn and filtered every movie frame, and with interpolation
every drawn frame (their contents move). At 6 in a medium scene that was ~8 ms
of GPU time per frame (the budget is 6.9 ms), and the main thread was 65-85% busy.
Redrawing a cache between movie frames only when something inside it moves (rather than
whenever it changed in that movie frame) was tried and didn't help: in those scenes nearly
all of them contain moving parts.

### 43. Roots of loaded movies have a default `PerspectiveProjection`

*Files: `core/src/display_object/movie_clip.rs` (`replace_with_movie`),
`core/src/avm2/globals/flash/geom/perspective_projection.rs` (`get_width`); test
`avm2/perspective_projection_loaded`.*

Animate's layer depth feature compiles a helper class (`privatePkg.___LayerProp___`) into
the SWF. On every `exitFrame`, each instance reads
`root.transform.perspectiveProjection.focalLength`. Only the player's root movie had a
default `PerspectiveProjection`; for the root of a movie loaded with `Loader`, the getter
returned `null`. So in loaded content, every such object threw a `TypeError` on every
frame, which Ruffle logs with a stack trace, and the layer's color effects, filters and
visibility were never applied. One loaded movie with two such objects logged 528 errors in
6 seconds.

The root of a loaded movie now gets the same default as the player's root (field of view
55, centered on the movie). As in Flash, its `focalLength` uses the loaded movie's width
rather than the stage's. The test's expected output comes from the AIR 51.3 runtime
(`adl`). Its SWFs are compiled with Ruffle's `tools/asc/asc.jar` (against the
`playerglobal_import.abc` in the build output), because the AIR SDK's compiler emits ABC
47.16, which Ruffle doesn't parse. That movie renders identically before and after.

## Memory: long sessions

Replacing most of the content over and over used to make the movie slower and slower:
seven 60-second rounds alternating between a heavy and a medium scene took `run_frame`
from 4 to 23 ms, the GC heap from 0.36M to 3.1M objects, and RSS from 1.0 to 2.5 GB.
Removed display objects were never freed. Every one stayed alive, and AVM2 orphans keep
running their frame scripts. Light scenes barely grew. Four causes:

### 15. Weak event listeners

*Files: `core/src/avm2/events.rs`, `core/src/avm2/globals/flash/events/event_dispatcher.rs`.*

`addEventListener(..., useWeakReference = true)` was ignored, so every listener
kept its function (and through it, its object) alive. The test content registers ~1300 weak
listeners. A weak listener now holds a `FunctionObjectWeak`, and collected
listeners are pruned when the list is next used. Bound methods are cached on their
receiver (`bound_methods`), so a weakly held method closure stays alive exactly as
long as its object, like in Flash.

### 16. Weakly keyed `Dictionary`

*Files: `core/src/avm2/object/dictionary_object.rs`, `Dictionary.as`,
`core/src/avm2/globals/flash/utils/dictionary.rs`.*

`new Dictionary(true)` held its keys strongly. Object keys of such dictionaries now
live in a separate table of `(WeakObject, Value)` slots, indexed by key pointer,
so an entry disappears when its key is collected. Enumeration visits those slots
after the normal ones (enumerant indices from `1 << 30`). Collected entries are
pruned, and the table compacted, as it grows.

### 17. Ephemerons: weak keys whose values refer back to them

*Files: gc-arena fork (see [Patched dependencies](#patched-dependencies)), `dictionary_object.rs`.*

With 15 and 16, growth halved, but removed clips still leaked. `RUFFLE_LEAK_PATHS`
(below) showed the path:

```
ClassObjectData:<document class> -> StageObjectData:<a static cache> -> DictionaryObjectData:Dictionary
  -> ScriptObjectData:Object[<property>] -> StageObjectData:<a container clip> -> MovieClipData
```

The content caches `cache[clip] = {container: <the clip that contains
clip>, ...}` in a weakly keyed dictionary. The value refers to the container,
whose display list refers to the key, so plain weak keys never die. (Flash's
`Dictionary` holds values strongly too, so this probably leaks in Flash Player
as well.)

Weakly keyed dictionary entries are now *ephemerons*, like JavaScript's
`WeakMap`: a value is only kept alive while its key is reachable from outside the
entry. Programs can only tell the difference by enumerating a weak dictionary and
seeing an entry disappear whose key is reachable only through dictionary values.
That is the same nondeterminism weak keys already have.

gc-arena has no ephemerons, so the fork adds them:

* `Trace::ephemeron_key_reachable(key)` (default `true`): the collector returns
  whether the key is already marked. If not, it remembers the object being traced.
  The dictionary then skips that value.
* When the gray queues are empty, the collector traces those objects again.
  Newly marked keys get their values traced, which can mark more keys. This
  repeats until a round marks nothing new (`ephemerons_settled`). The list is kept
  until sweeping starts, in case a write barrier resumes marking.

The dictionary must never touch the value of a dead key, which may have been
freed:

* Tracing skips values of dropped keys, but still traces every key weakly. Tracing
  keeps a key's allocation (and so its pointer in the index) from being reused. An
  early version skipped dropped keys entirely. A new object then got a dead key's
  address, and looking it up returned a freed value (segfault in `get_property`).
* Enumeration checks liveness with `upgrade`, not `is_dropped`. During an
  incremental sweep, a key that's about to be dropped can't be upgraded, but its
  value may already be freed.

A standalone test (scratch crate, not in the repo) checks the semantics with full
and incremental collections. The cases are:
* a key reachable only through its own value: both freed
* a key chain inserted in reverse order: kept, which needs several rounds
* an unreachable key whose value is reachable elsewhere: only the key freed
* a key made reachable at various points of an incremental mark, and after
  marking finished: kept

gc-arena's own tests pass (except the `ui` compile-error snapshot test, which also
fails unpatched with this compiler).

### 18. Image loaders were never removed

*File: `core/src/loader.rs` (`movie_loader_complete`).*

`LoadManager::run_exit_frame` removes a SWF loader once its `init`/`complete`
events fire. Images (JPEG/PNG/GIF) don't run frames, so their loaders stayed
registered forever. Each one kept its `Loader`, `LoaderInfo` and everything
reachable from them. The test content loaded a dozen images every time it replaced
its content (the `[loader]` lines showed them). An
AVM2 image loader is now removed as soon as its events have fired.

### 22. Vtables share their superclass's traits

*Files: `core/src/avm2/vtable.rs` (`VTableTraits`), `core/src/avm2/property_map.rs`.*

What was still growing? Most RSS wasn't GC objects. After a forced full collection
the GC heap halved, but malloc still held 1.5 GB in 3.7M
allocations. To attribute it, launch with `MallocStackLogging=lite`, then run
`malloc_history <pid> -callTree -ignoreThreads -collapseRecursion`. Charging each
node's own bytes to its nearest Ruffle frame gives, after 4 content changes:

| Live bytes | Allocated by |
|---|---|
| 318 MB | `VTable::init_vtable` |
| 175 MB | `swf` parsing (mostly shape records) |
| 105 MB | wgpu `create_buffer` (mesh data) |
| 70 MB | `MovieClip::preload` |

Every class's vtable copied its superclass's whole `PropertyMap` of resolved traits.
MovieClip has ~200, and the test content defines thousands of MovieClip subclasses, so
there were tens of thousands of copies. Now a vtable keeps only the traits its class
adds or changes, over a shared `Rc` of its superclass's flattened map:
* Lookups check the class's own traits first, so they find the same entries as
  before (`PropertyMap::insert` puts new entries first).
* Changing an inherited entry (adding a setter to an inherited getter) copies it
  into the class's own traits.
* A class builds its own flattened map only when it gets a subclass, and keeps
  it.
* `copy_interface_properties` no longer re-inserts interface entries that are already
  identical.

`init_vtable` went from 318 MB to 72 MB after 4 content changes (the rest is mostly method
tables, still copied per class). Logic time didn't change measurably. The
lookup checks a usually tiny own map first, and skips it when it's empty.

Also, `swf::Reader::read_define_shape` now shrinks the shape record vector to fit.
Pushing records one at a time left up to half of its capacity unused, for as long
as the movie lives.

### 23. A minimum garbage collection pace

*File: `core/src/player.rs` (`add_minimum_gc_work`).*

In a test replacing the content 20 times, logic time crept up from ~3 to 5-8.5 ms, and
stayed there while idle. It tracked the number of orphans (500-1,000). All the
orphans checked were unreachable: garbage that hadn't been collected.
Orphaned movie clips keep running their frames until the GC frees them. But gc-arena
only collects in proportion to allocations, and an idle movie allocates little, so a
cycle over a heap of millions of objects could take minutes.

Now every frame adds GC debt in proportion to the heap size (`adjust_debt`), so a
whole cycle takes at most ~10 seconds even without allocations
(`RUFFLE_GC_MAX_CYCLE_SECS`, 0 disables it). With gc-arena's default pacing, a cycle
costs ~0.55 debt units per object.

Effect, 20 content changes, before and after (both with 22):

| | Before | After |
|---|---|---|
| `run_frame` later on | crept up to 5-8.6 ms | stays at 2.2-4.2 ms (including the GC work) |
| Orphans | up to 1,057 | 29-120 |
| GC heap | up to 2.3M objects | at most 0.92M |
| RSS after 19 changes | 2.29 GB | 1.99 GB (~68 MB per change) |

### Results

Laptop display (2880x1800 fullscreen), seven 60-second rounds alternating between a heavy
and a medium scene. Scenes differ between runs, so these are indicative:

| Build | `run_frame` start → end | GC objects | Orphaned clips | RSS |
|---|---|---|---|---|
| Before (weak listeners + weak keys only) | 4 → 23 ms | 0.36M → 3.1M | 78 | 1.0 → 2.5 GB |
| + image loader fix (18) | 2.6 → 11.1 ms | 0.32M → 1.58M | 83 | 1.0 → 2.4 GB |
| + ephemerons (17) | 3.8 → 2.4-4.5 ms (flat) | 0.27M → 0.8-1.2M (falls after collections) | 1 (unreachable) | 1.05 → 2.0-2.5 GB |

With all fixes, reachable GC objects grew 0.26M → 0.46M over the seven rounds. Most
of that is classes, vtables, methods and clips of SWFs not loaded before (for
example `ClassData` 7.7K → 25K), which the content keeps. The total heap is ~2.7x the
reachable part with gc-arena's default pacing.

RSS barely changed, though, so most of it isn't GC objects. At the end: ~1.3 GB of
small malloc blocks (3.5M allocations, 6% fragmentation), ~1 GB of GPU memory,
and a 169 MB pool of `MTLResourceList`s (one per command buffer in flight, at its
high-water mark). Memory doesn't grow while idle, only when new content
loads. A test replacing the content 20 times, with everything up to 21:
* No crashes.
* No leaked clips: the orphans checked were never reachable from the root.
* `run_frame` stayed at 2-7.5 ms, with no trend. The spikes coincide with many
  orphans (600-800) right after a content change, until they're collected.
* RSS still grew by ~100 MB per change (0.85 → 2.8 GB), and the GC heap to 2.3M objects.

(That test was before 22.) Ideas: share method tables between vtables (like
22), drop parsed shape records after tessellation (re-parse from the SWF when
needed), fewer command buffers in flight, tune the GC pacing.

## Patched dependencies

`Cargo.toml` has `[patch.crates-io]` entries that take wgpu and gc-arena from forks, at
tags. Changes are marked `Ruffle patch` in the sources.

* [sushipando/wgpu](https://github.com/sushipando/wgpu): branch `perf` on upstream
  `v30.0.1`, tags `perf-30.0.1-<n>`. All wgpu crates (`wgpu`, `wgpu-core`, `wgpu-hal`,
  `wgpu-types`, `naga`) come from the fork, so that they share one `wgpu-types`.
  * wgpu-hal (Metal): **Lazy command buffers.** `begin_encoding` no longer creates an
    `MTLCommandBuffer`; `ensure_raw_cmd_buf` does on first use. Empty ones (such
    as the per-pass barrier encoders wgpu-core creates, since barriers are no-ops
    on Metal) are skipped at submit. `CommandBuffer::raw` is now an `Option`.
    Effect: render CPU 150-300 → ~37 ms/frame in the first heavy scene.
  * wgpu-hal (Metal): **`gpu_stats`.** With `RUFFLE_PERF`, a completion handler records each command
    buffer's `GPUStartTime`/`GPUEndTime` and prints GPU busy time once a second.
    It needs the `objc2-core-foundation` feature of `objc2-metal`. With
    `RUFFLE_GPU_LABELS`, it also sums time per command buffer label (see Tooling).
  * wgpu-hal (Metal): exceeding 4096 outstanding command buffers now panics where it used to return
    `DeviceError::Lost`, which ended in the same panic in the GUI anyway.
  * wgpu-core: **sparse usage-scope clearing.** Each render pass
    takes a pooled `UsageScope` sized to all buffers/textures on the device, and
    clearing it on drop (plus resizing on reuse) was O(all resources). That was ~24%
    of render CPU with thousands of shapes. `clear` now only resets the indices
    that were used (from the `owned` bitvector), keeping the tables' size.
    Effect: 1283 → 129 samples; render CPU per pass 0.10 → 0.04 ms.
  * wgpu-core: **render passes sharing command buffers** on Metal (see 20), with a
    runtime switch in `ruffle_patch.rs`.
  * wgpu-hal (Metal): **presenting after a minimum duration** (see 40) and **GPU busy
    time on request** (see 41), switched through `ruffle_patch.rs` too.
* [sushipando/gc-arena](https://github.com/sushipando/gc-arena): branch `perf` on upstream
  `v0.7.0`, tag `perf-0.7.0-1`.
  * **Ephemerons** (`collect.rs`, `context.rs`): see 17.
  * **`debug_paths`** (`debug_paths.rs`, `types.rs`): the per-type vtable also
    stores a tracing function for a non-collector `Trace` implementation, plus
    the type name. `find_paths(root, targets)` runs a breadth-first search from
    the root over strong pointers (it doesn't follow ephemeron values). It returns
    a shortest path to each target, with each node's pointer and type name. Only
    `RUFFLE_LEAK_PATHS` uses it.

Updating to a new upstream version of either: in the fork, rebase `perf` onto the new
upstream tag (`git rebase --onto <new tag> <old tag> perf`), resolve conflicts, tag the
result `perf-<version>-1`, and push the branch and the tag. Then change the tags in
`Cargo.toml`, run `cargo update`, and run the tests. Tags are never moved, so older
commits of this branch keep building after the fork's branch is rebased.

## Tooling

### `RUFFLE_PERF=1`

Prints once a second to stderr:

* `[perf]`: frames run per second (logic fps), average/max `run_frame` time, renders per
  second (shown fps) and average/max CPU time of `Player::render`, the rest of
  `tick`, and how busy the main thread was (`core/src/perf_stats.rs`).
* `[gpu]`: per-frame counts of render passes, offscreen targets, cache entries,
  filters, and blends (`render/wgpu/src/perf_counters.rs`).
* `[gpu-named]`: ad-hoc breakdowns, such as blend mode and size, filtered cache
  entries by filter type, and wave scheduler stats.
* `[gpu-time]`: GPU busy time per second, command buffers per second, and GPU
  time per command buffer (patched wgpu-hal).

The `[perf]` line also shows the GC heap size (`gc objects`), the number of AVM2
orphan display objects (`orphans`), and the number of loaders in the
`LoadManager` (`loaders`).

### GPU time per pass type

`RUFFLE_GPU_LABELS=1` (with `RUFFLE_PERF`) prints `[gpu-label]` once a second: GPU
time summed by command buffer label, with digits removed (patched wgpu-hal). Each
command buffer is only charged for the time between the previous one's end (or its
own start) and its end, because they overlap on the GPU. Labels need a build with
`--features ruffle_render_wgpu/render_debug_labels`. Turn the `share-cmdbufs` flag off, so
that each pass has its own command buffer.

`RUFFLE_BLEND_DEBUG=1` prints each display object with a `Layer` or `Alpha` blend mode
once, with its path, class, layer bounds and children's blend modes (`[blend]`).

### Metal System Trace

Instruments' Metal System Trace shows what the GPU actually executes, per encoder,
including the idle time between them (`[gpu-time]` measures command buffers from
start to end, which includes such gaps, so it says 100% while the GPU may be idle half
of the time). From the command line:

```sh
xcrun xctrace record --template 'Metal System Trace' --attach <pid> --time-limit 2s --output mst.trace
xcrun xctrace export --input mst.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' > intervals.xml
```

Also export `metal-application-encoders-list` (encoder labels) and
`metal-gpu-submission-to-command-buffer-id` (which maps an interval's
`gpu-submission-id` to its encoder). The XML deduplicates values (`id`/`ref`
attributes). Encoder labels need a build with `render_debug_labels`, which makes the
CPU side slower (debug groups).

### Upper bounds

Two flags exist only to measure what a kind of work costs in a scene: `skip-cache-filters`
(cached bitmaps aren't filtered) and `complex-as-normal` (complex blends are drawn as
normal blends, so they don't split render passes). Both make the content look wrong.

`RUFFLE_BLUR_DEBUG=1` (with `RUFFLE_PERF`) prints each blur of over 20M texture samples
(`[blur]`, with its sizes, quality and source size).

### Leak debugging

`RUFFLE_GC_SLEEP_FACTOR=<f>` sets gc-arena's `Pacing::sleep_factor` (default 0.5).
The collector waits until the heap has grown by that fraction of the last cycle's
live size before collecting again.

* `RUFFLE_ORPHAN_DEBUG=1` (with `RUFFLE_PERF`): every 240 renders, prints an
  `[orphans]` histogram of live orphans by AVM2 class, and a `[loader]` line per
  loader still in the `LoadManager`.
* `RUFFLE_LEAK_PATHS=<AS3 class>` (with both): also prints, as `[path]` lines, why
  up to 8 orphans of that class are still alive. Each line is a shortest chain of
  strong references from the GC root, using `gc_arena::debug_paths`. AVM2 objects
  are shown with their class and the property holding the next object
  (such as `Object[name]` or `Class.field`), and display objects with their name.
  Orphans that are unreachable (garbage not collected yet) have no path. It walks
  the whole heap (~150 ms for 1M objects).

The counters cost nothing when `RUFFLE_PERF` isn't set.

### Optimization flags

Optimization flags (`render/src/perf_flags.rs`, all on by default in the desktop
player): `bounded-blends`, `lazy-resolve`, `layer-atlas`, `visible-region`,
`wave-schedule`, `pick-cache`, `async-submit`, `opaque-multiply`,
`filter-batching`, `share-cmdbufs`, `blur-downscale` (off by default in the
library, see 21), `async-present`, `overlap-record`, `blur-merge` (off by default in the
library, see 25), `fetch-blends`, `merge-chunks`, `cache-slack`, `batch-large-filters`,
`cache-bind-groups`, `filter-together`, `direct-blends`, `release-shapes`, `evict-meshes`,
`paced-present`. Measurement only (off): `skip-cache-filters`, `complex-as-normal`.

### Measurement procedure

1. Launch fullscreen with `RUFFLE_PERF=1`.
2. Go to the scene to measure.
3. Content can differ between sessions, so compare several sessions, or normalize
   (for example per render pass).
4. CPU profiles: `sample <pid> 10 -file out.txt` and read the main thread.
5. Launch Ruffle from zsh with `setopt NO_BG_NICE`. Otherwise background (`&`)
   jobs run at nice 5, which made a heavy scene drop from ~5 to <1 fps while idle in one
   session.

## Testing

`cargo test --release -p tests --features imgtests` passes on this branch (4577
passed, 37 ignored, the same feature-gated ignores as upstream), including the 150
`visual/` image comparisons on Metal, with all optimizations enabled. It was last
run with everything up to 41. A corrupted
reference image was confirmed to fail, so the comparisons really run. These tests
render to an offscreen `TextureTarget`, so they don't exercise async submission
(7), which was only verified in the desktop player.

## Known issues

* **Memory in long sessions.** See "Memory: long sessions". The leaks found are
  fixed, and logic time no longer grows (15-18, 23). With 37-39, RSS grows by ~22 MB per
  content change (627 → 806 MB over 8 changes), down from ~65 MB: meshes of
  shapes that aren't drawn are freed after 2 minutes (38), shape definitions are parsed on
  demand (37), and vtables share method tables (39). What's left is mostly data that
  the content keeps (data and classes of loaded SWFs) and per-clip data (~30 MB of `MovieClipShared`
  and preload data after 4 changes, mostly empty label and scene tables).

## Known remaining costs and ideas

With 24-36, heavy scenes run at 60 fps on the laptop display (2880x1800),
with the GPU at ~10-16 ms/frame and the main thread 25-50% busy. The external
5120x1440 display shows fewer pixels (the letterboxed stage is 2514x1440, see 9), but
hasn't been measured at 60 fps. What's left, roughly by expected value:

* **Blend layers that aren't a single fill.** Shapes with several fills or strokes
  (~230 single-command layers per frame in one heavy scene, mostly Multiply and
  Overlay) still go through the layer atlas, a copy out of it, and a composite (35).
  Sampling the atlas directly (instead of copying each layer out of it) would save the
  copies; it needs the layer's slot known when its composite is recorded.
* **Gradient and bitmap fills with complex blends** could be drawn directly like solid
  fills (35), with Metal versions of the gradient and bitmap fill shaders that read the
  color attachment.
* **Logic** at 60 fps costs 2-7 ms per frame in a heavy scene, mostly `enter_frame` and
  `construct_frame` walking the whole display list every frame. Skipping subtrees
  that can't change would need careful invalidation.
* **Memory:** keep fewer command buffers in flight (the `MTLResourceList` pool), make
  `MovieClipShared` smaller (box its rarely used label, scene and sound tables), and tune
  the GC pacing (see "Memory: long sessions").

Measured and ruled out: memoryless MSAA (passes on the main multisampled target are
expensive, ~0.45 ms each in 28, but 27-28 left one or two per frame), more aggressive
blur downscaling (blur cost was per-pass overhead, not
texture samples, once 25-26 were in), and running CPU work earlier (24: in GPU-bound
scenes the CPU already waits for drawables).
