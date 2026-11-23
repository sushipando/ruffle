//! Content-addressed LRU cache for looping pure-vector filtered CABs.
//!
//! Animated capes, wings, weapons, and pulsing auras in Flash/AQW frequently repeat short
//! 10-60 frame vector animations with identical shapes, transforms, color transforms, and
//! filters. Caching their finished filtered textures by exact canonical inputs lets looping
//! CABs skip 4x MSAA rasterization and blur/glow filtering completely after their first loop,
//! while strictly guarding against stale bitmaps, sub-pixel mismatches, pointer ABA reuse,
//! hash collisions, and unbounded VRAM growth.

use crate::as_texture;
use crate::descriptors::Descriptors;
use crate::filters::FilterAtlasResult;
use crate::mesh::mesh_id;
use lru::LruCache;
use ruffle_render::bitmap::PixelSnapping;
use ruffle_render::commands::{Command, CommandList, RenderBlendMode};
use ruffle_render::filters::Filter;
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;
use swf::{Color, Twips};

/// Maximum VRAM retained by the small/medium keyframe tier (<= 384x384 px = <= 576 KB/frame).
/// Isolated from both large entries and interpolated subframes so giant wings or 6x-volume
/// subframe animations can never starve dozens of smaller glowing weapons, flames, and runes.
const SMALL_MAX_BYTES: usize = 384 * 1024 * 1024;
/// Pixel area threshold separating the small/medium tier from the large tier (384x384).
const SMALL_MAX_PIXELS: u64 = 384 * 384;
/// Maximum VRAM retained by the large keyframe tier (384x384 < pixels <= 1024x1024).
const LARGE_MAX_BYTES: usize = 256 * 1024 * 1024;
/// Dedicated VRAM tier for interpolated subframe CABs (<= 384x384 px), sized so heavy
/// 19-20 player rooms with 75+ glowing CABs can keep all 5 interpolated subframe phases
/// warm at 144 fps without overflowing.
const SUBFRAME_MAX_BYTES: usize = 256 * 1024 * 1024;
/// Maximum single CAB pixel area eligible for caching (1024x1024 = 4 MB).
const MAX_ENTRY_PIXELS: u64 = 1024 * 1024;
/// Maximum VRAM retained by the small/medium blend-layer tier (<= 512x512 px).
const BLEND_SMALL_MAX_BYTES: usize = 320 * 1024 * 1024;
/// Pixel area threshold separating small/medium and large blend layers (512x512).
const BLEND_SMALL_MAX_PIXELS: u64 = 512 * 512;
/// Maximum VRAM retained by the large blend-layer tier (512x512 < px <= 2560x1440).
const BLEND_LARGE_MAX_BYTES: usize = 64 * 1024 * 1024;
/// Maximum single blend-layer pixel area eligible for caching (covers up to 2560x1440).
const BLEND_MAX_ENTRY_PIXELS: u64 = 2560 * 1440;
/// Maximum single CAB pixel area eligible for promoting interpolated subframes (384x384,
/// matching the dedicated subframe tier).
pub const SUBFRAME_PROMOTE_MAX_PIXELS: u64 = SMALL_MAX_PIXELS;
/// Maximum command count per CAB eligible for canonical key serialization.
const MAX_COMMANDS: usize = 512;
/// Evict cached textures unused for this many submitted frames (~62 s at 144 fps, ~125 s at
/// 72 fps) so player attack/skill animations on 15-45 s cooldowns stay warm in-room while
/// rooms left behind still free VRAM smoothly.
const MAX_IDLE_FRAMES: u64 = 9000;
/// Refuse to evict a cached entry used within this many frames (~5 s at 144 fps) so active
/// loops stay pinned at 100% hit rate instead of thrashing when a room exceeds the tier budget.
const HOT_WINDOW_FRAMES: u64 = 720;
/// Size of the 2-hit admission filter (only promote keys seen at least twice once a tier is
/// >50% full).
const SEEN_CAPACITY: usize = 32768;

/// Exact canonical content key for a pure-vector filtered CAB.
/// Comparing `payload` byte-for-byte eliminates hash-collision bugs completely.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CabContentKey {
    used_size: (u32, u32),
    quality: u8,
    clear: [u8; 4],
    payload: Vec<u8>,
}

impl CabContentKey {
    /// Attempts to serialize a CAB's exact rendering & filtering inputs into a canonical key.
    /// Returns `None` if the CAB is too large/complex or contains any command that cannot be
    /// safely keyed (`RenderStage3D`, `RenderAlphaMask`, `BlendMode::Shader`, or a
    /// `RenderBitmap` currently aliased into the transient per-frame `filter_atlas`).
    pub fn try_from_entry(
        used_size: (u32, u32),
        clear: Color,
        quality: StageQuality,
        filters: &[Filter],
        commands: &CommandList,
        filter_atlas: Option<&FilterAtlasResult>,
    ) -> Option<Self> {
        if filters.is_empty() {
            return None;
        }
        Self::try_from_entry_inner(used_size, clear, quality, filters, commands, filter_atlas)
    }

    /// Same as `try_from_entry`, but also permits unfiltered (`filters.is_empty()`) entries
    /// for zero-VRAM same-handle skip checks on `Persistent` CABs.
    pub fn try_from_any_entry(
        used_size: (u32, u32),
        clear: Color,
        quality: StageQuality,
        filters: &[Filter],
        commands: &CommandList,
        filter_atlas: Option<&FilterAtlasResult>,
    ) -> Option<Self> {
        Self::try_from_entry_inner(used_size, clear, quality, filters, commands, filter_atlas)
    }

    /// Serializes an offscreen blend layer's exact rendering inputs into a canonical key,
    /// normalized to layer-local `(0, 0)` coordinates by subtracting `origin`.
    pub fn try_from_blend_layer(
        used_size: (u32, u32),
        origin: (i32, i32),
        clear: Color,
        quality: StageQuality,
        is_layer: bool,
        commands: &CommandList,
        filter_atlas: Option<&FilterAtlasResult>,
    ) -> Option<Self> {
        let pixels = u64::from(used_size.0) * u64::from(used_size.1);
        if pixels == 0 || pixels > BLEND_MAX_ENTRY_PIXELS {
            return None;
        }
        if commands.commands.is_empty() || commands.commands.len() > MAX_COMMANDS {
            return None;
        }
        let mut payload = Vec::with_capacity(64 + commands.commands.len() * 48);
        payload.push(if is_layer { 0xB1 } else { 0xB0 });
        let offset = (
            Twips::from_pixels_i32(origin.0),
            Twips::from_pixels_i32(origin.1),
        );
        let mut total_cmds = 0usize;
        if !encode_commands(
            commands,
            &mut payload,
            0,
            &mut total_cmds,
            offset,
            filter_atlas,
        ) {
            return None;
        }
        Some(Self {
            used_size,
            quality: quality_to_byte(quality),
            clear: [clear.r, clear.g, clear.b, clear.a],
            payload,
        })
    }

    fn try_from_entry_inner(
        used_size: (u32, u32),
        clear: Color,
        quality: StageQuality,
        filters: &[Filter],
        commands: &CommandList,
        filter_atlas: Option<&FilterAtlasResult>,
    ) -> Option<Self> {
        let pixels = u64::from(used_size.0) * u64::from(used_size.1);
        if pixels == 0 || pixels > MAX_ENTRY_PIXELS {
            if pixels > MAX_ENTRY_PIXELS {
                crate::perf_counters::count_named("cab-nokey/size");
            }
            return None;
        }
        if commands.commands.is_empty() || commands.commands.len() > MAX_COMMANDS {
            if commands.commands.len() > MAX_COMMANDS {
                crate::perf_counters::count_named("cab-nokey/cmds");
            }
            return None;
        }
        let mut payload = Vec::with_capacity(64 + commands.commands.len() * 48);
        payload.extend_from_slice(&(filters.len() as u32).to_le_bytes());
        for filter in filters {
            if !encode_filter(filter, &mut payload) {
                crate::perf_counters::count_named("cab-nokey/filter");
                return None;
            }
        }
        let mut total_cmds = 0usize;
        if !encode_commands(
            commands,
            &mut payload,
            0,
            &mut total_cmds,
            (Twips::ZERO, Twips::ZERO),
            filter_atlas,
        ) {
            return None;
        }
        Some(Self {
            used_size,
            quality: quality_to_byte(quality),
            clear: [clear.r, clear.g, clear.b, clear.a],
            payload,
        })
    }

    fn fast_hash(&self) -> u64 {
        let mut hasher = fnv::FnvHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

fn encode_filter(filter: &Filter, out: &mut Vec<u8>) -> bool {
    match filter {
        Filter::BlurFilter(b) => {
            out.push(1);
            out.extend_from_slice(&b.blur_x.get().to_le_bytes());
            out.extend_from_slice(&b.blur_y.get().to_le_bytes());
            out.push(b.flags.bits());
            true
        }
        Filter::GlowFilter(g) => {
            out.push(2);
            out.extend_from_slice(&[g.color.r, g.color.g, g.color.b, g.color.a]);
            out.extend_from_slice(&g.blur_x.get().to_le_bytes());
            out.extend_from_slice(&g.blur_y.get().to_le_bytes());
            out.extend_from_slice(&g.strength.get().to_le_bytes());
            out.push(g.flags.bits());
            true
        }
        Filter::ColorMatrixFilter(cm) => {
            out.push(3);
            for &v in &cm.matrix {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            true
        }
        Filter::DropShadowFilter(d) => {
            out.push(4);
            out.extend_from_slice(&[d.color.r, d.color.g, d.color.b, d.color.a]);
            out.extend_from_slice(&d.blur_x.get().to_le_bytes());
            out.extend_from_slice(&d.blur_y.get().to_le_bytes());
            out.extend_from_slice(&d.angle.get().to_le_bytes());
            out.extend_from_slice(&d.distance.get().to_le_bytes());
            out.extend_from_slice(&d.strength.get().to_le_bytes());
            out.push(d.flags.bits());
            true
        }
        Filter::BevelFilter(b) => {
            out.push(5);
            out.extend_from_slice(&[
                b.shadow_color.r,
                b.shadow_color.g,
                b.shadow_color.b,
                b.shadow_color.a,
                b.highlight_color.r,
                b.highlight_color.g,
                b.highlight_color.b,
                b.highlight_color.a,
            ]);
            out.extend_from_slice(&b.blur_x.get().to_le_bytes());
            out.extend_from_slice(&b.blur_y.get().to_le_bytes());
            out.extend_from_slice(&b.angle.get().to_le_bytes());
            out.extend_from_slice(&b.distance.get().to_le_bytes());
            out.extend_from_slice(&b.strength.get().to_le_bytes());
            out.push(b.flags.bits());
            true
        }
        _ => false,
    }
}

fn quality_to_byte(quality: StageQuality) -> u8 {
    match quality {
        StageQuality::Low => 0,
        StageQuality::Medium => 1,
        StageQuality::High => 2,
        StageQuality::Best => 3,
        StageQuality::High8x8 => 4,
        StageQuality::High8x8Linear => 5,
        StageQuality::High16x16 => 6,
        StageQuality::High16x16Linear => 7,
    }
}

fn encode_matrix(m: &Matrix, offset: (Twips, Twips), out: &mut Vec<u8>) {
    out.extend_from_slice(&m.a.to_bits().to_le_bytes());
    out.extend_from_slice(&m.b.to_bits().to_le_bytes());
    out.extend_from_slice(&m.c.to_bits().to_le_bytes());
    out.extend_from_slice(&m.d.to_bits().to_le_bytes());
    out.extend_from_slice(&(m.tx - offset.0).get().to_le_bytes());
    out.extend_from_slice(&(m.ty - offset.1).get().to_le_bytes());
}

fn encode_transform(t: &Transform, offset: (Twips, Twips), out: &mut Vec<u8>) -> bool {
    if (offset.0 != Twips::ZERO || offset.1 != Twips::ZERO)
        && (t.perspective_projection.is_some() || t.tz != 0.0)
    {
        return false;
    }
    encode_matrix(&t.matrix, offset, out);
    for v in t.color_transform.mult_rgba_normalized() {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    for v in t.color_transform.add_rgba_normalized() {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out.extend_from_slice(&t.tz.to_bits().to_le_bytes());
    match t.perspective_projection {
        Some(p) => {
            out.push(1);
            out.extend_from_slice(&p.field_of_view.to_bits().to_le_bytes());
            out.extend_from_slice(&p.center.0.to_bits().to_le_bytes());
            out.extend_from_slice(&p.center.1.to_bits().to_le_bytes());
        }
        None => out.push(0),
    }
    true
}

fn encode_commands(
    commands: &CommandList,
    out: &mut Vec<u8>,
    depth: u8,
    total_cmds: &mut usize,
    offset: (Twips, Twips),
    filter_atlas: Option<&FilterAtlasResult>,
) -> bool {
    if depth > 4 {
        crate::perf_counters::count_named("cab-nokey/depth");
        return false;
    }
    out.extend_from_slice(&(commands.commands.len() as u32).to_le_bytes());
    for cmd in &commands.commands {
        *total_cmds += 1;
        if *total_cmds > MAX_COMMANDS {
            crate::perf_counters::count_named("cab-nokey/cmds");
            return false;
        }
        match cmd {
            Command::RenderShape { shape, transform } => {
                out.push(1);
                out.extend_from_slice(&mesh_id(shape).to_le_bytes());
                if !encode_transform(transform, offset, out) {
                    crate::perf_counters::count_named("cab-nokey/3d");
                    return false;
                }
            }
            Command::DrawRect { color, matrix } => {
                out.push(2);
                out.extend_from_slice(&[color.r, color.g, color.b, color.a]);
                encode_matrix(matrix, offset, out);
            }
            Command::DrawLine { color, matrix } => {
                out.push(3);
                out.extend_from_slice(&[color.r, color.g, color.b, color.a]);
                encode_matrix(matrix, offset, out);
            }
            Command::DrawLineRect { color, matrix } => {
                out.push(4);
                out.extend_from_slice(&[color.r, color.g, color.b, color.a]);
                encode_matrix(matrix, offset, out);
            }
            Command::PushMask => out.push(5),
            Command::ActivateMask => out.push(6),
            Command::DeactivateMask => out.push(7),
            Command::PopMask => out.push(8),
            Command::Blend(inner, RenderBlendMode::Builtin(mode)) => {
                out.push(9);
                out.push(*mode as u8);
                if !encode_commands(inner, out, depth + 1, total_cmds, offset, filter_atlas) {
                    return false;
                }
            }
            Command::BlendBounded(inner, RenderBlendMode::Builtin(mode), bounds) => {
                out.push(10);
                out.push(*mode as u8);
                out.extend_from_slice(&(bounds.x_min - offset.0).get().to_le_bytes());
                out.extend_from_slice(&(bounds.x_max - offset.0).get().to_le_bytes());
                out.extend_from_slice(&(bounds.y_min - offset.1).get().to_le_bytes());
                out.extend_from_slice(&(bounds.y_max - offset.1).get().to_le_bytes());
                if !encode_commands(inner, out, depth + 1, total_cmds, offset, filter_atlas) {
                    return false;
                }
            }
            Command::RenderBitmap {
                bitmap,
                transform,
                smoothing,
                pixel_snapping,
                region,
            } => {
                let ptr = Arc::as_ptr(&bitmap.0) as *const () as usize;
                if let Some(fa) = filter_atlas
                    && fa.slots.contains_key(&ptr)
                {
                    // Upstream CAB lives in the transient per-frame filter atlas: if it has
                    // a canonical content key, inline-encode its exact key so nested filtered
                    // CABs and blend layers remain 100% cacheable.
                    let Some(child_key) = fa.slot_keys.get(&ptr) else {
                        crate::perf_counters::count_named("cab-nokey/atlas-bmp");
                        return false;
                    };
                    out.push(12);
                    out.extend_from_slice(&child_key.used_size.0.to_le_bytes());
                    out.extend_from_slice(&child_key.used_size.1.to_le_bytes());
                    out.push(child_key.quality);
                    out.extend_from_slice(&child_key.clear);
                    out.extend_from_slice(&(child_key.payload.len() as u32).to_le_bytes());
                    out.extend_from_slice(&child_key.payload);
                } else {
                    let tex = as_texture(bitmap);
                    out.push(11);
                    out.extend_from_slice(&tex.id.to_le_bytes());
                    out.extend_from_slice(&tex.generation.get().to_le_bytes());
                }
                out.push(u8::from(*smoothing));
                out.push(match pixel_snapping {
                    PixelSnapping::Never => 0,
                    PixelSnapping::Always => 1,
                    PixelSnapping::Auto => 2,
                });
                out.extend_from_slice(&region.x_min.to_le_bytes());
                out.extend_from_slice(&region.y_min.to_le_bytes());
                out.extend_from_slice(&region.x_max.to_le_bytes());
                out.extend_from_slice(&region.y_max.to_le_bytes());
                if !encode_transform(transform, offset, out) {
                    crate::perf_counters::count_named("cab-nokey/3d");
                    return false;
                }
            }
            // Never cache commands referencing Stage3D, alpha masks, or shader blends.
            Command::RenderStage3D { .. }
            | Command::RenderAlphaMask { .. }
            | Command::Blend(_, RenderBlendMode::Shader(_))
            | Command::BlendBounded(_, RenderBlendMode::Shader(_), _) => {
                crate::perf_counters::count_named("cab-nokey/other");
                return false;
            }
        }
    }
    true
}

/// Page dimension (width and height) for persistent small-texture atlas pages (16 MB/page).
const ATLAS_PAGE_SIZE: u32 = 2048;
/// Maximum persistent atlas pages for small cached blend layers (16 × 16 MB = 256 MB).
const BLEND_ATLAS_MAX_PAGES: usize = 16;
/// Maximum persistent atlas pages for small cached filtered CABs (24 × 16 MB = 384 MB).
const CAB_ATLAS_MAX_PAGES: usize = 24;
/// Maximum width or height (in pixels) for nearest-sampled atlas packing (+1 px gutter <= 256).
const ATLAS_MAX_NEAREST_DIM: u32 = 255;
/// Maximum width or height (in pixels) for linear-sampled atlas packing (+2 px 1-on-each-side border <= 256).
const ATLAS_MAX_LINEAR_DIM: u32 = 254;
/// Size of the lazy zero-initialized texture used to clear reused linear atlas slots.
const ATLAS_ZERO_TEX_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AtlasSlot {
    page_idx: usize,
    shelf_idx: usize,
    x: u32,
    alloc_w: u32,
}

#[derive(Debug)]
struct AtlasShelf {
    y: u32,
    height: u32,
    /// Sorted, non-overlapping `(x, width)` free horizontal spans inside this shelf.
    free_spans: Vec<(u32, u32)>,
    /// True once any slot in this shelf has been freed (so linear-filtered allocations must
    /// zero the 1 px border before writing new pixels).
    reused: bool,
}

impl AtlasShelf {
    fn new(y: u32, height: u32, page_width: u32) -> Self {
        Self {
            y,
            height,
            free_spans: vec![(0, page_width)],
            reused: false,
        }
    }

    fn alloc(&mut self, width: u32) -> Option<u32> {
        for idx in 0..self.free_spans.len() {
            let (sx, sw) = self.free_spans[idx];
            if sw >= width {
                if sw == width {
                    self.free_spans.remove(idx);
                } else {
                    self.free_spans[idx] = (sx + width, sw - width);
                }
                return Some(sx);
            }
        }
        None
    }

    fn free(&mut self, x: u32, width: u32) {
        self.reused = true;
        let pos = self
            .free_spans
            .binary_search_by_key(&x, |&(sx, _)| sx)
            .unwrap_or_else(|e| e);
        self.free_spans.insert(pos, (x, width));
        // Coalesce with next span if contiguous.
        if pos + 1 < self.free_spans.len() {
            let (ax, aw) = self.free_spans[pos];
            let (bx, bw) = self.free_spans[pos + 1];
            if ax + aw == bx {
                self.free_spans[pos] = (ax, aw + bw);
                self.free_spans.remove(pos + 1);
            }
        }
        // Coalesce with previous span if contiguous.
        if pos > 0 {
            let (px, pw) = self.free_spans[pos - 1];
            let (ax, aw) = self.free_spans[pos];
            if px + pw == ax {
                self.free_spans[pos - 1] = (px, pw + aw);
                self.free_spans.remove(pos);
            }
        }
    }
}

#[inline]
fn atlas_bucket_height(padded_h: u32) -> u32 {
    match padded_h {
        0..=16 => 16,
        17..=32 => 32,
        33..=48 => 48,
        49..=64 => 64,
        65..=96 => 96,
        97..=128 => 128,
        129..=192 => 192,
        _ => 256,
    }
}

struct PersistentAtlasPage {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    shelves: Vec<AtlasShelf>,
    next_y: u32,
    live_slots: usize,
    /// True once this page has been reset after emptying, meaning prior texels may be non-zero.
    reused: bool,
}

impl PersistentAtlasPage {
    fn new(descriptors: &Descriptors, linear: bool) -> Self {
        let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
            label: create_debug_label!("Persistent cache atlas (linear: {})", linear).as_deref(),
            size: wgpu::Extent3d {
                width: ATLAS_PAGE_SIZE,
                height: ATLAS_PAGE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = descriptors.bitmap_samplers.get_sampler(false, linear);
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &descriptors.bind_layouts.bitmap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
                label: create_debug_label!("Persistent cache atlas bind group (linear: {})", linear)
                    .as_deref(),
            });
        Self {
            texture,
            view,
            bind_group,
            shelves: Vec::new(),
            next_y: 0,
            live_slots: 0,
            reused: false,
        }
    }
}

/// Persistent multi-page shelf atlas packing small cached blend layers (nearest) or filtered
/// CABs (linear, with a 1 px transparent border on all 4 sides) onto shared `2048×2048`
/// textures so dozens of cached textures draw with a single `wgpu::BindGroup`.
struct PersistentCacheAtlas {
    pages: Vec<PersistentAtlasPage>,
    deferred_frees: Vec<AtlasSlot>,
    linear: bool,
    max_pages: usize,
    /// Lazily created `256×256` zero-initialized texture used to clear reused linear slots
    /// inside the command encoder so linear sampling can never bleed stale neighbor pixels.
    zero_texture: Option<wgpu::Texture>,
}

impl PersistentCacheAtlas {
    fn new(linear: bool, max_pages: usize) -> Self {
        Self {
            pages: Vec::new(),
            deferred_frees: Vec::new(),
            linear,
            max_pages,
            zero_texture: None,
        }
    }

    fn clear(&mut self) {
        self.pages.clear();
        self.deferred_frees.clear();
    }

    fn flush_deferred_frees(&mut self) {
        for slot in self.deferred_frees.drain(..) {
            if let Some(page) = self.pages.get_mut(slot.page_idx) {
                if let Some(shelf) = page.shelves.get_mut(slot.shelf_idx) {
                    shelf.free(slot.x, slot.alloc_w);
                }
                page.live_slots = page.live_slots.saturating_sub(1);
                if page.live_slots == 0 {
                    page.shelves.clear();
                    page.next_y = 0;
                    page.reused = true;
                }
            }
        }
    }

    fn clear_reused_slot(
        zero_texture: &mut Option<wgpu::Texture>,
        descriptors: &Descriptors,
        encoder: &mut wgpu::CommandEncoder,
        dst_tex: &wgpu::Texture,
        clear_origin: (u32, u32),
        clear_size: (u32, u32),
    ) {
        let zero = zero_texture.get_or_insert_with(|| {
            descriptors.device.create_texture(&wgpu::TextureDescriptor {
                label: create_debug_label!("Persistent cache atlas zero clear source").as_deref(),
                size: wgpu::Extent3d {
                    width: ATLAS_ZERO_TEX_SIZE,
                    height: ATLAS_ZERO_TEX_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        });
        encoder.copy_texture_to_texture(
            zero.as_image_copy(),
            wgpu::TexelCopyTextureInfo {
                texture: dst_tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: clear_origin.0,
                    y: clear_origin.1,
                    z: 0,
                },
                aspect: Default::default(),
            },
            wgpu::Extent3d {
                width: clear_size.0,
                height: clear_size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn allocate(
        &mut self,
        descriptors: &Descriptors,
        mut clear_encoder: Option<&mut wgpu::CommandEncoder>,
        used_size: (u32, u32),
    ) -> Option<CacheAllocation> {
        let (w, h) = used_size;
        let max_dim = if self.linear {
            ATLAS_MAX_LINEAR_DIM
        } else {
            ATLAS_MAX_NEAREST_DIM
        };
        if w == 0 || h == 0 || w > max_dim || h > max_dim {
            return None;
        }
        let border = u32::from(self.linear);
        let pad = if self.linear { 2 } else { 1 };
        let alloc_w = w + pad;
        let alloc_h = h + pad;
        let bucket_h = atlas_bucket_height(alloc_h);

        // 1. Try existing shelves with matching bucket height.
        for (page_idx, page) in self.pages.iter_mut().enumerate() {
            for (shelf_idx, shelf) in page.shelves.iter_mut().enumerate() {
                if shelf.height == bucket_h
                    && let Some(x) = shelf.alloc(alloc_w)
                {
                    let y = shelf.y;
                    let needs_clear = self.linear && (page.reused || shelf.reused);
                    page.live_slots += 1;
                    if needs_clear && let Some(enc) = clear_encoder.as_deref_mut() {
                        Self::clear_reused_slot(
                            &mut self.zero_texture,
                            descriptors,
                            enc,
                            &page.texture,
                            (x, y),
                            (alloc_w, alloc_h),
                        );
                    }
                    return Some(Self::build_alloc(
                        page,
                        AtlasSlot {
                            page_idx,
                            shelf_idx,
                            x,
                            alloc_w,
                        },
                        (x + border, y + border),
                        used_size,
                    ));
                }
            }
        }

        // 2. Open a new shelf in an existing page with vertical room.
        for (page_idx, page) in self.pages.iter_mut().enumerate() {
            if page.next_y + bucket_h <= ATLAS_PAGE_SIZE {
                let y = page.next_y;
                page.next_y += bucket_h;
                let mut shelf = AtlasShelf::new(y, bucket_h, ATLAS_PAGE_SIZE);
                let x = shelf.alloc(alloc_w).expect("fresh shelf fits <= 256 px");
                let shelf_idx = page.shelves.len();
                page.shelves.push(shelf);
                let needs_clear = self.linear && page.reused;
                page.live_slots += 1;
                if needs_clear && let Some(enc) = clear_encoder.as_deref_mut() {
                    Self::clear_reused_slot(
                        &mut self.zero_texture,
                        descriptors,
                        enc,
                        &page.texture,
                        (x, y),
                        (alloc_w, alloc_h),
                    );
                }
                return Some(Self::build_alloc(
                    page,
                    AtlasSlot {
                        page_idx,
                        shelf_idx,
                        x,
                        alloc_w,
                    },
                    (x + border, y + border),
                    used_size,
                ));
            }
        }

        // 3. Allocate a new 2048×2048 atlas page if under page cap.
        if self.pages.len() >= self.max_pages {
            return None;
        }
        let page_idx = self.pages.len();
        self.pages
            .push(PersistentAtlasPage::new(descriptors, self.linear));
        let page = &mut self.pages[page_idx];
        let y = 0;
        page.next_y = bucket_h;
        let mut shelf = AtlasShelf::new(y, bucket_h, ATLAS_PAGE_SIZE);
        let x = shelf.alloc(alloc_w).expect("fresh shelf fits <= 256 px");
        page.shelves.push(shelf);
        page.live_slots += 1;
        Some(Self::build_alloc(
            page,
            AtlasSlot {
                page_idx,
                shelf_idx: 0,
                x,
                alloc_w,
            },
            (x + border, y + border),
            used_size,
        ))
    }

    fn build_alloc(
        page: &PersistentAtlasPage,
        slot: AtlasSlot,
        dst_origin: (u32, u32),
        used_size: (u32, u32),
    ) -> CacheAllocation {
        let inv = 1.0 / ATLAS_PAGE_SIZE as f32;
        let u0 = dst_origin.0 as f32 * inv;
        let v0 = dst_origin.1 as f32 * inv;
        let u1 = (dst_origin.0 + used_size.0) as f32 * inv;
        let v1 = (dst_origin.1 + used_size.1) as f32 * inv;
        CacheAllocation {
            texture: page.texture.clone(),
            view: page.view.clone(),
            bind_group: page.bind_group.clone(),
            dst_origin,
            atlas_size: (ATLAS_PAGE_SIZE, ATLAS_PAGE_SIZE),
            used_size,
            uv_rect: Some([u0, v0, u1, v1]),
            atlas_slot: Some(slot),
        }
    }
}

/// A newly allocated storage target (either a sub-rect inside a persistent atlas page or a
/// standalone texture) waiting to be filled and inserted into a cache.
pub struct CacheAllocation {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    pub dst_origin: (u32, u32),
    pub atlas_size: (u32, u32),
    pub used_size: (u32, u32),
    pub uv_rect: Option<[f32; 4]>,
    atlas_slot: Option<AtlasSlot>,
}

pub type BlendCacheAllocation = CacheAllocation;

pub struct CachedCabTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    pub src_origin: (u32, u32),
    pub atlas_size: (u32, u32),
    pub uv_rect: Option<[f32; 4]>,
    atlas_slot: Option<AtlasSlot>,
    bytes: usize,
    last_used_frame: u64,
}

struct CabTier {
    entries: LruCache<CabContentKey, CachedCabTexture>,
    total_bytes: usize,
    max_bytes: usize,
}

impl CabTier {
    fn new(max_bytes: usize) -> Self {
        Self {
            entries: LruCache::unbounded(),
            total_bytes: 0,
            max_bytes,
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
    }

    fn expire_stale(
        &mut self,
        frame: u64,
        seen: &mut LruCache<u64, ()>,
        mut deferred_frees: Option<&mut Vec<AtlasSlot>>,
    ) {
        for _ in 0..8 {
            let expired = self.entries.peek_lru().is_some_and(|(_, entry)| {
                frame.saturating_sub(entry.last_used_frame) > MAX_IDLE_FRAMES
            });
            if expired {
                if let Some((evicted_key, evicted)) = self.entries.pop_lru() {
                    seen.pop(&evicted_key.fast_hash());
                    self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes);
                    if let (Some(slot), Some(frees)) =
                        (evicted.atlas_slot, deferred_frees.as_deref_mut())
                    {
                        frees.push(slot);
                    }
                }
            } else {
                break;
            }
        }
    }

    fn get(&mut self, key: &CabContentKey, frame: u64) -> Option<&CachedCabTexture> {
        let entry = self.entries.get_mut(key)?;
        entry.last_used_frame = frame;
        Some(entry)
    }

    #[inline]
    fn is_under_half(&self) -> bool {
        self.total_bytes * 2 < self.max_bytes
    }

    fn can_promote(&self, bytes: usize, frame: u64) -> bool {
        if bytes > self.max_bytes {
            return false;
        }
        if self.total_bytes + bytes <= self.max_bytes {
            return true;
        }
        self.entries.peek_lru().is_some_and(|(_, oldest)| {
            frame.saturating_sub(oldest.last_used_frame) > HOT_WINDOW_FRAMES
        })
    }

    fn insert_alloc(
        &mut self,
        key: CabContentKey,
        alloc: CacheAllocation,
        frame: u64,
        seen: &mut LruCache<u64, ()>,
        mut deferred_frees: Option<&mut Vec<AtlasSlot>>,
    ) {
        let bytes = alloc.used_size.0 as usize * alloc.used_size.1 as usize * 4;
        if bytes > self.max_bytes {
            if let (Some(slot), Some(frees)) = (alloc.atlas_slot, deferred_frees) {
                frees.push(slot);
            }
            return;
        }
        while self.total_bytes + bytes > self.max_bytes {
            let can_evict = self.entries.peek_lru().is_some_and(|(_, oldest)| {
                frame.saturating_sub(oldest.last_used_frame) > HOT_WINDOW_FRAMES
            });
            if !can_evict {
                if let (Some(slot), Some(frees)) = (alloc.atlas_slot, deferred_frees) {
                    frees.push(slot);
                }
                return;
            }
            let Some((evicted_key, evicted)) = self.entries.pop_lru() else {
                break;
            };
            seen.pop(&evicted_key.fast_hash());
            self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes);
            if let (Some(slot), Some(frees)) = (evicted.atlas_slot, deferred_frees.as_deref_mut()) {
                frees.push(slot);
            }
        }
        if let Some(old) = self.entries.put(
            key,
            CachedCabTexture {
                texture: alloc.texture,
                view: alloc.view,
                bind_group: alloc.bind_group,
                src_origin: alloc.dst_origin,
                atlas_size: alloc.atlas_size,
                uv_rect: alloc.uv_rect,
                atlas_slot: alloc.atlas_slot,
                bytes,
                last_used_frame: frame,
            },
        ) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
            if let (Some(slot), Some(frees)) = (old.atlas_slot, deferred_frees) {
                frees.push(slot);
            }
        }
        self.total_bytes += bytes;
    }
}

pub struct FilteredCabCache {
    seen: LruCache<u64, ()>,
    small: CabTier,
    large: CabTier,
    subframe: CabTier,
    atlas: PersistentCacheAtlas,
    frame: u64,
}

impl Default for FilteredCabCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FilteredCabCache {
    pub fn new() -> Self {
        Self {
            seen: LruCache::new(NonZeroUsize::new(SEEN_CAPACITY).unwrap()),
            small: CabTier::new(SMALL_MAX_BYTES),
            large: CabTier::new(LARGE_MAX_BYTES),
            subframe: CabTier::new(SUBFRAME_MAX_BYTES),
            atlas: PersistentCacheAtlas::new(true, CAB_ATLAS_MAX_PAGES),
            frame: 0,
        }
    }

    #[inline]
    fn is_small(used_size: (u32, u32)) -> bool {
        u64::from(used_size.0) * u64::from(used_size.1) <= SMALL_MAX_PIXELS
    }

    pub fn clear(&mut self) {
        self.seen.clear();
        self.small.clear();
        self.large.clear();
        self.subframe.clear();
        self.atlas.clear();
    }

    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.atlas.flush_deferred_frees();
        // Evict up to 8 stale entries per tier per frame so room-hopping frees VRAM smoothly.
        self.small.expire_stale(
            self.frame,
            &mut self.seen,
            Some(&mut self.atlas.deferred_frees),
        );
        self.large.expire_stale(self.frame, &mut self.seen, None);
        self.subframe.expire_stale(
            self.frame,
            &mut self.seen,
            Some(&mut self.atlas.deferred_frees),
        );
    }

    pub fn get(&mut self, key: &CabContentKey) -> Option<&CachedCabTexture> {
        let frame = self.frame;
        if Self::is_small(key.used_size) {
            if let Some(hit) = self.small.get(key, frame) {
                return Some(hit);
            }
            self.subframe.get(key, frame)
        } else {
            self.large.get(key, frame)
        }
    }

    /// Returns whether there is room in the entry's tier to promote `used_size`
    /// without evicting any currently-hot looping entry (used within `HOT_WINDOW_FRAMES`).
    pub fn can_promote(&self, used_size: (u32, u32), is_subframe: bool) -> bool {
        let bytes = used_size.0 as usize * used_size.1 as usize * 4;
        if is_subframe {
            Self::is_small(used_size) && self.subframe.can_promote(bytes, self.frame)
        } else if Self::is_small(used_size) {
            self.small.can_promote(bytes, self.frame)
        } else {
            self.large.can_promote(bytes, self.frame)
        }
    }

    fn tier_under_half(&self, used_size: (u32, u32), is_subframe: bool) -> bool {
        if is_subframe {
            Self::is_small(used_size) && self.subframe.is_under_half()
        } else if Self::is_small(used_size) {
            self.small.is_under_half()
        } else {
            self.large.is_under_half()
        }
    }

    /// Adaptive admission filter: promotes on first sight while the target tier is <50% full
    /// (cutting 144 fps warmup time in half), and tightens to 2-hit admission once busy so
    /// one-off particles never displace looping animations.
    pub fn should_promote(
        &mut self,
        key: &CabContentKey,
        used_size: (u32, u32),
        is_subframe: bool,
    ) -> bool {
        let h = key.fast_hash();
        let already_seen = self.seen.get(&h).is_some();
        if !already_seen {
            self.seen.put(h, ());
        }
        if already_seen || self.tier_under_half(used_size, is_subframe) {
            let ok = self.can_promote(used_size, is_subframe);
            if !ok {
                let counter = if is_subframe {
                    "cab-miss/subframe-full"
                } else if Self::is_small(used_size) {
                    "cab-miss/small-full"
                } else {
                    "cab-miss/large-full"
                };
                crate::perf_counters::count_named(counter);
            }
            ok
        } else {
            crate::perf_counters::count_named("cab-miss/1st-seen");
            false
        }
    }

    /// Allocates storage for a promoted filtered CAB: small (`<= 254×254`) entries pack into
    /// shared `2048×2048` linear-sampled atlas pages with a 1 px transparent border on all 4
    /// sides; larger entries (or overflow) allocate standalone textures.
    pub fn allocate_entry(
        &mut self,
        descriptors: &Descriptors,
        encoder: &mut wgpu::CommandEncoder,
        used_size: (u32, u32),
    ) -> CacheAllocation {
        if let Some(alloc) = self.atlas.allocate(descriptors, Some(encoder), used_size) {
            return alloc;
        }
        let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
            label: create_debug_label!("Cached filtered CAB").as_deref(),
            size: wgpu::Extent3d {
                width: used_size.0,
                height: used_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let linear = descriptors.bitmap_samplers.get_sampler(false, true);
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &descriptors.bind_layouts.bitmap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(linear),
                    },
                ],
                label: create_debug_label!("Cached filtered CAB bind group").as_deref(),
            });
        CacheAllocation {
            texture,
            view,
            bind_group,
            dst_origin: (0, 0),
            atlas_size: used_size,
            used_size,
            uv_rect: None,
            atlas_slot: None,
        }
    }

    pub fn cancel_allocation(&mut self, alloc: CacheAllocation) {
        if let Some(slot) = alloc.atlas_slot {
            self.atlas.deferred_frees.push(slot);
        }
    }

    pub fn insert_entry(
        &mut self,
        key: CabContentKey,
        alloc: CacheAllocation,
        is_subframe: bool,
    ) {
        if is_subframe {
            if Self::is_small(alloc.used_size) {
                self.subframe.insert_alloc(
                    key,
                    alloc,
                    self.frame,
                    &mut self.seen,
                    Some(&mut self.atlas.deferred_frees),
                );
            } else {
                self.cancel_allocation(alloc);
            }
        } else if Self::is_small(alloc.used_size) {
            self.small.insert_alloc(
                key,
                alloc,
                self.frame,
                &mut self.seen,
                Some(&mut self.atlas.deferred_frees),
            );
        } else {
            self.large
                .insert_alloc(key, alloc, self.frame, &mut self.seen, None);
        }
    }
}

/// Content-addressed LRU cache for offscreen blend layers (`Multiply`, `Overlay`,
/// `HardLight`, `Add`, `Screen`, etc.).
///
/// With 6× frame interpolation, 24 fps animated blend layers emit identical commands on
/// all 6 subframes (and repeat every 1–2 s loop). Caching their resolved 4× MSAA offscreen
/// texture by exact canonical inputs (normalized to layer-local `(0, 0)`) eliminates
/// offscreen rasterization and atlas/target passes completely on hits, while packing small
/// (`<= 255×255`) cached layers onto shared `2048×2048` atlas pages to minimize bind-group
/// switches in the main pass.
pub struct BlendLayerCache {
    seen: LruCache<u64, ()>,
    small: CabTier,
    large: CabTier,
    atlas: PersistentCacheAtlas,
    frame: u64,
}

impl Default for BlendLayerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BlendLayerCache {
    pub fn new() -> Self {
        Self {
            seen: LruCache::new(NonZeroUsize::new(SEEN_CAPACITY).unwrap()),
            small: CabTier::new(BLEND_SMALL_MAX_BYTES),
            large: CabTier::new(BLEND_LARGE_MAX_BYTES),
            atlas: PersistentCacheAtlas::new(false, BLEND_ATLAS_MAX_PAGES),
            frame: 0,
        }
    }

    #[inline]
    fn is_small(used_size: (u32, u32)) -> bool {
        u64::from(used_size.0) * u64::from(used_size.1) <= BLEND_SMALL_MAX_PIXELS
    }

    pub fn clear(&mut self) {
        self.seen.clear();
        self.small.clear();
        self.large.clear();
        self.atlas.clear();
    }

    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        // Drain frees deferred from prior frames before allocating any new atlas slots.
        self.atlas.flush_deferred_frees();
        self.small.expire_stale(
            self.frame,
            &mut self.seen,
            Some(&mut self.atlas.deferred_frees),
        );
        self.large.expire_stale(self.frame, &mut self.seen, None);
    }

    pub fn get(&mut self, key: &CabContentKey) -> Option<&CachedCabTexture> {
        let frame = self.frame;
        if Self::is_small(key.used_size) {
            self.small.get(key, frame)
        } else {
            self.large.get(key, frame)
        }
    }

    pub fn should_promote(&mut self, key: &CabContentKey, used_size: (u32, u32)) -> bool {
        let h = key.fast_hash();
        let already_seen = self.seen.get(&h).is_some();
        if !already_seen {
            self.seen.put(h, ());
        }
        let under_half = if Self::is_small(used_size) {
            self.small.is_under_half()
        } else {
            self.large.is_under_half()
        };
        if already_seen || under_half {
            let bytes = used_size.0 as usize * used_size.1 as usize * 4;
            let ok = if Self::is_small(used_size) {
                self.small.can_promote(bytes, self.frame)
            } else {
                self.large.can_promote(bytes, self.frame)
            };
            if !ok {
                crate::perf_counters::count_named(if Self::is_small(used_size) {
                    "blend-lru/small-full"
                } else {
                    "blend-lru/large-full"
                });
            }
            ok
        } else {
            false
        }
    }

    /// Allocates storage for a promoted blend layer: small (`<= 255×255`) layers pack into
    /// shared `2048×2048` atlas pages so consecutive hits reuse one bind group; larger layers
    /// (or overflow) allocate standalone textures.
    pub fn allocate_entry(
        &mut self,
        descriptors: &Descriptors,
        used_size: (u32, u32),
    ) -> BlendCacheAllocation {
        if let Some(alloc) = self.atlas.allocate(descriptors, None, used_size) {
            return alloc;
        }
        let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
            label: create_debug_label!("Cached blend layer").as_deref(),
            size: wgpu::Extent3d {
                width: used_size.0,
                height: used_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        // Matches `WgpuCommandHandler::texture_bind_group` (exact pixel-aligned nearest sampling).
        let nearest = descriptors.bitmap_samplers.get_sampler(false, false);
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &descriptors.bind_layouts.bitmap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(nearest),
                    },
                ],
                label: create_debug_label!("Cached blend layer bind group").as_deref(),
            });
        BlendCacheAllocation {
            texture,
            view,
            bind_group,
            dst_origin: (0, 0),
            atlas_size: used_size,
            used_size,
            uv_rect: None,
            atlas_slot: None,
        }
    }

    pub fn insert_entry(&mut self, key: CabContentKey, alloc: BlendCacheAllocation) {
        if Self::is_small(alloc.used_size) {
            self.small.insert_alloc(
                key,
                alloc,
                self.frame,
                &mut self.seen,
                Some(&mut self.atlas.deferred_frees),
            );
        } else {
            self.large
                .insert_alloc(key, alloc, self.frame, &mut self.seen, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shelf_alloc_free_and_coalesce() {
        let mut shelf = AtlasShelf::new(0, 64, 256);
        let a = shelf.alloc(100).unwrap();
        let b = shelf.alloc(80).unwrap();
        let c = shelf.alloc(76).unwrap();
        assert_eq!((a, b, c), (0, 100, 180));
        assert!(shelf.alloc(1).is_none());

        // Free left and right, then middle: should coalesce back into one 256-px span.
        shelf.free(a, 100);
        shelf.free(c, 76);
        assert_eq!(shelf.free_spans, vec![(0, 100), (180, 76)]);
        shelf.free(b, 80);
        assert_eq!(shelf.free_spans, vec![(0, 256)]);
        assert_eq!(shelf.alloc(256), Some(0));
    }

    #[test]
    fn bucket_heights_cover_padded_small_entries() {
        assert_eq!(atlas_bucket_height(1), 16);
        assert_eq!(atlas_bucket_height(16), 16);
        assert_eq!(atlas_bucket_height(17), 32);
        assert_eq!(atlas_bucket_height(64), 64);
        assert_eq!(atlas_bucket_height(65), 96);
        assert_eq!(atlas_bucket_height(256), 256);
    }
}
