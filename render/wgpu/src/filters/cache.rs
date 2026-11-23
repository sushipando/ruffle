//! Content-addressed LRU cache for looping pure-vector filtered CABs.
//!
//! Animated capes, wings, weapons, and pulsing auras in Flash/AQW frequently repeat short
//! 10-60 frame vector animations with identical shapes, transforms, color transforms, and
//! filters. Caching their finished filtered textures by exact canonical inputs lets looping
//! CABs skip 4x MSAA rasterization and blur/glow filtering completely after their first loop,
//! while strictly guarding against stale bitmaps, sub-pixel mismatches, pointer ABA reuse,
//! hash collisions, and unbounded VRAM growth.

use crate::descriptors::Descriptors;
use crate::mesh::mesh_id;
use lru::LruCache;
use ruffle_render::commands::{Command, CommandList, RenderBlendMode};
use ruffle_render::filters::Filter;
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use swf::{Color, Twips};

/// Maximum VRAM retained by the small/medium keyframe tier (<= 384x384 px = <= 576 KB/frame).
/// Isolated from both large entries and interpolated subframes so giant wings or 6x-volume
/// subframe animations can never starve dozens of smaller glowing weapons, flames, and runes.
const SMALL_MAX_BYTES: usize = 256 * 1024 * 1024;
/// Pixel area threshold separating the small/medium tier from the large tier (384x384).
const SMALL_MAX_PIXELS: u64 = 384 * 384;
/// Maximum VRAM retained by the large keyframe tier (384x384 < pixels <= 1024x1024).
const LARGE_MAX_BYTES: usize = 256 * 1024 * 1024;
/// Dedicated VRAM tier for interpolated subframe CABs (<= 384x384 px), isolated from the
/// keyframe tiers so 5x-per-tick subframe volume never evicts or blocks keyframe promotions.
const SUBFRAME_MAX_BYTES: usize = 64 * 1024 * 1024;
/// Maximum single CAB pixel area eligible for caching (1024x1024 = 4 MB).
const MAX_ENTRY_PIXELS: u64 = 1024 * 1024;
/// Maximum VRAM retained by the small/medium blend-layer tier (<= 512x512 px).
const BLEND_SMALL_MAX_BYTES: usize = 192 * 1024 * 1024;
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
const MAX_COMMANDS: usize = 256;
/// Evict cached textures unused for this many submitted frames (~62 s at 144 fps, ~125 s at
/// 72 fps) so player attack/skill animations on 15-45 s cooldowns stay warm in-room while
/// rooms left behind still free VRAM smoothly.
const MAX_IDLE_FRAMES: u64 = 9000;
/// Refuse to evict a cached entry used within this many frames (~5 s at 144 fps) so active
/// loops stay pinned at 100% hit rate instead of thrashing when a room exceeds the tier budget.
const HOT_WINDOW_FRAMES: u64 = 720;
/// Size of the 2-hit admission filter (only promote keys seen at least twice).
const SEEN_CAPACITY: usize = 16384;

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
    /// Returns `None` if the CAB is too large/complex or contains any command that could
    /// reference mutable pixels (`RenderBitmap`, `RenderStage3D`, `RenderAlphaMask`, or
    /// `BlendMode::Shader`).
    pub fn try_from_entry(
        used_size: (u32, u32),
        clear: Color,
        quality: StageQuality,
        filters: &[Filter],
        commands: &CommandList,
    ) -> Option<Self> {
        if filters.is_empty() {
            return None;
        }
        Self::try_from_entry_inner(used_size, clear, quality, filters, commands)
    }

    /// Same as `try_from_entry`, but also permits unfiltered (`filters.is_empty()`) entries
    /// for zero-VRAM same-handle skip checks on `Persistent` CABs.
    pub fn try_from_any_entry(
        used_size: (u32, u32),
        clear: Color,
        quality: StageQuality,
        filters: &[Filter],
        commands: &CommandList,
    ) -> Option<Self> {
        Self::try_from_entry_inner(used_size, clear, quality, filters, commands)
    }

    /// Serializes a pure-vector offscreen blend layer's exact rendering inputs into a
    /// canonical key, normalized to layer-local `(0, 0)` coordinates by subtracting `origin`.
    pub fn try_from_blend_layer(
        used_size: (u32, u32),
        origin: (i32, i32),
        clear: Color,
        quality: StageQuality,
        is_layer: bool,
        commands: &CommandList,
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
        if !encode_commands(commands, &mut payload, 0, &mut total_cmds, offset) {
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
    ) -> Option<Self> {
        let pixels = u64::from(used_size.0) * u64::from(used_size.1);
        if pixels == 0 || pixels > MAX_ENTRY_PIXELS {
            return None;
        }
        if commands.commands.is_empty() || commands.commands.len() > MAX_COMMANDS {
            return None;
        }
        let mut payload = Vec::with_capacity(64 + commands.commands.len() * 48);
        payload.extend_from_slice(&(filters.len() as u32).to_le_bytes());
        for filter in filters {
            if !encode_filter(filter, &mut payload) {
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
) -> bool {
    if depth > 4 {
        return false;
    }
    out.extend_from_slice(&(commands.commands.len() as u32).to_le_bytes());
    for cmd in &commands.commands {
        *total_cmds += 1;
        if *total_cmds > MAX_COMMANDS {
            return false;
        }
        match cmd {
            Command::RenderShape { shape, transform } => {
                out.push(1);
                out.extend_from_slice(&mesh_id(shape).to_le_bytes());
                if !encode_transform(transform, offset, out) {
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
                if !encode_commands(inner, out, depth + 1, total_cmds, offset) {
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
                if !encode_commands(inner, out, depth + 1, total_cmds, offset) {
                    return false;
                }
            }
            // Never cache commands referencing mutable bitmaps, video, alpha masks, or
            // shader blends.
            Command::RenderBitmap { .. }
            | Command::RenderStage3D { .. }
            | Command::RenderAlphaMask { .. }
            | Command::Blend(_, RenderBlendMode::Shader(_))
            | Command::BlendBounded(_, RenderBlendMode::Shader(_), _) => return false,
        }
    }
    true
}

pub struct CachedCabTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    pub used_size: (u32, u32),
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

    fn expire_stale(&mut self, frame: u64, seen: &mut LruCache<u64, ()>) {
        for _ in 0..8 {
            let expired = self.entries.peek_lru().is_some_and(|(_, entry)| {
                frame.saturating_sub(entry.last_used_frame) > MAX_IDLE_FRAMES
            });
            if expired {
                if let Some((evicted_key, evicted)) = self.entries.pop_lru() {
                    seen.pop(&evicted_key.fast_hash());
                    self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes);
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

    fn insert(
        &mut self,
        key: CabContentKey,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        bind_group: wgpu::BindGroup,
        used_size: (u32, u32),
        frame: u64,
        seen: &mut LruCache<u64, ()>,
    ) {
        let bytes = used_size.0 as usize * used_size.1 as usize * 4;
        if bytes > self.max_bytes {
            return;
        }
        while self.total_bytes + bytes > self.max_bytes {
            let can_evict = self.entries.peek_lru().is_some_and(|(_, oldest)| {
                frame.saturating_sub(oldest.last_used_frame) > HOT_WINDOW_FRAMES
            });
            if !can_evict {
                return;
            }
            let Some((evicted_key, evicted)) = self.entries.pop_lru() else {
                break;
            };
            seen.pop(&evicted_key.fast_hash());
            self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes);
        }
        if let Some(old) = self.entries.put(
            key,
            CachedCabTexture {
                texture,
                view,
                bind_group,
                used_size,
                bytes,
                last_used_frame: frame,
            },
        ) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
        self.total_bytes += bytes;
    }
}

pub struct FilteredCabCache {
    seen: LruCache<u64, ()>,
    small: CabTier,
    large: CabTier,
    subframe: CabTier,
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
    }

    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        // Evict up to 8 stale entries per tier per frame so room-hopping frees VRAM smoothly.
        self.small.expire_stale(self.frame, &mut self.seen);
        self.large.expire_stale(self.frame, &mut self.seen);
        self.subframe.expire_stale(self.frame, &mut self.seen);
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

    /// 2-hit admission filter: returns `true` if this key's hash was already seen before
    /// AND the entry's tier has cold space to promote it without thrashing hot entries.
    pub fn should_promote(
        &mut self,
        key: &CabContentKey,
        used_size: (u32, u32),
        is_subframe: bool,
    ) -> bool {
        let h = key.fast_hash();
        if self.seen.get(&h).is_some() {
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
            self.seen.put(h, ());
            false
        }
    }

    pub fn allocate_texture(
        descriptors: &Descriptors,
        used_size: (u32, u32),
    ) -> (wgpu::Texture, wgpu::BindGroup) {
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
        (texture, bind_group)
    }

    pub fn insert(
        &mut self,
        key: CabContentKey,
        texture: wgpu::Texture,
        bind_group: wgpu::BindGroup,
        used_size: (u32, u32),
        is_subframe: bool,
    ) {
        let view = texture.create_view(&Default::default());
        if is_subframe {
            if Self::is_small(used_size) {
                self.subframe.insert(
                    key,
                    texture,
                    view,
                    bind_group,
                    used_size,
                    self.frame,
                    &mut self.seen,
                );
            }
        } else if Self::is_small(used_size) {
            self.small.insert(
                key,
                texture,
                view,
                bind_group,
                used_size,
                self.frame,
                &mut self.seen,
            );
        } else {
            self.large.insert(
                key,
                texture,
                view,
                bind_group,
                used_size,
                self.frame,
                &mut self.seen,
            );
        }
    }
}

/// Content-addressed LRU cache for pure-vector offscreen blend layers (`Multiply`,
/// `Overlay`, `HardLight`, `Add`, `Screen`, etc.).
///
/// With 6× frame interpolation, 24 fps animated blend layers emit identical commands on
/// all 6 subframes (and repeat every 1–2 s loop). Caching their resolved 4× MSAA offscreen
/// texture by exact canonical inputs (normalized to layer-local `(0, 0)`) eliminates
/// offscreen rasterization and atlas/target passes completely on hits, while strictly
/// preserving bit-exact output.
pub struct BlendLayerCache {
    seen: LruCache<u64, ()>,
    small: CabTier,
    large: CabTier,
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
    }

    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.small.expire_stale(self.frame, &mut self.seen);
        self.large.expire_stale(self.frame, &mut self.seen);
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
        if self.seen.get(&h).is_some() {
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
            self.seen.put(h, ());
            false
        }
    }

    pub fn allocate_texture(
        descriptors: &Descriptors,
        used_size: (u32, u32),
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
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
        (texture, view, bind_group)
    }

    pub fn insert(
        &mut self,
        key: CabContentKey,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        bind_group: wgpu::BindGroup,
        used_size: (u32, u32),
    ) {
        if Self::is_small(used_size) {
            self.small.insert(
                key,
                texture,
                view,
                bind_group,
                used_size,
                self.frame,
                &mut self.seen,
            );
        } else {
            self.large.insert(
                key,
                texture,
                view,
                bind_group,
                used_size,
                self.frame,
                &mut self.seen,
            );
        }
    }
}
