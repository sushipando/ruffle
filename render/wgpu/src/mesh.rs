use crate::{Descriptors, GradientUniforms, PosColorVertex, PosUvVertex, as_texture};
use std::any::Any;
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::buffer_builder::BufferBuilder;
use ruffle_render::backend::{RenderBackend, ShapeHandle, ShapeHandleImpl};
use ruffle_render::bitmap::{BitmapHandle, BitmapSize, BitmapSource};
use ruffle_render::shape_utils::{DistilledShape, DrawCommand, DrawPath, FillRule};
use ruffle_render::tessellator::{
    Bitmap, Draw as LyonDraw, DrawType as TessDrawType, Gradient, Mesh as LyonMesh,
    ShapeTessellator,
};
use swf::{CharacterId, FillStyle, GradientInterpolation, LineStyle, Rectangle, Twips};

/// How big to make gradient textures. Larger will keep more detail, but be slower and use more memory.
const GRADIENT_SIZE: usize = 256;

#[derive(Debug)]
pub struct MeshData {
    pub draws: Vec<Draw>,
    /// The vertices and indices of all draws (one buffer, so that creating a shape's
    /// mesh only creates one buffer, unless it has gradients).
    pub buffer: wgpu::Buffer,
    /// The bounds of all vertices, as `(x_min, y_min, x_max, y_max)` in local pixels.
    pub bounds: Option<(f32, f32, f32, f32)>,
}

/// A shape's mesh. With `PerfFlag::ParallelTessellation`, it's pending at first:
/// `WgpuRenderBackend::finish_pending_shapes` tessellates all pending shapes at once, on
/// several threads, before a frame is drawn, and anything that needs a pending mesh
/// earlier tessellates it on the spot (see `Mesh::data`).
pub struct Mesh {
    data: OnceCell<MeshData>,
    pending: RefCell<Option<PendingShape>>,
}

impl ShapeHandleImpl for Mesh {}

impl fmt::Debug for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh")
            .field("data", &self.data.get())
            .field("pending", &self.pending.borrow().is_some())
            .finish()
    }
}

impl Mesh {
    pub fn ready(data: MeshData) -> Self {
        Self {
            data: OnceCell::from(data),
            pending: RefCell::new(None),
        }
    }

    pub fn pending(shape: PendingShape) -> Self {
        Self {
            data: OnceCell::new(),
            pending: RefCell::new(Some(shape)),
        }
    }

    /// The mesh's data, tessellating the shape now if it's still pending.
    pub fn data(&self) -> &MeshData {
        self.data.get_or_init(|| {
            let pending = self
                .pending
                .borrow_mut()
                .take()
                .expect("A mesh is either ready or pending");
            let PendingShape { job, finish } = pending;
            finish.build(job.tessellate(&mut ShapeTessellator::new()))
        })
    }

    /// Takes the shape still to be tessellated, if any (see `set_tessellated`).
    pub(crate) fn take_pending(&self) -> Option<PendingShape> {
        self.pending.borrow_mut().take()
    }

    /// Fills in the mesh of a shape taken with `take_pending`, once it's tessellated.
    pub(crate) fn set_tessellated(&self, finish: FinishShape, mesh: LyonMesh) {
        let _ = self.data.set(finish.build(mesh));
    }
}

/// A shape waiting to be tessellated: what's needed to tessellate it (which can be done
/// on another thread), and to build its mesh from that (on the thread using the backend).
pub struct PendingShape {
    pub(crate) job: TessellationJob,
    pub(crate) finish: FinishShape,
}

impl PendingShape {
    /// Takes what's needed to tessellate `shape` and build its mesh, so that both can be
    /// done later. The handles of the bitmaps it fills with are looked up now, as that may
    /// need the backend (and the bitmap source isn't available later).
    pub fn new(
        descriptors: Arc<Descriptors>,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        renderer: &mut dyn RenderBackend,
        scale: f32,
    ) -> Self {
        let mut bitmap_sizes = HashMap::new();
        let mut bitmaps = HashMap::new();
        for path in &shape.paths {
            let style = match path {
                DrawPath::Fill { style, .. } => *style,
                DrawPath::Stroke { style, .. } => style.fill_style(),
            };
            if let FillStyle::Bitmap { id, .. } = style
                && !bitmap_sizes.contains_key(id)
                && let Some(size) = bitmap_source.bitmap_size(*id)
            {
                bitmap_sizes.insert(*id, size);
                if let Some(handle) = bitmap_source.bitmap_handle(*id, renderer) {
                    bitmaps.insert(*id, handle);
                }
            }
        }
        Self {
            finish: FinishShape {
                descriptors,
                shape_id: shape.id,
                bitmaps,
            },
            job: TessellationJob {
                shape: OwnedShape::from(shape),
                scale,
                bitmap_sizes,
            },
        }
    }
}

/// Tessellates `jobs`, on several threads where possible (with at least two shapes each),
/// returning the meshes in the same order.
pub fn tessellate_all(
    jobs: Vec<TessellationJob>,
    tessellator: &mut ShapeTessellator,
) -> Vec<LyonMesh> {
    #[cfg(not(target_family = "wasm"))]
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Mutex, OnceLock};
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(jobs.len() / 2);
        if threads > 1 {
            let count = jobs.len();
            let jobs: Vec<_> = jobs.into_iter().map(|job| Mutex::new(Some(job))).collect();
            let results: Vec<OnceLock<LyonMesh>> = (0..count).map(|_| OnceLock::new()).collect();
            let next = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        let mut tessellator = ShapeTessellator::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(job) = jobs.get(index) else {
                                break;
                            };
                            let job = job
                                .lock()
                                .expect("Tessellation job lock poisoned")
                                .take()
                                .expect("Each job is taken once");
                            let _ = results[index].set(job.tessellate(&mut tessellator));
                        }
                    });
                }
            });
            return results
                .into_iter()
                .map(|result| result.into_inner().expect("Every job was tessellated"))
                .collect();
        }
    }
    jobs.into_iter()
        .map(|job| job.tessellate(tessellator))
        .collect()
}

/// Tessellating a shape: plain data, so that it can be done on another thread.
pub struct TessellationJob {
    shape: OwnedShape,
    scale: f32,
    bitmap_sizes: HashMap<CharacterId, BitmapSize>,
}

impl TessellationJob {
    pub fn tessellate(self, tessellator: &mut ShapeTessellator) -> LyonMesh {
        let OwnedShape {
            id,
            shape_bounds,
            edge_bounds,
            fills,
            lines,
            paths,
        } = self.shape;
        let paths = paths
            .into_iter()
            .map(|path| match path {
                OwnedPath::Fill {
                    style,
                    commands,
                    winding_rule,
                } => DrawPath::Fill {
                    style: &fills[style],
                    commands,
                    winding_rule,
                },
                OwnedPath::Stroke {
                    style,
                    is_closed,
                    commands,
                } => DrawPath::Stroke {
                    style: &lines[style],
                    is_closed,
                    commands,
                },
            })
            .collect();
        let shape = DistilledShape {
            paths,
            shape_bounds,
            edge_bounds,
            id,
        };
        tessellator.tessellate_shape_with_scale(shape, &BitmapSizes(&self.bitmap_sizes), self.scale)
    }
}

/// Building a tessellated shape's mesh (its buffers and bind groups).
pub struct FinishShape {
    descriptors: Arc<Descriptors>,
    shape_id: CharacterId,
    bitmaps: HashMap<CharacterId, BitmapHandle>,
}

impl FinishShape {
    pub fn build(self, mesh: LyonMesh) -> MeshData {
        build_mesh(&self.descriptors, mesh, self.shape_id, &self.bitmaps)
    }
}

/// A `DistilledShape` that owns its styles.
struct OwnedShape {
    id: CharacterId,
    shape_bounds: Rectangle<Twips>,
    edge_bounds: Rectangle<Twips>,
    fills: Vec<FillStyle>,
    lines: Vec<LineStyle>,
    paths: Vec<OwnedPath>,
}

/// A `DrawPath` whose style is an index into the `OwnedShape`'s styles.
enum OwnedPath {
    Fill {
        style: usize,
        commands: Vec<DrawCommand>,
        winding_rule: FillRule,
    },
    Stroke {
        style: usize,
        is_closed: bool,
        commands: Vec<DrawCommand>,
    },
}

impl From<DistilledShape<'_>> for OwnedShape {
    fn from(shape: DistilledShape<'_>) -> Self {
        let mut fills = Vec::new();
        let mut lines = Vec::new();
        let paths = shape
            .paths
            .into_iter()
            .map(|path| match path {
                DrawPath::Fill {
                    style,
                    commands,
                    winding_rule,
                } => {
                    fills.push(style.clone());
                    OwnedPath::Fill {
                        style: fills.len() - 1,
                        commands,
                        winding_rule,
                    }
                }
                DrawPath::Stroke {
                    style,
                    is_closed,
                    commands,
                } => {
                    lines.push(style.clone());
                    OwnedPath::Stroke {
                        style: lines.len() - 1,
                        is_closed,
                        commands,
                    }
                }
            })
            .collect();
        Self {
            id: shape.id,
            shape_bounds: shape.shape_bounds,
            edge_bounds: shape.edge_bounds,
            fills,
            lines,
            paths,
        }
    }
}

/// The sizes of the bitmaps a shape fills with, which is all the tessellator needs.
struct BitmapSizes<'a>(&'a HashMap<CharacterId, BitmapSize>);

impl BitmapSource for BitmapSizes<'_> {
    fn bitmap_size(&self, id: u16) -> Option<BitmapSize> {
        self.0.get(&id).copied()
    }

    fn bitmap_handle(&self, _id: u16, _renderer: &mut dyn RenderBackend) -> Option<BitmapHandle> {
        None
    }
}

/// Builds a mesh (buffers and bind groups) from a tessellated shape.
pub fn build_mesh(
    descriptors: &Descriptors,
    lyon_mesh: LyonMesh,
    shape_id: CharacterId,
    bitmaps: &HashMap<CharacterId, BitmapHandle>,
) -> MeshData {
    let mut draws = Vec::with_capacity(lyon_mesh.draws.len());
    let mut uniform_buffer = BufferBuilder::new(
        &descriptors.limits,
        descriptors.limits.min_uniform_buffer_offset_alignment,
    );
    // 4-byte aligned: vertex buffer offsets and 32-bit index offsets must be.
    let mut geometry = BufferBuilder::new(&descriptors.limits, 4);
    let mut gradients = Vec::with_capacity(lyon_mesh.gradients.len());

    for gradient in lyon_mesh.gradients {
        gradients.push(CommonGradient::new(
            descriptors,
            gradient,
            &mut uniform_buffer,
        ));
    }

    let mut bounds: Option<(f32, f32, f32, f32)> = None;
    for vertex in lyon_mesh.draws.iter().flat_map(|draw| &draw.vertices) {
        let (x, y) = (vertex.x, vertex.y);
        bounds = Some(match bounds {
            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            None => (x, y, x, y),
        });
    }

    for draw in lyon_mesh.draws {
        let draw_id = draws.len();
        if let Some(draw) =
            PendingDraw::new(descriptors, bitmaps, draw, shape_id, draw_id, &mut geometry)
        {
            draws.push(draw);
        }
    }

    // Only gradients have uniforms.
    let uniform_buffer = (!uniform_buffer.is_empty()).then(|| {
        uniform_buffer.finish(
            &descriptors.device,
            create_debug_label!("Shape {} uniforms", shape_id),
            wgpu::BufferUsages::UNIFORM,
        )
    });
    let buffer = geometry.finish(
        &descriptors.device,
        create_debug_label!("Shape {} geometry", shape_id),
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX,
    );

    let draws = draws
        .into_iter()
        .map(|d| d.finish(descriptors, uniform_buffer.as_ref(), &gradients))
        .collect();

    MeshData::new(draws, buffer, bounds)
}

/// The number of meshes and the size of their buffers (for `RUFFLE_PERF`).
pub static MESH_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static MESH_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl MeshData {
    pub fn new(
        draws: Vec<Draw>,
        buffer: wgpu::Buffer,
        bounds: Option<(f32, f32, f32, f32)>,
    ) -> Self {
        use std::sync::atomic::Ordering;
        MESH_COUNT.fetch_add(1, Ordering::Relaxed);
        MESH_BYTES.fetch_add(buffer.size(), Ordering::Relaxed);
        Self {
            draws,
            buffer,
            bounds,
        }
    }
}

impl Drop for MeshData {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        MESH_COUNT.fetch_sub(1, Ordering::Relaxed);
        MESH_BYTES.fetch_sub(self.buffer.size(), Ordering::Relaxed);
    }
}

pub fn as_mesh(handle: &ShapeHandle) -> &MeshData {
    <dyn Any>::downcast_ref::<Mesh>(&*handle.0)
        .expect("Shape handle must be a WGPU ShapeData")
        .data()
}

#[derive(Debug)]
pub struct PendingDraw {
    pub draw_type: PendingDrawType,
    pub vertices: Range<wgpu::BufferAddress>,
    pub indices: Range<wgpu::BufferAddress>,
    pub num_indices: u32,
    pub num_mask_indices: u32,
    pub single_fill: bool,
}

impl PendingDraw {
    pub fn finish(
        self,
        descriptors: &Descriptors,
        uniform_buffer: Option<&wgpu::Buffer>,
        gradients: &[CommonGradient],
    ) -> Draw {
        Draw {
            draw_type: self
                .draw_type
                .finish(descriptors, uniform_buffer, gradients),
            vertices: self.vertices,
            indices: self.indices,
            num_indices: self.num_indices,
            num_mask_indices: self.num_mask_indices,
            single_fill: self.single_fill,
        }
    }
}

#[derive(Debug)]
pub struct Draw {
    pub draw_type: DrawType,
    pub vertices: Range<wgpu::BufferAddress>,
    pub indices: Range<wgpu::BufferAddress>,
    pub num_indices: u32,
    pub num_mask_indices: u32,
    /// See `ruffle_render::tessellator::Draw::single_fill`.
    pub single_fill: bool,
}

impl PendingDraw {
    pub fn new(
        descriptors: &Descriptors,
        bitmaps: &HashMap<CharacterId, BitmapHandle>,
        draw: LyonDraw,
        shape_id: CharacterId,
        draw_id: usize,
        geometry: &mut BufferBuilder,
    ) -> Option<Self> {
        let vertices = match &draw.draw_type {
            TessDrawType::Color => {
                let vertices: Vec<_> = draw
                    .vertices
                    .into_iter()
                    .map(PosColorVertex::from)
                    .collect();
                geometry
                    .add(&vertices)
                    .expect("Mesh vertex buffer was too large!")
            }
            TessDrawType::Gradient { matrix, .. } => {
                let vertices: Vec<_> = draw
                    .vertices
                    .into_iter()
                    .map(|v| PosUvVertex::from_tessellator(v, matrix))
                    .collect();
                geometry
                    .add(&vertices)
                    .expect("Mesh vertex buffer was too large!")
            }
            TessDrawType::Bitmap(bitmap) => {
                let vertices: Vec<_> = draw
                    .vertices
                    .into_iter()
                    .map(|v| PosUvVertex::from_tessellator(v, &bitmap.matrix))
                    .collect();
                geometry
                    .add(&vertices)
                    .expect("Mesh vertex buffer was too large!")
            }
        };

        let indices = geometry
            .add(&draw.indices)
            .expect("Mesh index buffer was too large!");

        let index_count = draw.indices.len() as u32;
        let draw_type = match draw.draw_type {
            TessDrawType::Color => PendingDrawType::color(),
            TessDrawType::Gradient {
                matrix: _,
                gradient,
            } => PendingDrawType::gradient(gradient, shape_id, draw_id),
            TessDrawType::Bitmap(bitmap) => PendingDrawType::bitmap(bitmap, bitmaps, descriptors)?,
        };
        Some(PendingDraw {
            draw_type,
            vertices,
            indices,
            num_indices: index_count,
            num_mask_indices: draw.mask_index_count,
            single_fill: draw.single_fill,
        })
    }
}

#[derive(Debug)]
pub enum PendingDrawType {
    Color,
    Gradient {
        gradient_index: usize,
        bind_group_label: Option<String>,
    },
    Bitmap {
        binds: BitmapBinds,
    },
}

/// Converts an RGBA color from sRGB space to linear color space.
fn srgb_to_linear(color: f32) -> f32 {
    if color <= 0.04045 {
        color / 12.92
    } else {
        f32::powf((color + 0.055) / 1.055, 2.4)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl PendingDrawType {
    pub fn color() -> Self {
        PendingDrawType::Color
    }

    pub fn gradient(gradient_index: usize, shape_id: CharacterId, draw_id: usize) -> Self {
        let bind_group_label =
            create_debug_label!("Shape {} (gradient) draw {} bindgroup", shape_id, draw_id);
        PendingDrawType::Gradient {
            gradient_index,
            bind_group_label,
        }
    }

    pub fn bitmap(
        bitmap: Bitmap,
        bitmaps: &HashMap<CharacterId, BitmapHandle>,
        descriptors: &Descriptors,
    ) -> Option<Self> {
        let handle = bitmaps.get(&bitmap.bitmap_id)?;
        let texture = as_texture(handle);
        let binds = texture.bind_group(
            bitmap.is_repeating,
            bitmap.is_smoothed,
            &descriptors.device,
            &descriptors.bind_layouts.bitmap,
            handle.clone(),
            &descriptors.bitmap_samplers,
        );

        Some(PendingDrawType::Bitmap {
            binds: binds.clone(),
        })
    }

    pub fn finish(
        self,
        descriptors: &Descriptors,
        uniform_buffer: Option<&wgpu::Buffer>,
        gradients: &[CommonGradient],
    ) -> DrawType {
        match self {
            PendingDrawType::Color => DrawType::Color,
            PendingDrawType::Gradient {
                gradient_index,
                bind_group_label,
            } => {
                let common = &gradients[gradient_index];
                let bind_group = descriptors
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &descriptors.bind_layouts.gradient,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: uniform_buffer
                                        .expect("A shape with gradients has uniforms"),
                                    offset: common.buffer_offset,
                                    size: wgpu::BufferSize::new(
                                        std::mem::size_of::<GradientUniforms>() as u64,
                                    ),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&common.texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(
                                    descriptors.bitmap_samplers.get_sampler(false, true),
                                ),
                            },
                        ],
                        label: bind_group_label.as_deref(),
                    });
                DrawType::Gradient { bind_group }
            }
            PendingDrawType::Bitmap { binds } => DrawType::Bitmap { binds },
        }
    }
}

#[derive(Debug)]
pub enum DrawType {
    Color,
    Gradient { bind_group: wgpu::BindGroup },
    Bitmap { binds: BitmapBinds },
}

#[derive(Debug)]
pub struct CommonGradient {
    texture_view: wgpu::TextureView,
    buffer_offset: wgpu::BufferAddress,
}

impl CommonGradient {
    pub fn new(
        descriptors: &Descriptors,
        gradient: Gradient,
        uniform_buffers: &mut BufferBuilder,
    ) -> Self {
        let colors = if gradient.records.is_empty() {
            [0; GRADIENT_SIZE * 4]
        } else {
            let mut colors = [0; GRADIENT_SIZE * 4];
            let mut last = 0;
            let mut next;

            let convert = if gradient.interpolation == GradientInterpolation::LinearRgb {
                |c| srgb_to_linear(c / 255.0) * 255.0
            } else {
                |c| c
            };

            for t in 0..GRADIENT_SIZE {
                if last + 1 < gradient.records.len()
                    && t > gradient.records[last + 1].ratio as usize
                {
                    last += 1;
                }
                next = (last + 1).min(gradient.records.len() - 1);

                assert!(last == next || last + 1 == next);

                let last_record = &gradient.records[last];
                let next_record = &gradient.records[next];

                let a = if t <= last_record.ratio as usize || last_record.ratio == next_record.ratio
                {
                    // We are before the first gradient record,
                    // or this record's ratio is equal to the next one,
                    // meaning we need to do a full stop of this color for 1 pixel.
                    0.0
                } else if t > next_record.ratio as usize {
                    // We are after the last record
                    1.0
                } else {
                    (t as f32 - last_record.ratio as f32)
                        / (next_record.ratio as f32 - last_record.ratio as f32)
                };

                colors[t * 4] = lerp(
                    convert(last_record.color.r as f32),
                    convert(next_record.color.r as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 1] = lerp(
                    convert(last_record.color.g as f32),
                    convert(next_record.color.g as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 2] = lerp(
                    convert(last_record.color.b as f32),
                    convert(next_record.color.b as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 3] =
                    lerp(last_record.color.a as f32, next_record.color.a as f32, a) as u8;
            }

            colors
        };
        let texture = descriptors.device.create_texture_with_data(
            &descriptors.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: GRADIENT_SIZE as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &colors[..],
        );
        let view = texture.create_view(&Default::default());

        let buffer_offset = uniform_buffers
            .add(&[GradientUniforms::from(gradient)])
            .expect("Mesh uniform buffer was too large!")
            .start;

        Self {
            texture_view: view,
            buffer_offset,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BitmapBinds {
    pub bind_group: wgpu::BindGroup,
}

impl BitmapBinds {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        texture_view: wgpu::TextureView,
        label: Option<String>,
    ) -> Self {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
            label: label.as_deref(),
        });
        Self { bind_group }
    }
}
