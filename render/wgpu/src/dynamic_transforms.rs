use crate::buffer_builder::BufferBuilder;
use crate::descriptors::Descriptors;
use crate::{PosUvVertex, Transforms};
use std::mem;
use std::sync::Mutex;

const ESTIMATED_OBJECTS_PER_CHUNK: u64 = 200;

/// The size of the per-frame arenas that chunk transforms and vertices are stored in.
const TRANSFORMS_ARENA_SIZE: u64 = 8 * 1024 * 1024;
const VERTICES_ARENA_SIZE: u64 = 4 * 1024 * 1024;

/// Holds the per-object data (transforms and dynamic vertices) of draw chunks.
///
/// Every chunk gets its own region of an arena buffer, and all regions written
/// since the last submission are uploaded at once by `flush` (with a single
/// `Queue::write_buffer`, which is ordered after previously submitted work, so
/// the arena can be reused every frame). Previously, every chunk was copied into
/// the same small buffer right before its render pass, which costs a copy (and
/// on Metal, a command buffer) per chunk.
///
/// If an arena fills up, chunks fall back to being copied into a reserved region
/// at the end of the arena right before their pass, like before.
pub struct DynamicTransforms {
    pub buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    /// Binds `binding_size` bytes of `buffer`, at a dynamic offset.
    pub bind_group: wgpu::BindGroup,
    /// The maximum size of a chunk's transforms.
    pub binding_size: u64,
    /// The maximum size of a chunk's vertices.
    pub chunk_vertices_size: u64,
    uniform_alignment: u64,
    pending: Mutex<Pending>,
}

#[derive(Default)]
struct Pending {
    transforms: Vec<u8>,
    vertices: Vec<u8>,
}

/// Where a chunk's data is, for binding it.
#[derive(Clone, Copy)]
pub struct ChunkOffsets {
    /// The dynamic offset of the transforms bind group.
    pub transforms: u32,
    /// The offset of the chunk's vertices in `vertex_buffer`.
    pub vertices: wgpu::BufferAddress,
    /// Whether the data is in the region reserved for chunks that don't fit in the
    /// arenas, which only one chunk can use at a time.
    pub fallback: bool,
}

pub fn transforms_per_draw(limits: &wgpu::Limits) -> u64 {
    let maximum = limits.max_uniform_buffer_binding_size / size_of::<Transforms>() as u64;
    maximum.min(ESTIMATED_OBJECTS_PER_CHUNK)
}

fn align(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

impl DynamicTransforms {
    pub fn new(descriptors: &Descriptors) -> Self {
        let binding_size =
            transforms_per_draw(&descriptors.limits) * size_of::<Transforms>() as u64;
        let uniform_alignment = descriptors.limits.min_uniform_buffer_offset_alignment as u64;
        let chunk_vertices_size =
            mem::size_of::<PosUvVertex>() as u64 * 4 * ESTIMATED_OBJECTS_PER_CHUNK;

        let buffer = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
            label: create_debug_label!("Transforms arena").as_deref(),
            size: TRANSFORMS_ARENA_SIZE
                .min(descriptors.limits.max_buffer_size)
                .max(binding_size * 2),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
            label: create_debug_label!("Vertices arena").as_deref(),
            size: VERTICES_ARENA_SIZE
                .min(descriptors.limits.max_buffer_size)
                .max(chunk_vertices_size * 2),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &descriptors.bind_layouts.transforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(binding_size),
                    }),
                }],
            });
        Self {
            buffer,
            bind_group,
            vertex_buffer,
            binding_size,
            chunk_vertices_size,
            uniform_alignment,
            pending: Default::default(),
        }
    }

    /// The offsets of the regions reserved for chunks that don't fit in the arenas.
    fn fallback_offsets(&self) -> ChunkOffsets {
        let transforms = (self.buffer.size() - self.binding_size) / self.uniform_alignment
            * self.uniform_alignment;
        let vertices = (self.vertex_buffer.size() - self.chunk_vertices_size) / 256 * 256;
        ChunkOffsets {
            transforms: transforms as u32,
            vertices,
            fallback: true,
        }
    }

    /// Stores a chunk's data, returning where it can be bound from.
    pub fn upload(
        &self,
        transforms: BufferBuilder,
        vertices: BufferBuilder,
        staging_belt: &mut wgpu::util::StagingBelt,
        encoder: &mut wgpu::CommandEncoder,
    ) -> ChunkOffsets {
        let fallback = self.fallback_offsets();
        let mut pending = self.pending.lock().expect("pending lock");
        let transforms_offset = align(pending.transforms.len() as u64, self.uniform_alignment);
        let vertices_offset = align(pending.vertices.len() as u64, 256);
        let transforms_end = transforms_offset + transforms.bytes().len() as u64;
        let vertices_end = vertices_offset + vertices.bytes().len() as u64;

        if transforms_end <= fallback.transforms as u64 && vertices_end <= fallback.vertices {
            pending.transforms.resize(transforms_offset as usize, 0);
            pending.transforms.extend_from_slice(transforms.bytes());
            pending.vertices.resize(vertices_offset as usize, 0);
            pending.vertices.extend_from_slice(vertices.bytes());
            return ChunkOffsets {
                transforms: transforms_offset as u32,
                vertices: vertices_offset,
                fallback: false,
            };
        }

        // The arenas are full; copy into the reserved regions right before this chunk.
        drop(pending);
        transforms.copy_to(
            staging_belt,
            encoder,
            &self.buffer,
            fallback.transforms as u64,
        );
        vertices.copy_to(
            staging_belt,
            encoder,
            &self.vertex_buffer,
            fallback.vertices,
        );
        fallback
    }

    /// Uploads all chunk data stored since the last flush.
    /// Must be called before submitting any commands that draw those chunks.
    pub fn flush(&self, queue: &wgpu::Queue) {
        let mut guard = self.pending.lock().expect("pending lock");
        let pending = &mut *guard;
        for (data, buffer) in [
            (&mut pending.transforms, &self.buffer),
            (&mut pending.vertices, &self.vertex_buffer),
        ] {
            if !data.is_empty() {
                data.resize(
                    align(data.len() as u64, wgpu::COPY_BUFFER_ALIGNMENT) as usize,
                    0,
                );
                queue.write_buffer(buffer, 0, data);
                data.clear();
            }
        }
    }
}
