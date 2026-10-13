/// Resolves a region of a multisampled texture into a single-sampled texture,
/// by averaging all samples (a box filter, like hardware MSAA resolves).
///
/// The offset from destination texels to source texels is passed packed into the
/// instance index (as signed 16 bit x in the low bits, y in the high bits), which
/// avoids creating a uniform buffer for every resolve.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) offset: vec2<i32>,
};

@group(0) @binding(0) var source: texture_multisampled_2d<f32>;

@vertex
fn main_vertex(
    @builtin(vertex_index) index: u32,
    @builtin(instance_index) packed_offset: u32,
) -> VertexOutput {
    // A single triangle covering the whole destination.
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let packed = bitcast<i32>(packed_offset);
    let offset = vec2<i32>((packed << 16u) >> 16u, packed >> 16u);
    return VertexOutput(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0), offset);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.position.xy) + in.offset;
    let samples = textureNumSamples(source);
    var sum = vec4<f32>(0.0);
    for (var i = 0u; i < samples; i += 1u) {
        sum += textureLoad(source, coord, i32(i));
    }
    return sum / f32(samples);
}
