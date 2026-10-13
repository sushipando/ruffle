// Complex blends that read the pixel they blend onto from the color attachment
// (framebuffer fetch, Apple GPUs only), instead of from a copy of the backdrop.
// See `fetch_blend_shaders` in `shaders.rs`, which defines `BLEND_FUNC` or
// `COMPOSITE` before this source.
//
// The vertex stage is `main_vertex` of `bitmap.wgsl` (compiled by naga), so the
// input must match naga's names for its outputs. Bindings are those of the bitmap
// pipelines: the texture and sampler of group 2 are the fragment stage's first.
// The formulas are those of the other shaders in this directory.

#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float4 position [[position]];
    float2 uv [[user(loc0), center_perspective]];
};

fragment float4 main_fragment(
    FragmentInput in [[stage_in]],
    float4 dst [[color(0)]],
    texture2d<float> current_texture [[texture(0)]],
    sampler texture_sampler [[sampler(0)]]
) {
    // dst is the pixel (sample) we're blending onto, src the pixel we apply.
    float4 src = current_texture.sample(texture_sampler, in.uv);
    if (src.a <= 0.0) {
        return dst;
    }
#ifdef COMPOSITE
    return COMPOSITE;
#else
    float3 d = dst.a > 0.0 ? dst.rgb / dst.a : float3(0.0);
    float3 s = src.rgb / src.a;
    return float4(
        src.rgb * (1.0 - dst.a) + dst.rgb * (1.0 - src.a) + src.a * dst.a * BLEND_FUNC(s, d),
        src.a + dst.a * (1.0 - src.a)
    );
#endif
}
