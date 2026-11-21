// Complex blends that read the pixel they blend onto from the color attachment
// (framebuffer fetch, Apple GPUs only), instead of from a copy of the backdrop.
// See `fetch_blend_shaders` in `shaders.rs`, which defines `BLEND_FUNC` or
// `COMPOSITE` (and one of `VERTEX_COLOR`, `FETCH_GRADIENT`, `FETCH_BITMAP` for
// direct fills) before this source.
//
// The vertex stage is `main_vertex` of `bitmap.wgsl`, `color.wgsl`, or `gradient.wgsl`
// (compiled by naga), so the input must match naga's names for its outputs. Bindings
// are those of group 2 (groups 0 and 1 are vertex-only).
// The formulas are those of the other shaders in this directory.

#include <metal_stdlib>
using namespace metal;

#ifdef VERTEX_COLOR
// A solid fill: the vertex stage is `main_vertex` of `color.wgsl`, which outputs the
// final (premultiplied) color, and there are no bindings.
struct FragmentInput {
    float4 position [[position]];
    float4 color [[user(loc0), center_perspective]];
};
#elif defined(FETCH_GRADIENT) || defined(FETCH_BITMAP)
// A gradient or bitmap fill: the vertex stage outputs uv, mult_color, and add_color.
struct FragmentInput {
    float4 position [[position]];
    float2 uv [[user(loc0), center_perspective]];
    float4 mult_color [[user(loc1), center_perspective]];
    float4 add_color [[user(loc2), center_perspective]];
};
#ifdef FETCH_GRADIENT
struct Gradient {
    float focal_point;
    int interpolation;
    int shape;
    int repeat;
};
#endif
#else
struct FragmentInput {
    float4 position [[position]];
    float2 uv [[user(loc0), center_perspective]];
};
#endif

fragment float4 main_fragment(
    FragmentInput in [[stage_in]],
    float4 dst [[color(0)]]
#ifdef FETCH_GRADIENT
    , constant Gradient& gradient [[buffer(0)]]
    , texture2d<float> current_texture [[texture(0)]]
    , sampler texture_sampler [[sampler(0)]]
#elif !defined(VERTEX_COLOR)
    , texture2d<float> current_texture [[texture(0)]]
    , sampler texture_sampler [[sampler(0)]]
#endif
) {
    // dst is the pixel (sample) we're blending onto, src the pixel we apply.
#ifdef VERTEX_COLOR
    float4 src = in.color;
#elif defined(FETCH_GRADIENT)
    float t;
    if (gradient.shape == 1) {
        t = in.uv.x;
    } else if (gradient.shape == 2) {
        t = length(in.uv * 2.0 - 1.0);
    } else {
        float2 uv = in.uv * 2.0 - 1.0;
        float2 d = float2(gradient.focal_point, 0.0) - uv;
        float l = length(d);
        d = d / l;
        t = l / (sqrt(1.0 - gradient.focal_point * gradient.focal_point * d.y * d.y) + gradient.focal_point * d.x);
    }
    if (gradient.repeat == 1) {
        t = saturate(t);
    } else if (gradient.repeat == 2) {
        if (t < 0.0) {
            t = -t;
        }
        if ((int(t) & 1) == 0) {
            t = fract(t);
        } else {
            t = 1.0 - fract(t);
        }
    } else if (gradient.repeat == 3) {
        t = fract(t);
    }
    float4 color = current_texture.sample(texture_sampler, float2(t, 0.0));
    if (gradient.interpolation != 0) {
        float3 rgb = color.a > 0.0 ? color.rgb / color.a : color.rgb;
        float3 a = 12.92 * rgb;
        float3 b = 1.055 * pow(max(rgb, float3(0.0)), float3(1.0 / 2.4)) - 0.055;
        float3 c = step(float3(0.0031308), rgb);
        color = float4(mix(a, b, c) * color.a, color.a);
    }
    float4 out_c = saturate(color * in.mult_color + in.add_color);
    float alpha = saturate(out_c.a);
    float4 src = float4(out_c.rgb * alpha, alpha);
#elif defined(FETCH_BITMAP)
    float4 color = current_texture.sample(texture_sampler, in.uv);
    if (color.a > 0.0) {
        color = float4(color.rgb / color.a, color.a);
        color = saturate(color * in.mult_color + in.add_color);
        color = float4(color.rgb * color.a, color.a);
    }
    float4 src = color;
#else
    float4 src = current_texture.sample(texture_sampler, in.uv);
#endif
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
