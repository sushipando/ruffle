struct Filter {
    color: vec4<f32>,
    strength: f32,
    inner: u32,
    knockout: u32,
    composite_source: u32,
}

struct GlowVBlur {
    glow: Filter,
    dir_x: f32,
    dir_y: f32,
    count: u32,
    _padding: u32,
    taps: array<vec4<f32>, 16>,
}

struct GlowColorMatrix {
    glow: Filter,
    row_r: vec4<f32>,
    row_g: vec4<f32>,
    row_b: vec4<f32>,
    row_a: vec4<f32>,
    extra: vec4<f32>,
}

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var<uniform> filter_args: Filter;
@group(0) @binding(3) var blurred: texture_2d<f32>;
@group(0) @binding(4) var<uniform> vblur_args: GlowVBlur;
@group(0) @binding(5) var<uniform> cm_args: GlowColorMatrix;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) source_uv: vec2<f32>,
    @location(1) blur_uv: vec2<f32>,
};

struct VertexInput {
    /// The position of the vertex in texture space (topleft 0,0, bottomright 1,1)
    @location(0) position: vec2<f32>,

    /// The coordinate of the source texture to sample in texture space (topleft 0,0, bottomright 1,1)
    @location(1) source_uv: vec2<f32>,

    /// The coordinate of the blur texture to sample in texture space (topleft 0,0, bottomright 1,1)
    @location(2) blur_uv: vec2<f32>,
};

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    // Convert texture space (topleft 0,0 to bottomright 1,1) to render space (topleft -1,1 to bottomright 1,-1)
    let pos = vec4<f32>((in.position.x * 2.0 - 1.0), (1.0 - in.position.y * 2.0), 0.0, 1.0);
    return VertexOutput(pos, in.source_uv, in.blur_uv);
}

fn apply_glow(dest: vec4<f32>, blur_in: f32, blur_uv: vec2<f32>, args: Filter) -> vec4<f32> {
    let inner = args.inner > 0u;
    let knockout = args.knockout > 0u;
    let composite_source = args.composite_source > 0u;
    var blur = blur_in;

    if (blur_uv.x < 0.0 || blur_uv.x > 1.0 || blur_uv.y < 0.0 || blur_uv.y > 1.0) {
        blur = 0.0;
    }

    // [NA] It'd be nice to use hardware blending but the operation is too complex :( Only knockouts would work.

    // Start with 1 alpha because we'll be multiplying the whole thing
    var color = vec4<f32>(args.color.r, args.color.g, args.color.b, 1.0);
    if (inner) {
        let alpha = args.color.a * saturate((1.0 - blur) * args.strength);
        if (knockout) {
            color = color * alpha * dest.a;
        } else if (composite_source) {
            color = color * alpha * dest.a + dest * (1.0 - alpha);
        } else {
            // [NA] Yes it's intentional that the !composite_source is different for inner/outer. Just Flash things.
            color = color * alpha * dest.a;
        }
    } else {
        let alpha = args.color.a * saturate(blur * args.strength);
        if (knockout) {
            color = color * alpha * (1.0 - dest.a);
        } else if (composite_source) {
            color = color * alpha * (1.0 - dest.a) + dest;
        } else {
            color = color * alpha;
        }
    }

    return color;
}

fn apply_color_matrix(glow_out: vec4<f32>, cm: GlowColorMatrix) -> vec4<f32> {
    let src = round(clamp(glow_out, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0) / 255.0;
    let unpremul = select(vec3<f32>(0.0), src.rgb / src.a, src.a > 0.0);
    let v = vec4<f32>(unpremul, src.a);
    let color = clamp(
        vec4<f32>(
            dot(cm.row_r, v),
            dot(cm.row_g, v),
            dot(cm.row_b, v),
            dot(cm.row_a, v),
        ) + cm.extra,
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
    return vec4<f32>(color.rgb * color.a, color.a);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let blur = textureSample(blurred, texture_sampler, in.blur_uv).a;
    let dest = textureSample(texture, texture_sampler, in.source_uv);
    return apply_glow(dest, blur, in.blur_uv, filter_args);
}

@fragment
fn main_fragment_r8(in: VertexOutput) -> @location(0) vec4<f32> {
    let blur = textureSample(blurred, texture_sampler, in.blur_uv).r;
    let dest = textureSample(texture, texture_sampler, in.source_uv);
    return apply_glow(dest, blur, in.blur_uv, filter_args);
}

@fragment
fn main_fragment_vblur_rgba(in: VertexOutput) -> @location(0) vec4<f32> {
    let direction = vec2<f32>(vblur_args.dir_x, vblur_args.dir_y);
    var total = 0.0;
    for (var i = 0u; i < vblur_args.count; i += 2u) {
        let pair = vblur_args.taps[i / 2u];
        total += textureSample(blurred, texture_sampler, in.blur_uv + direction * pair.x).a * pair.y;
        total += textureSample(blurred, texture_sampler, in.blur_uv + direction * pair.z).a * pair.w;
    }
    let blur = floor(total * 255.0) / 255.0;
    let dest = textureSample(texture, texture_sampler, in.source_uv);
    return apply_glow(dest, blur, in.blur_uv, vblur_args.glow);
}

@fragment
fn main_fragment_vblur_r8(in: VertexOutput) -> @location(0) vec4<f32> {
    let direction = vec2<f32>(vblur_args.dir_x, vblur_args.dir_y);
    var total = 0.0;
    for (var i = 0u; i < vblur_args.count; i += 2u) {
        let pair = vblur_args.taps[i / 2u];
        total += textureSample(blurred, texture_sampler, in.blur_uv + direction * pair.x).r * pair.y;
        total += textureSample(blurred, texture_sampler, in.blur_uv + direction * pair.z).r * pair.w;
    }
    let blur = floor(total * 255.0) / 255.0;
    let dest = textureSample(texture, texture_sampler, in.source_uv);
    return apply_glow(dest, blur, in.blur_uv, vblur_args.glow);
}

@fragment
fn main_fragment_cm(in: VertexOutput) -> @location(0) vec4<f32> {
    let blur = textureSample(blurred, texture_sampler, in.blur_uv).a;
    let dest = textureSample(texture, texture_sampler, in.source_uv);
    return apply_color_matrix(apply_glow(dest, blur, in.blur_uv, cm_args.glow), cm_args);
}

@fragment
fn main_fragment_r8_cm(in: VertexOutput) -> @location(0) vec4<f32> {
    let blur = textureSample(blurred, texture_sampler, in.blur_uv).r;
    let dest = textureSample(texture, texture_sampler, in.source_uv);
    return apply_color_matrix(apply_glow(dest, blur, in.blur_uv, cm_args.glow), cm_args);
}
