use crate::blend::ComplexBlend;
use crate::dynamic_transforms::transforms_per_draw;
use enum_map::{EnumMap, enum_map};
use ruffle_render::shader_source::SHADER_FILTER_COMMON;

#[derive(Debug)]
pub struct Shaders {
    pub color_shader: wgpu::ShaderModule,
    /// This has a pipeline-overridable `bool` constant, `late_saturate`,
    /// with a default of `false`. It switches to performing saturation
    /// after re-multiplying the alpha, rather than before. This is used
    /// for the Stage3D `bitmap_opaque` pipeline, which needs to be able to
    /// avoid changing initially-in-range rgb values (regadless of whether
    /// dividing by the alpha value would produce an out-of-range value).
    pub bitmap_shader: wgpu::ShaderModule,
    pub gradient_shader: wgpu::ShaderModule,
    pub copy_shader: wgpu::ShaderModule,
    pub alpha_mask_shader: wgpu::ShaderModule,
    pub blend_shaders: EnumMap<ComplexBlend, wgpu::ShaderModule>,
    /// Fragment shaders for complex blends that read the backdrop from the color
    /// attachment (framebuffer fetch), if the device supports them (see
    /// `supports_fetch_blends`).
    pub fetch_blend_shaders: Option<EnumMap<ComplexBlend, wgpu::ShaderModule>>,
    /// Like `fetch_blend_shaders`, for solid fills drawn directly with a complex blend
    /// (with `color.wgsl` as the vertex stage).
    pub fetch_color_shaders: Option<EnumMap<ComplexBlend, wgpu::ShaderModule>>,
    pub color_matrix_filter: wgpu::ShaderModule,
    pub blur_filter: wgpu::ShaderModule,
    pub glow_filter: wgpu::ShaderModule,
    pub bevel_filter: wgpu::ShaderModule,
    pub displacement_map_filter: wgpu::ShaderModule,
}

impl Shaders {
    pub fn new(device: &wgpu::Device) -> Self {
        let max_transforms = transforms_per_draw(&device.limits());
        let color_shader = make_shader(
            device,
            "color.wgsl",
            include_str!("../shaders/color.wgsl"),
            max_transforms,
        );
        let bitmap_shader = make_shader(
            device,
            "bitmap.wgsl",
            include_str!("../shaders/bitmap.wgsl"),
            max_transforms,
        );
        let copy_shader = make_shader(
            device,
            "copy.wgsl",
            include_str!("../shaders/copy.wgsl"),
            max_transforms,
        );
        let color_matrix_filter = make_filter_shader(
            device,
            "filter/color_matrix.wgsl",
            include_str!("../shaders/filter/color_matrix.wgsl"),
        );
        let blur_filter = make_filter_shader(
            device,
            "filter/blur.wgsl",
            include_str!("../shaders/filter/blur.wgsl"),
        );
        let glow_filter = make_filter_shader(
            device,
            "filter/glow.wgsl",
            include_str!("../shaders/filter/glow.wgsl"),
        );
        let bevel_filter = make_filter_shader(
            device,
            "filter/bevel.wgsl",
            include_str!("../shaders/filter/bevel.wgsl"),
        );
        let displacement_map_filter = make_filter_shader(
            device,
            "filter/displacement_map.wgsl",
            include_str!("../shaders/filter/displacement_map.wgsl"),
        );
        let gradient_shader = make_shader(
            device,
            "gradient.wgsl",
            include_str!("../shaders/gradient.wgsl"),
            max_transforms,
        );
        let alpha_mask_shader = make_shader(
            device,
            "alpha_mask.wgsl",
            include_str!("../shaders/alpha_mask.wgsl"),
            max_transforms,
        );

        let blend_shaders = enum_map! {
            ComplexBlend::Multiply => make_shader(device, "blend/multiply.wgsl", include_str!("../shaders/blend/multiply.wgsl"), max_transforms),
            ComplexBlend::Lighten => make_shader(device, "blend/lighten.wgsl", include_str!("../shaders/blend/lighten.wgsl"), max_transforms),
            ComplexBlend::Darken => make_shader(device, "blend/darken.wgsl", include_str!("../shaders/blend/darken.wgsl"), max_transforms),
            ComplexBlend::Difference => make_shader(device, "blend/difference.wgsl", include_str!("../shaders/blend/difference.wgsl"), max_transforms),
            ComplexBlend::Invert => make_shader(device, "blend/invert.wgsl", include_str!("../shaders/blend/invert.wgsl"), max_transforms),
            ComplexBlend::Alpha => make_shader(device, "blend/alpha.wgsl", include_str!("../shaders/blend/alpha.wgsl"), max_transforms),
            ComplexBlend::Erase => make_shader(device, "blend/erase.wgsl", include_str!("../shaders/blend/erase.wgsl"), max_transforms),
            ComplexBlend::Overlay => make_shader(device, "blend/overlay.wgsl", include_str!("../shaders/blend/overlay.wgsl"), max_transforms),
            ComplexBlend::HardLight => make_shader(device, "blend/hardlight.wgsl", include_str!("../shaders/blend/hardlight.wgsl"), max_transforms),
        };

        let fetch_blend_shaders = device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
            .then(|| make_fetch_blend_shaders(device, ""));
        let fetch_color_shaders = device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
            .then(|| make_fetch_blend_shaders(device, "#define VERTEX_COLOR"));

        Self {
            color_shader,
            bitmap_shader,
            gradient_shader,
            copy_shader,
            alpha_mask_shader,
            blend_shaders,
            fetch_blend_shaders,
            fetch_color_shaders,
            color_matrix_filter,
            blur_filter,
            glow_filter,
            bevel_filter,
            displacement_map_filter,
        }
    }
}

/// Whether complex blends can read their backdrop from the color attachment
/// (framebuffer fetch), which Metal supports on Apple GPUs. Requesting
/// `PASSTHROUGH_SHADERS` (see `request_device`) enables them.
pub fn supports_fetch_blends(adapter: &wgpu::Adapter) -> bool {
    let info = adapter.get_info();
    info.backend == wgpu::Backend::Metal
        && info.name.starts_with("Apple")
        && adapter
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
}

fn make_fetch_blend_shaders(
    device: &wgpu::Device,
    variant: &str,
) -> EnumMap<ComplexBlend, wgpu::ShaderModule> {
    let source = include_str!("../shaders/blend/fetch.metal");
    let make = |name: &str, definitions: &str| {
        let msl = format!("{variant}\n{definitions}\n{source}");
        // SAFETY: The source is ours, and matches the bitmap pipelines (see `fetch.metal`).
        unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: create_debug_label!("Fetch blend shader {name}").as_deref(),
                entry_points: std::borrow::Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed("main_fragment"),
                    workgroup_size: (0, 0, 0),
                }]),
                msl: Some(msl.into()),
                ..Default::default()
            })
        }
    };
    // `x <= 0.5 ? a : b` for each channel of `x`.
    let hard_light = "select(1.0 - 2.0 * (1.0 - d) * (1.0 - s), 2.0 * s * d, X <= 0.5)";
    enum_map! {
        ComplexBlend::Multiply => make("Multiply", "#define BLEND_FUNC(s, d) ((s) * (d))"),
        ComplexBlend::Lighten => make("Lighten", "#define BLEND_FUNC(s, d) max(s, d)"),
        ComplexBlend::Darken => make("Darken", "#define BLEND_FUNC(s, d) min(s, d)"),
        ComplexBlend::Difference => make("Difference", "#define BLEND_FUNC(s, d) abs((d) - (s))"),
        ComplexBlend::Invert => make("Invert", "#define BLEND_FUNC(s, d) (1.0 - (d))"),
        ComplexBlend::Alpha => make("Alpha", "#define COMPOSITE float4(dst.rgb * src.a, src.a * dst.a)"),
        ComplexBlend::Erase => make(
            "Erase",
            "#define COMPOSITE float4(dst.rgb * (1.0 - src.a), (1.0 - src.a) * dst.a)",
        ),
        ComplexBlend::Overlay => make(
            "Overlay",
            &format!("#define BLEND_FUNC(s, d) {}", hard_light.replace('X', "d")),
        ),
        ComplexBlend::HardLight => make(
            "HardLight",
            &format!("#define BLEND_FUNC(s, d) {}", hard_light.replace('X', "s")),
        ),
    }
}

fn make_shader(
    device: &wgpu::Device,
    name: &str,
    source: &str,
    max_transforms: u64,
) -> wgpu::ShaderModule {
    let common = include_str!("../shaders/common.wgsl");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: create_debug_label!("Shader {name}").as_deref(),
        source: wgpu::ShaderSource::Wgsl(
            format!("{common}\nconst max_transforms = {max_transforms};\n{source}").into(),
        ),
    })
}
fn make_filter_shader(device: &wgpu::Device, name: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: create_debug_label!("Shader {name}").as_deref(),
        source: wgpu::ShaderSource::Wgsl(format!("{SHADER_FILTER_COMMON}\n{source}").into()),
    })
}
