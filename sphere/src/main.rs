use std::io::Write;

use cgmath::prelude::*;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use futures::executor::block_on;

use wgpu::util::DeviceExt;

use bytemuck::{Pod, Zeroable};

mod mesh;
use mesh::Vertex;

mod blur;
use blur::{BlendUniforms, BlurUniforms, BlurVertex};

// sample count for MSAA
const SAMPLE_COUNT: u32 = 4;
const SHADOW_RES: u32 = 4;

const IMAGE_DIR: &str = "img";

// Number of frames to finish one iteration of subdivision
const INTERVAL: u32 = 100000;

// Threshold to choose the blight part to add bloom effect [0-1]
const BLIGHTNESS_THRESHOLD: f32 = 0.0;

// how many times to repeat gaussian blur
const BLUR_COUNT: usize = 20;

// exposure level used in blend.frag
const EXPOSURE: f32 = 5.0;

// gamma correction used in blend.frag
const GAMMA: f32 = 2.2;

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unused)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

struct PNGDimensions {
    width: usize,
    height: usize,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl PNGDimensions {
    fn new(width: usize, height: usize) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>();
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

const NUM_INSTANCES: u32 = 81;
const SIZE_OF_CUBE: f32 = 2.0;
const INTERVAL_BETWEEN_CUBE: f32 = 1.5;
const SHARPNESS: Option<f32> = Some(2.0);
const SUBDIVIDE_LIMIT: usize = 10;
const PLANE_SIZE: u32 = 1000;

const BG_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.22,
    b: 0.3,
    a: 1.0,
};

// The original code is https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/#another-better-way-storage-buffers
struct CubeInstance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    color: cgmath::Vector3<f32>,
    normal: f32,
}

impl CubeInstance {
    fn to_raw(&self) -> CubeInstanceRaw {
        let model =
            cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        CubeInstanceRaw {
            model: model.into(),
            color: self.color.into(),
            normal: self.normal,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CubeInstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 3],
    normal: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    position: [f32; 4],
    view_proj: [[f32; 4]; 4],
    num_of_lights: u32,
    blightness_threshold: f32,
    self_luminous_id: u32,
    _padding: [u32; 1],
}

fn generate_global_uniform(aspect_ratio: f32, frame: u32, num_of_lights: u32) -> Globals {
    let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 0.5, 200.0);
    let rot1 = (frame + 400) as f32 / 200.0;

    let rot2_max = std::f32::consts::PI * 1.0 / 13.0;
    let rot2 = rot2_max;
    // * ((3001 - std::cmp::min(frame, 3000)) as f32 / 3000.0).powi(3);

    let distance = 51.0f32 + (frame as f32 / 100.0);
    let eye = cgmath::Point3::new(
        distance * rot1.sin() * rot2.sin(),
        distance * rot1.cos() * rot2.sin(),
        distance * rot2.cos(),
    );
    let mx_view =
        cgmath::Matrix4::look_at(eye, cgmath::Point3::origin(), cgmath::Vector3::unit_z());

    Globals {
        position: eye.to_homogeneous().into(),
        view_proj: (OPENGL_TO_WGPU_MATRIX * mx_projection * mx_view).into(),
        num_of_lights,
        blightness_threshold: BLIGHTNESS_THRESHOLD,
        self_luminous_id: (frame as f32 / 5.0).ceil() as u32 % NUM_INSTANCES + 1, // TODO: exclude plane to be illuminated
        _padding: [0],
    }
}

// main.rs
#[repr(C)]
#[derive(Clone, Copy)]
struct Light {
    position: cgmath::Point3<f32>,
    color: cgmath::Vector3<f32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LightRaw {
    // Though we only need vec3 for position and color, due to uniforms
    // requiring 16 byte (4 float) spacing, use vec4 to align with the requirement
    view_proj: cgmath::Matrix4<f32>,
    position: cgmath::Vector4<f32>,
    color: cgmath::Vector4<f32>,
}

unsafe impl bytemuck::Pod for LightRaw {}
unsafe impl bytemuck::Zeroable for LightRaw {}

impl Light {
    fn new(position: cgmath::Point3<f32>, color: cgmath::Vector3<f32>) -> Self {
        Self { position, color }
    }

    fn to_raw(&self) -> LightRaw {
        let mx_view = cgmath::Matrix4::look_at(
            self.position,
            cgmath::Point3::origin(),
            cgmath::Vector3::unit_z(),
        );
        let mx_projection = cgmath::perspective(cgmath::Deg(60.0), 1.0, 10.0, 200.0);
        LightRaw {
            view_proj: OPENGL_TO_WGPU_MATRIX * mx_projection * mx_view,
            position: self.position.to_homogeneous(),
            color: self.color.extend(1.0),
        }
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,

    render_pipeline: wgpu::RenderPipeline,
    staging_texture: wgpu::Texture,

    cube: mesh::Polygon,
    vertex_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    cube_bind_group: wgpu::BindGroup,

    plane_vertex_buf: wgpu::Buffer,
    plane_index_buf: wgpu::Buffer,
    plane_index_count: usize,
    plane_bind_group: wgpu::BindGroup,

    depth_texture: wgpu::Texture, // TODO: This is exposed for debugging purposes.
    depth_texture_view: wgpu::TextureView,

    globals_bind_group: wgpu::BindGroup,
    globals_buffer: wgpu::Buffer,
    lights: Vec<Light>,
    light_buffer: wgpu::Buffer,
    light_tmp_buffer: wgpu::Buffer,

    // shadow
    shadow_bind_group: wgpu::BindGroup,
    shadow_target_views: Vec<wgpu::TextureView>,
    shadow_render_pipeline: wgpu::RenderPipeline,
    shadow_sampler: wgpu::Sampler, // shadow sampler is a sampler for depth texture and is used in blur as well to do DoF

    // A render pipeline to apply gaussian blur. To use gaussian blur, we need two
    // textures so that one can be rendered to another and vice versa.
    blur_bind_group_layout: wgpu::BindGroupLayout,
    blur_uniform_buffer: wgpu::Buffer,
    blur_uniform_bind_group: wgpu::BindGroup,
    blur_render_pipeline: wgpu::RenderPipeline,
    blur_texture_views: [wgpu::TextureView; 2],
    blur_sampler: wgpu::Sampler,
    square_vertex: wgpu::Buffer,

    // A render pipeline for blending the results
    blend_bind_group_layout: wgpu::BindGroupLayout,
    blend_uniform_buffer: wgpu::Buffer,
    blend_uniform_bind_group: wgpu::BindGroup,
    blend_render_pipeline: wgpu::RenderPipeline,

    // Texture for MASS
    multisample_texture_view: wgpu::TextureView, // two textures are needed at the same time as we need usual one and the one for blur separately
    multisample_png_texture_view: wgpu::TextureView, // a texture for PNG has a different TextureFormat, so we need another multisampled texture than others

    // Texture for writing out as PNG
    png_texture: wgpu::Texture,
    png_buffer: wgpu::Buffer,
    png_dimensions: PNGDimensions,

    size: winit::dpi::PhysicalSize<u32>,

    output_dir: std::path::PathBuf,

    write_buffers: bool,

    frame: u32,
    next_frame: u32,
    record: bool,
}

impl State {
    async fn new(window: &Window) -> Self {
        // create an instance
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        // create an surface
        let (size, surface) = unsafe {
            let size = window.inner_size();
            let surface = instance.create_surface(window);
            (size, surface)
        };

        // create an adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        // create a device and a queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::default() | wgpu::Features::DEPTH_CLAMPING,
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                // trace_path can be used for API call tracing
                None,
            )
            .await
            .unwrap();

        // create a swap chain
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        // Create vertices and indices buffers of cubes
        let cube = mesh::calculate_initial_cube(SHARPNESS);
        let (vertex_data, index_data) = cube.triangulate();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsage::INDEX,
        });

        // Create vertices and indices of plane
        let (plane_vertex_data, plane_index_data) = mesh::create_plane(PLANE_SIZE);
        let plane_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Vertex Buffer"),
            contents: bytemuck::cast_slice(&plane_vertex_data),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let plane_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Index Buffer"),
            contents: bytemuck::cast_slice(&plane_index_data),
            usage: wgpu::BufferUsage::INDEX,
        });

        let globals_size = std::mem::size_of::<Globals>();
        let instance_size = std::mem::size_of::<CubeInstanceRaw>();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(globals_size as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(
                            instance_size as _,
                            //     (NUM_INSTANCES * instance_size as u32) as _,
                        ),
                        readonly: true,
                    },
                    count: None,
                },
            ],
        });

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: globals_size as _,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_stencil_state = wgpu::DepthStencilStateDescriptor {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilStateDescriptor {
                front: wgpu::StencilStateFaceDescriptor::IGNORE,
                back: wgpu::StencilStateFaceDescriptor::IGNORE,
                read_mask: 0,
                write_mask: 0,
            },
        };

        // Instances ----------------------------------------------------------------------

        let instance_data = create_instance_data(0);
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            label: None,
        });

        let plane_instance_data = [CubeInstanceRaw {
            model: cgmath::Matrix4::identity().into(),
            color: cgmath::vec3(0.8, 0.8, 0.8).into(),
            normal: 0.1,
        }];
        let plane_instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&plane_instance_data),
            usage: wgpu::BufferUsage::STORAGE,
            label: None,
        });

        // Light ------------------------------------------------------------------------------------------------------------
        let lights = vec![
            Light::new((20.0, 20.0, 100.0).into(), (0.2, 0.2, 0.2).into()),
            Light::new((22.0, 18.0, 90.0).into(), (0.2, 0.2, 0.2).into()),
            Light::new((24.0, 16.0, 80.0).into(), (0.2, 0.2, 0.2).into()),
        ];
        let light_size = std::mem::size_of::<LightRaw>() as u64;

        // We'll want to update our lights position, so we use COPY_DST
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: light_size * lights.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::UNIFORM
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });
        // a buffer to use for baking depth map
        let light_tmp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: light_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::UniformBuffer {
                            dynamic: false,
                            min_binding_size: wgpu::BufferSize::new(
                                light_size * lights.len() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            component_type: wgpu::TextureComponentType::DepthComparison,
                            dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: true },
                        count: None,
                    },
                ],
                label: None,
            });

        // Create other resources
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear, // Change this to Nearest to disable hardware PCF
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: sc_desc.width * SHADOW_RES,
                height: sc_desc.height * SHADOW_RES,
                depth: lights.len() as _,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
            label: None,
        });

        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_target_views = (0..lights.len())
            .map(|i| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow"),
                    format: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    level_count: None,
                    base_array_layer: i as _,
                    array_layer_count: std::num::NonZeroU32::new(1),
                })
            })
            .collect::<Vec<_>>();

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &globals_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
            label: None,
        });

        let vertex_attrs_vertex = wgpu::vertex_attr_array![0 => Float4, 1 => Float3];
        let vertex_buffers = [wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &vertex_attrs_vertex,
        }];

        // Create bind group
        let cube_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buf.as_entire_binding(),
                },
            ],
            label: None,
        });
        let plane_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: plane_instance_buf.as_entire_binding(),
                },
            ],
            label: None,
        });

        // Render pipeline ------------------------------------------------------------------------------------------------------------

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout, &globals_bind_group_layout],
                push_constant_ranges: &[],
                label: None,
            });

        let render_pipeline = create_render_pipeline(
            &device,
            &render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.vert.spv")),
            Some(&device.create_shader_module(wgpu::include_spirv!("shaders/shader.frag.spv"))),
            None, // use the default rasterization_state_descripor
            &vertex_buffers,
            1,
            vec![sc_desc.format, sc_desc.format],
            Some(depth_stencil_state.clone()),
        );

        // Texture to draw the unmodified version
        let staging_texture = create_framebuffer(&device, &sc_desc, sc_desc.format);

        // shadow --------------------------------------------------------------------------------------------------------
        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(light_size),
                    },
                    count: None,
                }],
                label: None,
            });

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shadow_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_tmp_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let shadow_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout, &shadow_bind_group_layout],
                push_constant_ranges: &[],
                label: None,
            });

        let mut shadow_depth_stencil_state = depth_stencil_state.clone();
        shadow_depth_stencil_state.depth_compare = wgpu::CompareFunction::LessEqual;
        let shadow_render_pipeline = create_render_pipeline(
            &device,
            &shadow_render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/bake.vert.spv")),
            None,
            Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 2, // corresponds to bilinear filtering
                depth_bias_slope_scale: 2.0,
                depth_bias_clamp: 0.0,
                clamp_depth: device.features().contains(wgpu::Features::DEPTH_CLAMPING),
                polygon_mode: wgpu::PolygonMode::Fill,
            }),
            &vertex_buffers,
            1,
            vec![],
            Some(shadow_depth_stencil_state),
        );

        // Depth texture (this is used for blur as well) -----------------------------------------------------------------------------

        let depth_texture = create_depth_texture(&device, &sc_desc);
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Blur ----------------------------------------------------------------------------------------------------------------------
        //
        // Blur render pipeline needs two bind groups:
        //   - A bind group for texture containing the bright part of the drawing
        //   - A bind group for parameters of gaussian blur
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // texture to blur
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: false },
                        count: None,
                    },
                    // depth mapping of the camera view
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: true },
                        count: None,
                    },
                ],
                label: None,
            });

        let blur_uniform_buffer_size = std::mem::size_of::<BlurUniforms>() as _;

        let blur_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: true, // needs two uniforms for horizontally and vertically
                        min_binding_size: wgpu::BufferSize::new(blur_uniform_buffer_size),
                    },
                    count: None,
                }],
                label: None,
            });

        // Blur uniform buffer is dynamic because it needs to contain two uniforms for two-pass gaussian blur (horizontally and vertically)
        // This is because, as queue.write_buffer() is only ordered with submit(), we cannot write_buffer() on the same region of a buffer.
        // Note that dynamic offsets also have to be aligned to `BIND_BUFFER_ALIGNMENT`.
        let blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 2 * wgpu::BIND_BUFFER_ALIGNMENT,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let blur_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &blur_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &blur_uniform_buffer,
                    offset: 0, // We set a DynamicOffset at set_bind_group(), so this is 0
                    size: wgpu::BufferSize::new(blur_uniform_buffer_size),
                },
            }],
            label: None,
        });

        let blur_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&blur_bind_group_layout, &blur_uniform_bind_group_layout],
                push_constant_ranges: &[],
                label: None,
            });

        let blur_render_pipeline = create_render_pipeline(
            &device,
            &blur_render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.vert.spv")),
            Some(&device.create_shader_module(wgpu::include_spirv!("shaders/blur.frag.spv"))),
            None,
            &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<BlurVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float2],
            }],
            1,
            vec![sc_desc.format],
            None,
        );

        // Textures to process gaussian blur in a ping-pong manner
        let blur_texture_views = [
            create_framebuffer(&device, &sc_desc, sc_desc.format)
                .create_view(&wgpu::TextureViewDescriptor::default()),
            create_framebuffer(&device, &sc_desc, sc_desc.format)
                .create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        let blur_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        // Vetex to draw the texture identically
        let square_vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&blur::BLUR_VERTICES),
            usage: wgpu::BufferUsage::VERTEX,
        });

        // Blend ------------------------------------------------------------------------------------------------------------------
        //
        // Blend render pipeline needs two bind groups
        //   - A bind group for texture to blend
        //   - A bind group for parameters of blending
        let blend_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: true,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: true,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: false },
                        count: None,
                    },
                ],
                label: None,
            });

        // Parameters for blend
        let blend_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (std::mem::size_of::<f32>() * 2) as _,
                        ),
                    },
                    count: None,
                }],
                label: None,
            });

        let blend_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<BlendUniforms>() as _,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let blend_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &blend_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: blend_uniform_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let blend_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&blend_bind_group_layout, &blend_uniform_bind_group_layout],
                push_constant_ranges: &[],
                label: None,
            });

        let blend_render_pipeline = create_render_pipeline(
            &device,
            &blend_render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.vert.spv")),
            Some(&device.create_shader_module(wgpu::include_spirv!("shaders/blend.frag.spv"))),
            None,
            &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<BlurVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float2],
            }],
            SAMPLE_COUNT,
            vec![sc_desc.format, wgpu::TextureFormat::Rgba8UnormSrgb], // Texture to write out as PNG needs to be in RGBA format
            None,
        );

        // MSAA --------------------------------------------------------------------------------------------------------

        let multisample_texture_view =
            create_multisampled_framebuffer(&device, &sc_desc, sc_desc.format)
                .create_view(&wgpu::TextureViewDescriptor::default());
        let multisample_png_texture_view =
            create_multisampled_framebuffer(&device, &sc_desc, wgpu::TextureFormat::Rgba8UnormSrgb)
                .create_view(&wgpu::TextureViewDescriptor::default());

        // PNG output ----------------------------------------------------------------------------------------------------

        // Output dir
        let mut output_dir = std::path::PathBuf::new();
        output_dir.push(IMAGE_DIR);
        if !output_dir.is_dir() {
            std::fs::create_dir(output_dir.clone()).unwrap();
        }

        // PNG size, buffer, and texture
        let (png_dimensions, png_buffer, png_texture) =
            create_png_texture_and_buffer(&device, sc_desc.width as usize, sc_desc.height as usize);

        State {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,

            render_pipeline,
            staging_texture,

            cube,
            vertex_buf,
            instance_buf,
            index_buf,
            index_count: index_data.len(),
            cube_bind_group,

            plane_index_buf,
            plane_vertex_buf,
            plane_index_count: plane_index_data.len(),
            plane_bind_group,

            blur_bind_group_layout,
            blur_uniform_buffer,
            blur_uniform_bind_group,
            blur_render_pipeline,
            blur_texture_views,
            blur_sampler,
            square_vertex,

            blend_bind_group_layout,
            blend_render_pipeline,
            blend_uniform_buffer,
            blend_uniform_bind_group,

            multisample_texture_view,
            multisample_png_texture_view,

            depth_texture,
            depth_texture_view,

            globals_bind_group,
            globals_buffer,

            lights,
            light_buffer,
            light_tmp_buffer,
            shadow_render_pipeline,
            shadow_bind_group,
            shadow_target_views,
            shadow_sampler,

            png_texture,
            png_buffer,
            png_dimensions,

            size,

            output_dir,

            write_buffers: true,

            frame: 0,
            next_frame: INTERVAL / 4,
            record: false,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        println!("Resized to {:?}", new_size);
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);

        self.staging_texture = create_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format);

        self.multisample_texture_view =
            create_multisampled_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format)
                .create_view(&wgpu::TextureViewDescriptor::default());
        self.blur_texture_views = [
            create_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format)
                .create_view(&wgpu::TextureViewDescriptor::default()),
            create_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format)
                .create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        self.multisample_png_texture_view = create_multisampled_framebuffer(
            &self.device,
            &self.sc_desc,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

        self.depth_texture = create_depth_texture(&self.device, &self.sc_desc);
        self.depth_texture_view = self
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let (png_dimensions, png_buffer, png_texture) = create_png_texture_and_buffer(
            &self.device,
            self.sc_desc.width as usize,
            self.sc_desc.height as usize,
        );

        self.png_dimensions = png_dimensions;
        self.png_buffer = png_buffer;
        self.png_texture = png_texture;
    }

    fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        self.frame += 1;
        if self.record && (self.frame >= 1000) {
            println!("End recording");
            self.frame = 0;
            self.record = false;
        }

        // If there is too many points, give up deviding
        if self.cube.n_corners > SUBDIVIDE_LIMIT {
            return;
        }

        let mut changed = false;
        if self.frame >= self.next_frame {
            for _ in 0..(self.cube.n_corners as f32 / INTERVAL as f32).ceil() as usize {
                self.cube.subdivide();
                changed = true;
            }

            self.next_frame +=
                (INTERVAL as f32 / (self.cube.n_corners as f32).powf(1.1)).ceil() as u32;
        }

        if changed {
            let (vertex_data, index_data) = self.cube.triangulate();

            self.vertex_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsage::VERTEX,
                });

            self.index_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(&index_data),
                    usage: wgpu::BufferUsage::INDEX,
                });
            self.index_count = index_data.len();
        }
    }

    fn render(&mut self) {
        let frame = match self.swap_chain.get_current_frame() {
            Ok(frame) => frame,
            Err(_) => {
                self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
                self.swap_chain
                    .get_current_frame()
                    .expect("Failed to acquire next swap chain texture!")
            }
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let vp_uniforms = generate_global_uniform(
            self.sc_desc.width as f32 / self.sc_desc.height as f32,
            self.frame,
            self.lights.len() as _,
        );

        self.queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::cast_slice(&[vp_uniforms]),
        );

        if self.write_buffers {
            self.write_buffers = false;

            // write light buffer
            for (i, light) in self.lights.iter().enumerate() {
                self.queue.write_buffer(
                    &self.light_buffer,
                    (i * std::mem::size_of::<LightRaw>()) as wgpu::BufferAddress,
                    bytemuck::bytes_of(&light.to_raw()),
                );
            }

            // write blur uniform buffer
            let blur_uniforms = (0..=1).map(BlurUniforms::new).collect::<Vec<_>>();
            self.queue.write_buffer(
                &self.blur_uniform_buffer,
                0,
                bytemuck::cast_slice(&[blur_uniforms[0]]),
            );
            self.queue.write_buffer(
                &self.blur_uniform_buffer,
                wgpu::BIND_BUFFER_ALIGNMENT,
                bytemuck::cast_slice(&[blur_uniforms[1]]),
            );
        }

        self.queue.write_buffer(
            &self.instance_buf,
            0,
            bytemuck::cast_slice(&create_instance_data(self.frame)),
        );

        // render shadows --------------------------------------------------------------------------------

        for i in 0..self.lights.len() {
            encoder.copy_buffer_to_buffer(
                &self.light_buffer,
                (i * std::mem::size_of::<LightRaw>()) as wgpu::BufferAddress,
                &self.light_tmp_buffer,
                0,
                std::mem::size_of::<LightRaw>() as _,
            );

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.shadow_target_views[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.shadow_render_pipeline);

            // draw cube
            render_pass.set_bind_group(0, &self.cube_bind_group, &[]);
            render_pass.set_bind_group(1, &self.shadow_bind_group, &[]);
            render_pass.set_index_buffer(self.index_buf.slice(..));
            render_pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            render_pass.draw_indexed(0..(self.index_count as u32), 0, 0..NUM_INSTANCES);

            // draw plane (Probably I don't need to render plane as it doesn't make shadow by itself...?)
            // render_pass.set_bind_group(0, &self.plane_bind_group, &[]);
            // render_pass.set_index_buffer(self.plane_index_buf.slice(..));
            // render_pass.set_vertex_buffer(0, self.plane_vertex_buf.slice(..));
            // render_pass.draw_indexed(0..(self.plane_index_count as u32), 0, 0..1);
        }

        let staging_texture_view = self
            .staging_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // render forward pass ----------------------------------------------------------------------------------------

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &staging_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(BG_COLOR),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &self.blur_texture_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(BG_COLOR),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: true,
                    }),
                }),
            });

            render_pass.set_pipeline(&self.render_pipeline);

            // draw cube
            render_pass.set_bind_group(0, &self.cube_bind_group, &[]);
            render_pass.set_bind_group(1, &self.globals_bind_group, &[]);
            render_pass.set_index_buffer(self.index_buf.slice(..));
            render_pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            render_pass.draw_indexed(0..(self.index_count as u32), 0, 0..NUM_INSTANCES);

            // draw plane
            render_pass.set_bind_group(0, &self.plane_bind_group, &[]);
            render_pass.set_index_buffer(self.plane_index_buf.slice(..));
            render_pass.set_vertex_buffer(0, self.plane_vertex_buf.slice(..));
            render_pass.draw_indexed(0..(self.plane_index_count as u32), 0, 0..1);
        }

        // Blur ------------------------------------------------------------------------------------------------------------------

        // Apply blur multiple times
        let blur_count = BLUR_COUNT;

        for i in 0..blur_count {
            let src = i % 2;
            let dst = (i + 1) % 2;

            let blur_uniform_offset: wgpu::DynamicOffset =
                ((i % 2) * wgpu::BIND_BUFFER_ALIGNMENT as usize) as _;

            let blur_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.blur_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.blur_texture_views[src]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.blur_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.depth_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                ],
                label: None,
            });

            {
                let mut blur_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &self.blur_texture_views[dst],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });

                blur_render_pass.set_pipeline(&self.blur_render_pipeline);
                blur_render_pass.set_bind_group(0, &blur_bind_group, &[]);
                blur_render_pass.set_bind_group(
                    1,
                    &self.blur_uniform_bind_group,
                    &[blur_uniform_offset],
                );
                blur_render_pass.set_vertex_buffer(0, self.square_vertex.slice(..));

                blur_render_pass.draw(0..blur::BLUR_VERTICES.len() as u32, 0..1);
            }
        }

        // Blend ------------------------------------------------------------------------------------------------------------------

        let blend_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.blend_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // a texture that contains the unmodified version
                    resource: wgpu::BindingResource::TextureView(&staging_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    // a texture that contains the last result of gaussian blur
                    resource: wgpu::BindingResource::TextureView(
                        &self.blur_texture_views[blur_count % 2], // TODO: needs +1...?
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.blur_sampler),
                },
            ],
            label: None,
        });

        let blend_uniform = BlendUniforms::new(
            EXPOSURE * (0.7 + 0.3 * (self.frame as f32 / 100.0).sin()),
            GAMMA,
        );

        self.queue.write_buffer(
            &self.blend_uniform_buffer,
            0,
            bytemuck::cast_slice(&[blend_uniform]),
        );

        let png_texture_view = self
            .png_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut blend_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &self.multisample_texture_view,
                        resolve_target: Some(&frame.output.view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &self.multisample_png_texture_view,
                        resolve_target: Some(&png_texture_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: None,
            });

            blend_render_pass.set_pipeline(&self.blend_render_pipeline);
            blend_render_pass.set_bind_group(0, &blend_bind_group, &[]);
            blend_render_pass.set_bind_group(1, &self.blend_uniform_bind_group, &[]);
            blend_render_pass.set_vertex_buffer(0, self.square_vertex.slice(..));

            blend_render_pass.draw(0..blur::BLUR_VERTICES.len() as u32, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &self.png_texture,
                // texture: &self.shadow_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &self.png_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: self.png_dimensions.padded_bytes_per_row as u32,
                    rows_per_image: 0,
                },
            },
            wgpu::Extent3d {
                width: self.sc_desc.width,
                height: self.sc_desc.height,
                depth: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        if self.record {
            let file = self.output_dir.clone();
            block_on(create_png(
                &file
                    .join(format!("{:03}.png", self.frame))
                    .to_str()
                    .unwrap(),
                &self.device,
                &self.png_buffer,
                &self.png_dimensions,
            ))
        }
    }
}

fn create_instance_data(frame: u32) -> Vec<CubeInstanceRaw> {
    let width = (NUM_INSTANCES as f32).sqrt().round() as u32;
    let offset = (width / 2) as i32;

    let instances: Vec<CubeInstance> = (0..NUM_INSTANCES)
        .map(|x| {
            let row = (x / width) as i32;
            let col = (x % width) as i32;
            let phase = (5 * row + 3 * col + 2 * row * col) as f32 * std::f32::consts::PI / 30.0;
            let z = (3.0 + 4.0 * (frame as f32 / 60.0 + phase).sin())
                * (1.0 + frame as f32 / 1000.0)
                * 5.0;
            let position = cgmath::Vector3 {
                x: (row - offset) as f32 * (SIZE_OF_CUBE + INTERVAL_BETWEEN_CUBE),
                y: (col - offset) as f32 * (SIZE_OF_CUBE + INTERVAL_BETWEEN_CUBE),
                z,
            };

            let rotation = if position.is_zero() {
                // this is needed so an object at (0, 0, 0) won't get scaled to zero
                // as Quaternions can effect scale if they're not create correctly
                cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
            } else {
                cgmath::Quaternion::from_axis_angle(
                    position.clone().normalize(),
                    cgmath::Deg(frame as f32),
                )
            };

            let a = (row as f32 / width as f32 * 0.7 + 0.3).powi(2);
            let color = cgmath::vec3(a, 1.0 - a, 0.3);
            let normal = (col as f32 / width as f32 * 0.3 + 0.7).powi(2);

            CubeInstance {
                position,
                rotation,
                color,
                normal,
            }
        })
        .collect();

    instances
        .iter()
        .map(CubeInstance::to_raw)
        .collect::<Vec<_>>()
}

fn create_texture(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    sample_count: u32,
    usage: wgpu::TextureUsage,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let frame_descriptor = &wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: format,
        usage,
        label: None,
    };

    device.create_texture(frame_descriptor)
}

fn create_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    create_texture(
        device,
        sc_desc,
        1,
        wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        format,
    )
}

fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    create_texture(
        device,
        sc_desc,
        SAMPLE_COUNT,
        wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format,
    )
}

fn create_depth_texture(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
) -> wgpu::Texture {
    // COPY_SRC is just for debugging purpose (so that we can write the buffer to a PNG file)
    let usage = wgpu::TextureUsage::OUTPUT_ATTACHMENT
        | wgpu::TextureUsage::SAMPLED
        | wgpu::TextureUsage::COPY_SRC;
    create_texture(device, sc_desc, 1, usage, wgpu::TextureFormat::Depth32Float)
}

fn create_render_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: Option<&wgpu::ShaderModule>,
    rasterization_state_descripor: Option<wgpu::RasterizationStateDescriptor>,
    vertex_buffers: &[wgpu::VertexBufferDescriptor],
    sample_count: u32,
    formats: Vec<wgpu::TextureFormat>,
    depth_stencil_state: Option<wgpu::DepthStencilStateDescriptor>,
) -> wgpu::RenderPipeline {
    let v: Vec<_> = formats
        .iter()
        .map(|format| wgpu::ColorStateDescriptor {
            format: *format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        })
        .collect();

    let mut fragment_stage = None;
    if let Some(fs_mod_ref) = fs_mod {
        fragment_stage = Some(wgpu::ProgrammableStageDescriptor {
            module: fs_mod_ref,
            entry_point: "main",
        });
    }

    let rasterization_state =
        if let Some(rasterization_state_descriptor_ref) = rasterization_state_descripor {
            Some(rasterization_state_descriptor_ref)
        } else {
            Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                ..Default::default()
            })
        };

    // Load shader modules.
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: Some(pipeline_layout),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_mod,
            entry_point: "main",
        },
        fragment_stage,
        rasterization_state,
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &v.as_slice(),
        depth_stencil_state: depth_stencil_state,
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: vertex_buffers,
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
        label: None,
    })
}

fn create_png_texture_and_buffer(
    device: &wgpu::Device,
    width: usize,
    height: usize,
) -> (PNGDimensions, wgpu::Buffer, wgpu::Texture) {
    let png_dimensions = PNGDimensions::new(width, height);
    // The output buffer lets us retrieve the data as an array
    let png_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (png_dimensions.padded_bytes_per_row * png_dimensions.height) as u64,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    // The render pipeline renders data into this texture
    let png_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: png_dimensions.width as u32,
            height: png_dimensions.height as u32,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: None,
    });

    (png_dimensions, png_buffer, png_texture)
}
// The original code is https://github.com/gfx-rs/wgpu-rs/blob/8e4d0015862507027f3a6bd68056c64568d11366/examples/capture/main.rs#L122-L194
async fn create_png(
    png_output_path: &str,
    device: &wgpu::Device,
    output_buffer: &wgpu::Buffer,
    buffer_dimensions: &PNGDimensions,
) {
    // Note that we're not calling `.await` here.
    let buffer_slice = output_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let padded_buffer = buffer_slice.get_mapped_range();

        let mut png_encoder = png::Encoder::new(
            std::fs::File::create(png_output_path).unwrap(),
            buffer_dimensions.width as u32,
            buffer_dimensions.height as u32,
        );
        png_encoder.set_depth(png::BitDepth::Eight);
        png_encoder.set_color(png::ColorType::RGBA);
        png_encoder.set_compression(png::Compression::Fast);
        let mut png_writer = png_encoder
            .write_header()
            .unwrap()
            .into_stream_writer_with_size(buffer_dimensions.unpadded_bytes_per_row);

        // from the padded_buffer we write just the unpadded bytes into the image
        for chunk in padded_buffer.chunks(buffer_dimensions.padded_bytes_per_row) {
            png_writer
                .write(&chunk[..buffer_dimensions.unpadded_bytes_per_row])
                .unwrap();
        }
        png_writer.finish().unwrap();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(padded_buffer);

        output_buffer.unmap();
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("test")
        .build(&event_loop)
        .unwrap();

    // Since main can't be async, we're going to need to block
    let mut state = block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &mut so w have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::R),
                                ..
                            } => {
                                state.frame = 0;
                                state.record = true;
                                println!("Start recording");
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                state.render();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }
    });
}
