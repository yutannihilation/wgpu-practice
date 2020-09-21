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

// sample count for MSAA
const SAMPLE_COUNT: u32 = 4;

const IMAGE_DIR: &str = "img";

const BG_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.22,
    b: 0.3,
    a: 1.0,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    position: [f32; 4],
}

//   4-1
//   |/|
//   2-3
//
// #[rustfmt::skip]
// const VERTICES: &[Vertex] = &[
//     Vertex { position: [ 1.0,  1.0], },
//     Vertex { position: [-1.0, -1.0], },
//     Vertex { position: [ 1.0, -1.0], },

//     Vertex { position: [ 1.0,  1.0], },
//     Vertex { position: [-1.0, -1.0], },
//     Vertex { position: [-1.0,  1.0], },
// ];

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct View {
    position: [f32; 4],
    proj: [[f32; 4]; 4],
}

fn generate_view(aspect_ratio: f32, frame: u32) -> View {
    let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 0.5, 200.0);
    let rot1 = (frame + 400) as f32 / 200.0;

    let rot2_max = std::f32::consts::PI / 4.0;
    let rot2 = rot2_max;
    // * ((3001 - std::cmp::min(frame, 3000)) as f32 / 3000.0).powi(3);

    let distance = 2.0f32 + (frame as f32 / 100.0);
    let eye = cgmath::Point3::new(
        distance * rot1.sin() * rot2.sin(),
        distance * rot1.cos() * rot2.sin(),
        distance * rot2.cos(),
    );
    let mx_view =
        cgmath::Matrix4::look_at(eye, cgmath::Point3::origin(), cgmath::Vector3::unit_z());

    View {
        position: eye.to_homogeneous().into(),
        proj: (OPENGL_TO_WGPU_MATRIX * mx_projection * mx_view).into(),
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
    // Though we only need vec3, due to uniforms requiring 16 byte (4 float) spacing,
    // use vec4 to align with the requirement
    view_proj: cgmath::Matrix4<f32>,
    position: cgmath::Vector4<f32>,
    color: cgmath::Vector3<f32>,
}

unsafe impl bytemuck::Pod for LightRaw {}
unsafe impl bytemuck::Zeroable for LightRaw {}

impl Light {
    fn new() -> Self {
        Self {
            position: (3.0, 1.0, 100.0).into(),
            color: (1.0, 1.0, 1.0).into(),
        }
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
            color: self.color,
        }
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,

    view_bind_group_layout: wgpu::BindGroupLayout,
    light_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,

    staging_texture: wgpu::Texture,

    // depth
    depth_texture: wgpu::Texture,

    // Texture for MASS
    multisample_texture: wgpu::Texture,
    multisample_png_texture: wgpu::Texture, // a texture for PNG has a different TextureFormat, so we need another multisampled texture than others

    // Texture for writing out as PNG
    png_texture: wgpu::Texture,
    png_buffer: wgpu::Buffer,
    png_dimensions: PNGDimensions,

    size: winit::dpi::PhysicalSize<u32>,

    output_dir: std::path::PathBuf,

    frame: u32,
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

        // View ------------------------------------------------------------------------------------------------------------
        let view_size = std::mem::size_of::<View>();
        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(view_size as _),
                    },
                    count: None,
                }],
            });

        // Light ------------------------------------------------------------------------------------------------------------
        let light = Light::new().to_raw();
        let light_size = std::mem::size_of::<LightRaw>() as u64;

        // We'll want to update our lights position, so we use COPY_DST
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&[light]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            label: None,
        });

        let light_bind_group_layout =
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

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // Render pipeline ------------------------------------------------------------------------------------------------------------

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&view_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
                label: None,
            });

        let render_pipeline = create_render_pipeline(
            &device,
            &render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.vert.spv")),
            Some(&device.create_shader_module(wgpu::include_spirv!("shaders/shader.frag.spv"))),
            None, // use the default rasterization_state_descripor
            &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float4],
            }],
            SAMPLE_COUNT,
            vec![sc_desc.format, wgpu::TextureFormat::Rgba8UnormSrgb],
            Some(depth_stencil_state.clone()),
        );

        // Texture to draw the unmodified version
        let staging_texture = create_framebuffer(&device, &sc_desc, sc_desc.format);

        // MSAA --------------------------------------------------------------------------------------------------------
        let multisample_texture =
            create_multisampled_framebuffer(&device, &sc_desc, sc_desc.format);

        let multisample_png_texture =
            create_multisampled_framebuffer(&device, &sc_desc, wgpu::TextureFormat::Rgba8UnormSrgb);

        // Depth texture ----------------------------------------------------------------------------------------------------
        let depth_texture = create_depth_texture(&device, &sc_desc);

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

            view_bind_group_layout,
            light_bind_group,
            render_pipeline,

            staging_texture,

            depth_texture,

            multisample_texture,
            multisample_png_texture,

            png_texture,
            png_buffer,
            png_dimensions,

            size,

            output_dir,

            frame: 0,
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
        self.multisample_texture =
            create_multisampled_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format);
        self.multisample_png_texture = create_multisampled_framebuffer(
            &self.device,
            &self.sc_desc,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        self.depth_texture = create_depth_texture(&self.device, &self.sc_desc);

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

        let png_texture_view = self
            .png_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture_view = self
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let multisample_texture_view = self
            .multisample_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let multisample_png_texture_view = self
            .multisample_png_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let view_uniforms = generate_view(
            self.sc_desc.width as f32 / self.sc_desc.height as f32,
            self.frame,
        );
        let view_uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[view_uniforms]),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

        let view_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_uniform_buf.as_entire_binding(),
            }],
            label: None,
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &multisample_texture_view,
                        resolve_target: Some(&frame.output.view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(BG_COLOR),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &multisample_png_texture_view,
                        resolve_target: Some(&png_texture_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(BG_COLOR),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &depth_texture_view,
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
            render_pass.set_bind_group(0, &view_bind_group, &[]);
            render_pass.set_bind_group(1, &self.light_bind_group, &[]);
            render_pass.draw(0..1, 0..1);
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
    let usage = wgpu::TextureUsage::OUTPUT_ATTACHMENT
        | wgpu::TextureUsage::SAMPLED
        | wgpu::TextureUsage::COPY_SRC;
    create_texture(
        device,
        sc_desc,
        SAMPLE_COUNT,
        usage,
        wgpu::TextureFormat::Depth32Float,
    )
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
