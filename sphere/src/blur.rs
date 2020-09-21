use bytemuck::{Pod, Zeroable};

// Simple vertex to draw the texture identically as the original
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlurVertex {
    position: [f32; 2],
}

//   4-1
//   |/|
//   2-3
//
#[rustfmt::skip]
pub const BLUR_VERTICES: &[BlurVertex] = &[
    BlurVertex { position: [ 1.0,  1.0], },
    BlurVertex { position: [-1.0, -1.0], },
    BlurVertex { position: [ 1.0, -1.0], },

    BlurVertex { position: [ 1.0,  1.0], },
    BlurVertex { position: [-1.0,  1.0], },
    BlurVertex { position: [-1.0, -1.0], },
];

// Parameters for gaussian blur;
// As gaussian blur is done horizontally and vertically repeadedly,
// we need a flag to flip the orientation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlurUniforms {
    horizontal: u32,
    _padding: [u32; 3],
}

impl BlurUniforms {
    pub fn new() -> Self {
        Self {
            horizontal: 1,
            _padding: [0, 0, 0],
        }
    }

    pub fn flip(&mut self) {
        self.horizontal = (self.horizontal + 1) % 2;
    }
}

// Parameters for blending the original texture and the blurred texture
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlendUniforms {
    exposure: f32,
    gamma: f32,
    _padding: [u32; 2],
}

impl BlendUniforms {
    pub fn new(exposure: f32, gamma: f32) -> Self {
        Self {
            exposure,
            gamma,
            _padding: [0, 0],
        }
    }
}
