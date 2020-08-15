use cgmath::prelude::*;
use cgmath::vec3;

#[cfg_attr(rustfmt, rustfmt_skip)]
const points: [cgmath::Vector3<f32>; 8] = [
    vec3( 1.0,  1.0,  1.0),
    vec3( 1.0,  1.0, -1.0),
    vec3( 1.0, -1.0,  1.0),
    vec3( 1.0, -1.0, -1.0),
    vec3(-1.0,  1.0,  1.0),
    vec3(-1.0,  1.0, -1.0),
    vec3(-1.0, -1.0,  1.0),
    vec3(-1.0, -1.0, -1.0),
];

fn main() {
    let origin = cgmath::Point3::new(0.0, 0.0, 0.0);

    let p1 = cgmath::Point3::new(-1.0, -1.0, 1.0);
    let p2 = cgmath::Point3::new(1.0, -1.0, 1.0);
    let p3 = cgmath::Point3::new(1.0, 1.0, 1.0);

    let v1 = p2 - p1;
    let v2 = p3 - p1;

    let normal = v1.cross(v2);
    let result = if normal.dot(origin - p1) > 0.0 {
        -normal.normalize()
    } else {
        normal.normalize()
    };

    println!("{:?}", result);
    println!("{:?}", normal.dot(origin - p1));
}
