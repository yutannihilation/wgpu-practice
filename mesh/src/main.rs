use glam::{Quat, Vec3};
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

// Code is from https://github.com/gfx-rs/wgpu-rs/blob/master/examples/cube/main.rs
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Vertex {
    _pos: [f32; 4],
    _normal: [f32; 3],
}

// unsafe impl bytemuck::Pod for Vertex {}
// unsafe impl bytemuck::Zeroable for Vertex {}

#[derive(Debug)]
struct Triangle {
    points: [Vec3; 3],
    normal: Vec3,
}

impl Triangle {
    fn new(p0: Vec3, p1: Vec3, p2: Vec3) -> Self {
        let v0 = p1 - p0;
        let v1 = p2 - p0;

        // Calculate the normal vector
        let normal_unnormalized = v0.cross(v1);

        // Normal vector might goes outer or inner. Make sure it goes to the opposite side of (0, 0, 0).
        let normal = if normal_unnormalized.dot(Vec3::zero() - p1) > 0.0 {
            -normal_unnormalized.normalize()
        } else {
            normal_unnormalized.normalize()
        };

        Self {
            points: [p0, p1, p2],
            normal,
        }
    }
}

#[derive(Debug)]
struct Edge {
    points: [Vec3; 2],
}

impl Edge {
    fn edge_point(&self) -> Vec3 {
        (self.points[0] + self.points[1]) / 2.0
    }
}

#[derive(Debug)]
struct Face {
    points: VecDeque<Vec3>,
}

impl Face {
    fn face_point(&self) -> Vec3 {
        self.points.iter().fold(Vec3::zero(), |sum, p| sum + *p) / self.points.len() as f32
    }

    fn edges(&self) -> Vec<Edge> {
        let len = self.points.len();
        (0..len)
            .map(|i| Edge {
                points: [self.points[i], self.points[(i + 1) % len]],
            })
            .collect()
    }

    fn triangulate(&self) -> Vec<Triangle> {
        let len = self.points.len();
        (1..(len - 1))
            .map(|i| Triangle::new(self.points[0], self.points[i], self.points[(i + 1) % len]))
            .collect()
    }
}

// Since Vec3 (or float generally) doesn't support Hash, we need an integer version
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct IntVec3(i32, i32, i32);

impl IntVec3 {
    fn new(v: Vec3) -> Self {
        Self(
            v.x().round() as i32,
            v.y().round() as i32,
            v.z().round() as i32,
        )
    }
}

#[derive(Debug)]
struct Polygon {
    points: Vec<Vec3>,
    face_indices: Vec<Vec<usize>>,
    edge_indices: Vec<Vec<[usize; 2]>>,
}

impl Polygon {
    fn triangulate(&self) -> (Vec<Vertex>, Vec<u16>) {
        let mut indices: Vec<u16> = Vec::new();
        let mut vertices: Vec<Vertex> = Vec::new();

        for f in self.face_indices.iter() {
            // As all points belongs to the same surface, the normal should be the same.
            // Calculate a normal vector with arbitrary 3 points within the face for now.
            // (I might need to revisit here when I want to calculate the average of the
            // neibouring surfaces...)
            let i0 = f[0];
            let p0 = self.points[i0];
            let p1 = self.points[f[1]];
            let p2 = self.points[f[2]];

            let v0 = p1 - p0;
            let v1 = p2 - p0;

            // Calculate the normal vector
            let normal_unnormalized = v0.cross(v1);

            // Normal vector might goes outer or inner. Make sure it goes to the opposite side of (0, 0, 0).
            let normal = if normal_unnormalized.dot(Vec3::zero() - p1) > 0.0 {
                -normal_unnormalized.normalize()
            } else {
                normal_unnormalized.normalize()
            };

            let len = f.len();
            let indices_offset = vertices.len();

            // Create verticles
            for i in 0..len {
                let p = self.points[f[i]];
                vertices.push(Vertex {
                    _pos: [p.x(), p.y(), p.z(), 1.0],
                    _normal: [normal.x(), normal.y(), normal.z()],
                })
            }

            // Choose combination of three points
            for i in 1..(len - 1) {
                indices.push(indices_offset as u16);
                indices.push((indices_offset + i) as u16);
                indices.push((indices_offset + (i + 1) % len) as u16);
            }
        }

        (vertices, indices)
    }
}

fn calculate_initial_cube() -> Polygon {
    let directions = [1.0f32, -1.0];
    let axes = [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()];

    let faces: Vec<Vec<Vec3>> = directions
        .iter()
        .flat_map(|&dir| {
            axes.iter().map(move |&axis| {
                let base = dir * Vec3::one();
                vec![
                    base.clone(),
                    Quat::from_axis_angle(axis, 90.0_f32.to_radians()) * base,
                    Quat::from_axis_angle(axis, 180.0_f32.to_radians()) * base,
                    Quat::from_axis_angle(axis, 270.0_f32.to_radians()) * base,
                ]
            })
        })
        .collect();

    let mut faces_int: Vec<Vec<IntVec3>> = Vec::new();
    let mut points_tmp = Vec::new();
    let mut points_set = HashSet::new();

    for f in faces.iter() {
        let mut faces_int_inner: Vec<IntVec3> = Vec::new();
        for &v in f.iter() {
            points_tmp.push(v);
            let p = IntVec3::new(v);
            faces_int_inner.push(p.clone());
            points_set.insert(p);
        }
        faces_int.push(faces_int_inner);
    }

    let points_int: Vec<IntVec3> = points_set.into_iter().collect();

    let points = points_int
        .iter()
        .map(|p| Vec3::new(p.0 as f32, p.1 as f32, p.2 as f32))
        .collect();

    let mut face_indices: Vec<Vec<usize>> = Vec::new();
    let mut edge_indices: Vec<Vec<[usize; 2]>> = Vec::new();

    for f in faces_int.iter() {
        let mut face_indices_inner: Vec<usize> = Vec::new();
        for v in f.iter() {
            let idx = points_int.iter().position(|p| *p == *v).unwrap();
            face_indices_inner.push(idx);
        }

        face_indices.push(face_indices_inner.clone());

        let len = face_indices_inner.len();
        let edge_indices_inner: Vec<[usize; 2]> = (0..len)
            .map(|i| [face_indices_inner[i], face_indices_inner[(i + 1) % len]])
            .collect();
        edge_indices.push(edge_indices_inner);
    }

    Polygon {
        points,
        face_indices,
        edge_indices,
    }
}

fn main() {
    println!("{:#?}", calculate_initial_cube().triangulate());
}
