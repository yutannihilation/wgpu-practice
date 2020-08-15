use glam::{Quat, Vec3};
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

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
        Self(v.x() as i32, v.y() as i32, v.z() as i32)
    }
}

#[derive(Debug)]
struct Polygon {
    points: Vec<Vec3>,
    face_indices: Vec<Vec<usize>>,
    edge_indices: Vec<Vec<[usize; 2]>>,
}

impl Polygon {
    fn triangulate(&self) -> (&Vec<Vec3>, Vec<u16>) {
        let mut indices: Vec<u16> = Vec::new();

        for f in self.face_indices.iter() {
            let len = f.len();
            for i in 1..(len - 1) {
                indices.push(f[0] as u16);
                indices.push(f[i] as u16);
                indices.push(f[(i + 1) % len] as u16);
            }
        }

        (&self.points, indices)
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
    let mut points = Vec::new();
    let mut points_set = HashSet::new();

    for f in faces.iter() {
        let mut faces_int_inner: Vec<IntVec3> = Vec::new();
        for &v in f.iter() {
            points.push(v);
            let p = IntVec3::new(v);
            faces_int_inner.push(p.clone());
            points_set.insert(p);
        }
        faces_int.push(faces_int_inner);
    }

    let mut points_int: Vec<IntVec3> = points_set.into_iter().collect();

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
