use glam::{Quat, Vec3};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

// Code is from https://github.com/gfx-rs/wgpu-rs/blob/master/examples/cube/main.rs
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    _pos: [f32; 4],
    _normal: [f32; 3],
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

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
pub struct CornerNeighborings {
    corner: usize,
    neighborings: Vec<usize>,
}

#[derive(Debug)]
pub struct Polygon {
    // basic representation
    points: Vec<Vec3>,
    face_indices: Vec<Vec<usize>>,
    edge_indices: Vec<[usize; 2]>,

    // pre-calculated points to do subdivision
    face_point_indices: Vec<usize>,
    edge_midpoint_indices: Vec<usize>,
    new_points: HashMap<usize, Vec3>,

    // if the specified corner is in here, do subdivision
    corner_and_neighborings: Vec<CornerNeighborings>,
    pub n_corners: usize,

    // default: 2
    sharpness: Option<f32>,
}

impl Polygon {
    pub fn triangulate(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut indices: Vec<u32> = Vec::new();
        let mut vertices: Vec<Vertex> = Vec::new();

        for f in self.face_indices.iter() {
            // As all points belongs to the same surface, the normal should be the same.
            // Calculate a normal vector with arbitrary 3 points within the face for now.
            // (I might need to revisit here when I want to calculate the average of the
            // neighboring surfaces...)
            let i0 = f[0];
            let p0 = self.points[i0];
            let p1 = self.points[f[1]];
            let p2 = self.points[f[2]];

            let v0 = p1 - p0;
            let v1 = p2 - p0;

            // Calculate the normal vector
            let normal_unnormalized = v0.cross(v1);

            // Normal vector might goes outer or inner. Make sure it goes to the opposite side of (0, 0, 0).
            let reverse = normal_unnormalized.dot(Vec3::zero() - p1) > 0.0;
            let normal = if reverse {
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
                // indices needs to be added to make sure the triangle is clockwise
                if !reverse {
                    indices.push(indices_offset as u32);
                    indices.push((indices_offset + i) as u32);
                    indices.push((indices_offset + (i + 1) % len) as u32);
                } else {
                    indices.push((indices_offset + (i + 1) % len) as u32);
                    indices.push((indices_offset + i) as u32);
                    indices.push(indices_offset as u32);
                }
            }
        }

        (vertices, indices)
    }

    fn get_neighboring_edges_to_point(&self, idx: usize) -> Vec<usize> {
        self.edge_indices
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.contains(&idx))
            .map(|(i, _)| i)
            .collect()
    }
    fn get_neighboring_faces_to_point(&self, idx: usize) -> Vec<usize> {
        self.face_indices
            .iter()
            .enumerate()
            .filter(|(_, face)| face.contains(&idx))
            .map(|(i, _)| i)
            .collect()
    }
    fn get_neighboring_faces_to_edge(&self, idx0: usize, idx1: usize) -> Vec<usize> {
        self.face_indices
            .iter()
            .enumerate()
            // narrow to the faces that contains both points of the edge
            .filter(|(_, face)| face.contains(&idx0))
            .filter(|(_, face)| face.contains(&idx1))
            .map(|(i, _)| i)
            .collect()
    }

    fn find_edge_index(&self, point_idx0: usize, point_idx1: usize) -> usize {
        let edge = if point_idx0 < point_idx1 {
            [point_idx0, point_idx1]
        } else {
            [point_idx1, point_idx0]
        };

        match self.edge_indices.iter().position(|p| *p == edge) {
            Some(v) => v,
            None => {
                // println!("foo");
                panic!("foo")
            }
        }
    }

    pub fn precalculate_subdivisions(&mut self) {
        self.n_corners = self.points.len();

        // Calculate face points
        let mut face_points: Vec<Vec3> = self
            .face_indices
            .iter()
            .map(|face| {
                face.iter()
                    .fold(Vec3::zero(), |sum, &idx| sum + self.points[idx])
                    / face.len() as f32
            })
            .collect();

        // Calculate edge midpoints
        let mut edge_midpoints: Vec<Vec3> = self
            .edge_indices
            .iter()
            .map(|edge| (self.points[edge[0]] + self.points[edge[1]]) / 2.0)
            .collect();

        // Calculate edge points
        let mut edge_points: Vec<Vec3> = Vec::new();
        for edge in self.edge_indices.iter() {
            let neighboring_faces: Vec<usize> =
                self.get_neighboring_faces_to_edge(edge[0], edge[1]);

            let sum_of_face_points = neighboring_faces
                .iter()
                .fold(Vec3::zero(), |sum, &i| sum + face_points[i]);

            let edge_point = (sum_of_face_points + self.points[edge[0]] + self.points[edge[1]])
                / (neighboring_faces.len() as f32 + self.sharpness.unwrap_or(2.0));

            edge_points.push(edge_point);
        }

        // Calculate new points
        let mut new_corners: Vec<Vec3> = Vec::new();

        for (orig_corner_idx, orig_corner) in self.points.iter().enumerate() {
            let neighboring_edges: Vec<usize> =
                self.get_neighboring_edges_to_point(orig_corner_idx);

            let neighboring_faces: Vec<usize> =
                self.get_neighboring_faces_to_point(orig_corner_idx);

            // Should be equal to neighboring_faces.len()
            let n = neighboring_edges.len();

            let avg_face_points = neighboring_faces
                .iter()
                .fold(Vec3::zero(), |sum, &i| sum + face_points[i])
                / n as f32;

            let avg_edge_midpoints = neighboring_edges
                .iter()
                .fold(Vec3::zero(), |sum, &i| sum + edge_midpoints[i])
                / n as f32;

            let new_corner =
                (avg_face_points + 2.0 * avg_edge_midpoints + (n - 3) as f32 * *orig_corner)
                    / n as f32;

            new_corners.push(new_corner);
        }

        let mut point_indice_last = self.points.len();

        self.new_points = HashMap::new();
        for point_idx in 0..point_indice_last {
            self.new_points.insert(point_idx, new_corners[point_idx]);
        }

        let face_point_indice_last = point_indice_last + face_points.len();
        self.face_point_indices = (point_indice_last..face_point_indice_last).collect();
        self.points.append(&mut face_points);
        point_indice_last = face_point_indice_last;

        let edge_midpoint_indice_last = point_indice_last + edge_midpoints.len();
        self.edge_midpoint_indices = (point_indice_last..edge_midpoint_indice_last).collect();
        self.points.append(&mut edge_midpoints);

        for (edge_idx, point_idx) in (point_indice_last..edge_midpoint_indice_last).enumerate() {
            self.new_points.insert(point_idx, edge_points[edge_idx]);
        }

        // Quartering each surfaces, so that the points can be replaced easily
        //
        // x-----x      x-----x
        // |     |      |  |  |
        // |     |  ->  x--x--x
        // |     |      |  |  |
        // x-----x      x--x--x

        let mut new_faces: Vec<Vec<usize>> = Vec::new();

        let mut new_edges: HashSet<[usize; 2]> = HashSet::new();
        let mut neighboring_edge_midpoints_map: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (face_idx, face) in self.face_indices.iter().enumerate() {
            // For each face, create a new subface
            //
            // x-----x      x-----x  f: face point
            // |     |      |     |  1: edge point (right)
            // |     |  ->  |  f--1  2: edge point (left)
            // |     |      |  |  |
            // x-----o      x--2--o
            let n_points = face.len();
            let face_point = self.face_point_indices[face_idx];
            for (point_idx, point) in face.iter().enumerate() {
                let edge_right =
                    self.find_edge_index(face[point_idx], face[(point_idx + 1) % n_points]);
                let edge_midpoint_right = self.edge_midpoint_indices[edge_right];

                let edge_left = self.find_edge_index(
                    face[(point_idx as i32 - 1) as usize % n_points], // Note: temporarily convert to i32 as usize cannot be negative
                    face[point_idx],
                );
                let edge_midpoint_left = self.edge_midpoint_indices[edge_left];

                let new_face = vec![face_point, edge_midpoint_right, *point, edge_midpoint_left];

                for i in 0..4 {
                    let p0 = new_face[i];
                    let p1 = new_face[(i + 1) % 4];
                    new_edges.insert(if p0 < p1 { [p0, p1] } else { [p1, p0] });
                }
                new_faces.push(new_face);

                let neighboring_edge_midpoints = match neighboring_edge_midpoints_map.entry(*point)
                {
                    Entry::Occupied(o) => o.into_mut(),
                    Entry::Vacant(v) => v.insert(HashSet::new()),
                };

                neighboring_edge_midpoints.insert(edge_midpoint_right);
                neighboring_edge_midpoints.insert(edge_midpoint_left);
            }
        }

        // println!("new faces: {:?}", new_faces);
        // println!("new edges: {:?}", new_edges);
        // println!("neighborings: {:?}", neighboring_edge_midpoints_map);

        self.face_indices = new_faces;
        self.edge_indices = new_edges.into_iter().collect();
        self.corner_and_neighborings = neighboring_edge_midpoints_map
            .into_iter()
            .map(|(k, v)| CornerNeighborings {
                corner: k,
                neighborings: v.into_iter().collect::<Vec<usize>>(),
            })
            .collect();
    }

    // An implementation of https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
    //
    // 1. Replace edge midpoints with edge points.
    // 2. Replace the specified point with the new corner point.
    pub fn subdivide(&mut self) {
        if self.corner_and_neighborings.len() == 0 {
            // println!("Re-calculating");
            // println!("Current status: -----------------------------------------------------\n\n{:?}\n--------------------------------------------------------\n", self);
            self.precalculate_subdivisions();
            println!("Points: {:?}", self.points.len());
        }
        let target = self.corner_and_neighborings.pop().unwrap();

        // println!("affected corner and neighborings: {:#?}", target);

        self.points[target.corner] = self.new_points[&target.corner];

        for n in target.neighborings.iter() {
            self.points[*n] = self.new_points[n];
        }
    }
}

pub fn calculate_initial_cube(sharpness: Option<f32>) -> Polygon {
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

    let points: Vec<Vec3> = points_int
        .iter()
        .map(|p| Vec3::new(p.0 as f32, p.1 as f32, p.2 as f32))
        .collect();

    let mut face_indices: Vec<Vec<usize>> = Vec::new();
    let mut edge_indices: HashSet<[usize; 2]> = HashSet::new();

    for f in faces_int.iter() {
        let mut face_indices_inner: Vec<usize> = Vec::new();
        for v in f.iter() {
            let idx = points_int.iter().position(|p| *p == *v).unwrap();
            face_indices_inner.push(idx);
        }

        face_indices.push(face_indices_inner.clone());

        let len = face_indices_inner.len();
        for i in 0..len {
            let mut edge = [face_indices_inner[i], face_indices_inner[(i + 1) % len]];
            edge.sort();
            edge_indices.insert(edge);
        }
    }

    let n_points = points.len();
    let mut cube = Polygon {
        points,
        face_indices,
        edge_indices: edge_indices.into_iter().collect(),

        face_point_indices: Vec::new(),
        edge_midpoint_indices: Vec::new(),
        new_points: HashMap::new(),

        corner_and_neighborings: Vec::new(),
        n_corners: n_points,

        sharpness,
    };

    cube.precalculate_subdivisions();
    cube
}

pub fn create_plane(size: i32) -> (Vec<Vertex>, Vec<u32>) {
    let size = size as f32;

    let vertex_data = [
        Vertex {
            _pos: [size, -size, 0.0, 1.0],
            _normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            _pos: [size, size, 0.0, 1.0],
            _normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            _pos: [-size, -size, 0.0, 1.0],
            _normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            _pos: [-size, size, 0.0, 1.0],
            _normal: [0.0, 0.0, 1.0],
        },
    ];

    let index_data = &[0, 1, 2, 2, 1, 3];

    (vertex_data.to_vec(), index_data.to_vec())
}
