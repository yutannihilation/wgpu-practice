use glam::{Quat, Vec3};
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
pub struct Neighborings {
    point: usize,
    edges: Vec<usize>,
    faces: Vec<usize>,
}

#[derive(Debug)]
pub struct Polygon {
    // basic representation
    points: Vec<Vec3>,
    face_indices: Vec<Vec<usize>>,
    edge_indices: Vec<[usize; 2]>,

    // pre-calculated points to do subdivision
    face_points: Vec<Vec3>,
    edge_midpoints: Vec<Vec3>,
    edge_points: Vec<Vec3>,
    new_corners: Vec<Vec3>,

    // if the specified corner is in here, do subdivision
    unsubdivided_corners: Vec<Neighborings>,
}

impl Polygon {
    pub fn triangulate(&self) -> (Vec<Vertex>, Vec<u16>) {
        let mut indices: Vec<u16> = Vec::new();
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

    pub fn precalculate_subdivisions(&mut self) {
        // Calculate face points
        self.face_points = self
            .face_indices
            .iter()
            .map(|face| {
                face.iter()
                    .fold(Vec3::zero(), |sum, &idx| sum + self.points[idx])
                    / face.len() as f32
            })
            .collect();

        // Calculate edge midpoints
        self.edge_midpoints = self
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
                .fold(Vec3::zero(), |sum, &i| sum + self.face_points[i]);

            let edge_point = (sum_of_face_points + self.points[edge[0]] + self.points[edge[1]])
                / (neighboring_faces.len() + 2) as f32;

            edge_points.push(edge_point);
        }

        self.edge_points = edge_points;

        // Calculate new points
        let mut new_corners: Vec<Vec3> = Vec::new();
        let mut unsubdivided_corners: Vec<Neighborings> = Vec::new();

        for (orig_corner_idx, orig_corner) in self.points.iter().enumerate() {
            let neighboring_edges: Vec<usize> =
                self.get_neighboring_edges_to_point(orig_corner_idx);

            let neighboring_faces: Vec<usize> =
                self.get_neighboring_faces_to_point(orig_corner_idx);

            // Should be equal to neighboring_faces.len()
            let n = neighboring_edges.len();

            let avg_face_points = neighboring_faces
                .iter()
                .fold(Vec3::zero(), |sum, &i| sum + self.face_points[i])
                / n as f32;

            let avg_edge_midpoints = neighboring_edges
                .iter()
                .fold(Vec3::zero(), |sum, &i| sum + self.edge_midpoints[i])
                / n as f32;

            let new_corner =
                (avg_face_points + 2.0 * avg_edge_midpoints + (n - 3) as f32 * *orig_corner)
                    / n as f32;

            new_corners.push(new_corner);
            unsubdivided_corners.push(Neighborings {
                point: orig_corner_idx,
                edges: neighboring_edges,
                faces: neighboring_faces,
            })
        }

        self.new_corners = new_corners;
        self.unsubdivided_corners = unsubdivided_corners;
    }

    // An implementation of https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
    //
    // 1. Insert edge points before and after the specified point.
    // 2. Replace the specified point with the face point (on each face).
    // 3. Create a new faces using new points.
    pub fn subdivide(&mut self) {
        if self.unsubdivided_corners.len() == 0 {
            self.precalculate_subdivisions();
        }
        let point_idx = self.unsubdivided_corners.pop().unwrap();

        self.subdivide_at(point_idx);
    }

    fn subdivide_at(&mut self, neighborings: Neighborings) {
        println!("affected edges: {:#?}", neighborings.edges);

        // probably we can search on all the faces every time...?
        println!("affected faces: {:#?}", neighborings.faces);

        // new face would be like this, starting from top-left and goes clockwise order
        //
        // e--c    c: new corner point (this will be replaced with the original point, so use point_idx)
        // |  |    e: edge point
        // f--e    f: face point
        let mut new_faces: Vec<Vec<usize>> =
            vec![vec![neighborings.point; 4]; neighborings.faces.len()];

        // face index to point index
        let mut face_point_indices: HashMap<usize, usize> = HashMap::new();
        // edge index to point index
        let mut sum_edge_midpoints = Vec3::zero();

        // 1. For each face, add a face point.

        for (new_face_idx, &face_idx) in neighborings.faces.iter().enumerate() {
            // Calculate face point and add it to the points list
            let new_point_idx = self.points.len();
            self.points.push(self.face_point(face_idx));

            // Note that we cannot replace the actual face's point here yet since it will be
            // used for detecting the neighboring faces when calculating edge points.

            // The face point will be the last point of the new face
            new_faces[new_face_idx][3] = new_point_idx;
            face_point_indices.insert(face_idx, new_point_idx);
        }

        // 2. Insert edge points before and after the specified point.

        for &edge_idx in neighborings.edges.iter() {
            let edge = &self.edge_indices[edge_idx];

            let point_idx_opposite = if edge[0] == neighborings.point {
                edge[1]
            } else {
                edge[0]
            };

            // Filter the faces neighboring to the edge (= contains both points of the edge)
            let neighboring_faces: Vec<usize> = neighborings
                .faces
                .iter()
                .map(|&i| i) // Not sure why I need to dereference here before filter...
                .filter(|&f| self.face_indices[f].contains(&point_idx_opposite))
                .collect();

            let sum_of_face_points = neighboring_faces.iter().fold(Vec3::zero(), |sum, &i| {
                sum + self.points[face_point_indices[&i]]
            });

            let edge_point = (sum_of_face_points + self.points[edge[0]] + self.points[edge[1]])
                / (neighboring_faces.len() + 2) as f32;

            let new_point_idx = self.points.len();
            self.points.push(edge_point);

            // Since the edge will be modified in the next step, preserve the edge midpoints to use later
            sum_edge_midpoints += self.edge_midpoint(edge_idx);

            for (new_face_idx, &face_idx) in neighborings.faces.iter().enumerate() {
                let face = &mut self.face_indices[face_idx];
                print!(
                    "Adding the edge point of edge {} ({:?}) to face {} ({:?}) -> ",
                    face_idx, face, edge_idx, edge
                );
                let point_local_idx_opposite =
                    face.iter().position(|&idx| idx == point_idx_opposite);
                // Modify only on the face that contains both points of the edge
                if point_local_idx_opposite.is_none() {
                    println!("skip");
                    continue;
                }
                let point_local_idx = face
                    .iter()
                    .position(|&idx| idx == neighborings.point)
                    .unwrap();
                let point_local_idx_opposite = point_local_idx_opposite.unwrap();

                let idx_distance = (point_local_idx as i32 - point_local_idx_opposite as i32).abs();
                // if the index is at the first and last, insert to the front
                let pos = if idx_distance == (face.len() - 1) as i32 {
                    0
                } else {
                    std::cmp::max(point_local_idx, point_local_idx_opposite)
                };
                face.insert(pos, new_point_idx);

                if pos == point_local_idx {
                    // If the new point is inserted before the original point, it will be the first point of the new face
                    new_faces[new_face_idx][0] = new_point_idx;
                } else {
                    // If the new point is inserted after the original point, it will be the third point of the new face
                    new_faces[new_face_idx][2] = new_point_idx;
                }

                // result
                println!(" face {:?}", face);
            }

            // Update the edge
            let mut new_edge = [new_point_idx, point_idx_opposite];
            new_edge.sort();
            {
                let edge = &mut self.edge_indices[edge_idx];
                std::mem::replace(edge, new_edge);
            }
        }

        // 3. Since the old edge is now useless, replace it with new face points.

        for &face_idx in neighborings.faces.iter() {
            let face = &mut self.face_indices[face_idx];
            print!(
                "Replacing the point with face point of face {} ({:?}) -> ",
                face_idx, face
            );

            face.iter_mut().for_each(|i| {
                if *i == neighborings.point {
                    *i = face_point_indices[&face_idx];
                }
            });

            // result
            println!(" face {:?}", face);
        }

        // 4. Create a new corner point and replace the original corner.

        let n = face_point_indices.len();
        let avg_face_points = face_point_indices
            .values()
            .fold(Vec3::zero(), |sum, &i| sum + self.points[i])
            / n as f32;

        let avg_edge_midpoints = sum_edge_midpoints / n as f32;

        {
            // Borrow as mut so that we can replace this directly
            let original_point = &mut self.points[neighborings.point];

            let new_corner_point =
                (avg_face_points + 2.0 * avg_edge_midpoints + (n - 3) as f32 * *original_point)
                    / n as f32;

            std::mem::replace(original_point, new_corner_point);
        }

        // 5. Register the new faces and edges

        for new_face in new_faces.iter() {
            let len = new_face.len();
            for i in 0..len {
                let mut edge = [new_face[i], new_face[(i + 1) % len]];
                edge.sort();
                self.edge_indices.push(edge);
            }
        }

        println!("Generated new_faces: {:?}", new_faces);
        self.face_indices.append(&mut new_faces);
    }

    // Return the indices of neighboring edges to the specified point
    pub fn neighboring_edges(&self, point_idx: usize) -> Vec<usize> {
        let mut result = Vec::new();

        for (i, edge) in self.edge_indices.iter().enumerate() {
            if edge[0] == point_idx || edge[1] == point_idx {
                result.push(i);
            }
        }
        result
    }

    // Return the indices of neighboring faces to the specified point
    pub fn neighboring_faces(&self, point_idx: usize) -> Vec<usize> {
        let mut result = Vec::new();

        for (i, face) in self.face_indices.iter().enumerate() {
            if face.iter().position(|&idx| idx == point_idx).is_some() {
                result.push(i);
            }
        }

        result
    }

    pub fn edge_midpoint(&self, edge_idx: usize) -> Vec3 {
        let e = &self.edge_indices[edge_idx];
        (self.points[e[0]] + self.points[e[1]]) / 2.0
    }

    pub fn face_point(&self, face_idx: usize) -> Vec3 {
        let f = &self.face_indices[face_idx];
        f.iter().fold(Vec3::zero(), |sum, &i| sum + self.points[i]) / f.len() as f32
    }
}

pub fn calculate_initial_cube() -> Polygon {
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

    let len = points.len();
    let mut cube = Polygon {
        points,
        face_indices,
        edge_indices: edge_indices.into_iter().collect(),

        face_points: Vec::new(),
        edge_midpoints: Vec::new(),
        edge_points: Vec::new(),
        new_corners: Vec::new(),

        unsubdivided_corners: Vec::new(),
    };

    cube.precalculate_subdivisions();
    cube
}

fn main() {
    let mut cube = calculate_initial_cube();
    println!("{:?}", cube.face_indices);
    println!("{:?}", cube.edge_indices);
    cube.subdivide();
    println!("{:?}", cube.face_indices);
    println!("{:?}", cube.edge_indices);
}
