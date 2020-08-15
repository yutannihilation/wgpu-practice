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
    fn new_cube_face(base: Vec3, axis: Vec3) -> Self {
        let mut points = VecDeque::new();

        points.push_back(base);
        points.push_back(Quat::from_axis_angle(axis, 90.0_f32.to_radians()) * base);
        points.push_back(Quat::from_axis_angle(axis, 180.0_f32.to_radians()) * base);
        points.push_back(Quat::from_axis_angle(axis, 270.0_f32.to_radians()) * base);

        Face { points: points }
    }

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

#[derive(Debug, Hash, Eq, PartialEq)]
struct IntVec3(i32, i32, i32);

impl IntVec3 {
    fn new(v: &Vec3) -> Self {
        Self(v.x() as i32, v.y() as i32, v.z() as i32)
    }
}

#[derive(Debug)]
struct Polygon {
    faces: Vec<Face>,
}

impl Polygon {
    fn cube() -> Self {
        let faces = vec![
            Face::new_cube_face(Vec3::one(), Vec3::unit_x()),
            Face::new_cube_face(Vec3::one(), Vec3::unit_y()),
            Face::new_cube_face(Vec3::one(), Vec3::unit_z()),
            Face::new_cube_face(-Vec3::one(), Vec3::unit_x()),
            Face::new_cube_face(-Vec3::one(), Vec3::unit_y()),
            Face::new_cube_face(-Vec3::one(), Vec3::unit_z()),
        ];

        let mut points: HashSet<IntVec3> = HashSet::new();

        faces.iter().for_each(|f| {
            f.points.iter().for_each(|p| {
                let p_ = IntVec3::new(p);
                points.insert(p_);
            })
        });

        println!("{:#?}", points);
        println!("{:#?}", points.len());

        Polygon { faces }
    }

    fn triangulate(&self) -> Vec<Triangle> {
        self.faces.iter().flat_map(|f| f.triangulate()).collect()
    }
}

fn main() {
    let face = Face::new_cube_face(Vec3::one(), glam::Vec3::unit_z());

    println!("{:#?}", face);
    println!("{:#?}", face.face_point());
    println!("{:#?}", face.edges());
    println!("{:#?}", face.triangulate());

    Polygon::cube();
}
