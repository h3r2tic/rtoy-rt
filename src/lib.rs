use rendertoy::*;

#[macro_use]
extern crate static_assertions;
#[macro_use]
extern crate snoozy_macros;
#[macro_use]
extern crate abomonation_derive;

use bvh::{
    aabb::AABB,
    bvh::{BVHNode, BVH},
};
use std::convert::TryFrom;

#[derive(Clone, Copy, Abomonation)]
#[repr(C)]
pub struct GpuBlBvhNode {
    packed: (u32, u32, u32, u32),
}

// `f32_to_f16` from the `half`crate, with a different rounding behavior:
// Round strictly towards smaller values (never a positive offset).
pub fn f32_to_f16_negbias(value: f32) -> u16 {
    // Convert to raw bytes
    let x: u32 = unsafe { core::mem::transmute(value) };

    // Check for signed zero
    if x & 0x7FFFFFFFu32 == 0 {
        return (x >> 16) as u16;
    }

    // Extract IEEE754 components
    let sign = x & 0x80000000u32;
    let exp = x & 0x7F800000u32;
    let man = x & 0x007FFFFFu32;

    // Subnormals will underflow, so return signed zero
    if exp == 0 {
        return (sign >> 16) as u16;
    }

    // Check for all exponent bits being set, which is Infinity or NaN
    if exp == 0x7F800000u32 {
        // A mantissa of zero is a signed Infinity
        if man == 0 {
            return ((sign >> 16) | 0x7C00u32) as u16;
        }
        // Otherwise, this is NaN
        return ((sign >> 16) | 0x7E00u32) as u16;
    }

    // The number is normalized, start assembling half precision version
    let half_sign = sign >> 16;
    // Unbias the exponent, then bias for half precision
    let unbiased_exp = ((exp >> 23) as i32) - 127;
    let half_exp = unbiased_exp + 15;

    // Check for exponent overflow, return +infinity
    if half_exp >= 0x1F {
        return (half_sign | 0x7C00u32) as u16;
    }

    // Check for underflow
    if half_exp <= 0 {
        // Check mantissa for what we can do
        if 14 - half_exp > 24 {
            // No rounding possibility, so this is a full underflow, return signed zero
            return half_sign as u16;
        }
        // Don't forget about hidden leading mantissa bit when assembling mantissa
        let man = man | 0x00800000u32;
        let mut half_man = man >> (14 - half_exp);
        // Check for rounding
        if (man >> (13 - half_exp)) & 0x1u32 != 0 {
            half_man += 1;
        }
        // No exponent for subnormals
        return (half_sign | half_man) as u16;
    }

    // Rebias the exponent
    let half_exp = (half_exp as u32) << 10;
    let half_man = man >> 13;

    if sign != 0 {
        ((half_sign | half_exp | half_man) + 1) as u16
    } else {
        (half_sign | half_exp | half_man) as u16
    }
}

fn pack_bbox_extent_9e5(extent: Vector3) -> u32 {
    pack_rgb9e5_roundup(extent.x, extent.y, extent.z)
}

fn pack_gpu_bvh_node(node: BlBvhNode) -> GpuBlBvhNode {
    let bmin = (
        half::f16::from_bits(f32_to_f16_negbias(node.bbox_min.x)),
        half::f16::from_bits(f32_to_f16_negbias(node.bbox_min.y)),
        half::f16::from_bits(f32_to_f16_negbias(node.bbox_min.z)),
    );

    let box_extent_packed = {
        // The fp16 was rounded-down, so extent will be larger than for fp32
        let extent = node.bbox_max - Point3::new(bmin.0.to_f32(), bmin.1.to_f32(), bmin.2.to_f32());

        pack_bbox_extent_9e5(extent)
    };

    assert!(node.exit_idx < (1u32 << 24));
    assert!(node.prim_idx == std::u32::MAX || node.prim_idx < (1u32 << 24));

    GpuBlBvhNode {
        packed: (
            box_extent_packed,
            ((bmin.0.to_bits() as u32) << 16) | (bmin.1.to_bits() as u32),
            ((bmin.2.to_bits() as u32) << 16) | ((node.prim_idx >> 8) & 0xffff),
            ((node.prim_idx & 0xff) << 24) | node.exit_idx,
        ),
    }
}

pub struct BlBvhNode {
    bbox_min: Point3,
    exit_idx: u32,
    bbox_max: Point3,
    prim_idx: u32,
}

impl BlBvhNode {
    fn new_leaf(bbox_min: Point3, bbox_max: Point3, prim_idx: usize) -> Self {
        Self {
            bbox_min,
            exit_idx: 0,
            bbox_max,
            prim_idx: prim_idx as u32,
        }
    }

    fn new_interior(bbox_min: Point3, bbox_max: Point3) -> Self {
        Self {
            bbox_min,
            exit_idx: 0,
            bbox_max,
            prim_idx: std::u32::MAX,
        }
    }

    fn set_exit_idx(&mut self, idx: usize) {
        self.exit_idx = idx as u32;
    }

    fn get_exit_idx(&mut self) -> usize {
        self.exit_idx as usize
    }
}

#[derive(Clone, Copy, Abomonation)]
#[repr(C)]
pub struct GpuTriangle {
    v: (f32, f32, f32),
    e0_e1: (u32, u32, u32),
}

assert_eq_size!(GpuTriangle, [u8; 6 * 4]);

fn convert_bl_bvh<BoxOrderFn>(
    node: usize,
    nbox: &AABB,
    nodes: &[BVHNode],
    are_boxes_correctly_ordered: &BoxOrderFn,
    res: &mut Vec<BlBvhNode>,
) where
    BoxOrderFn: Fn(&AABB, &AABB) -> bool,
{
    let initial_node_count = res.len();
    let n = &nodes[node];

    let node_res_idx = if node != 0 {
        res.push(if let BVHNode::Node { .. } = n {
            BlBvhNode::new_interior(nbox.min, nbox.max)
        } else {
            BlBvhNode::new_leaf(
                nbox.min,
                nbox.max,
                n.shape_index().expect("bvh leaf shape index"),
            )
        });
        Some(initial_node_count)
    } else {
        None
    };

    if let BVHNode::Node { .. } = n {
        let boxes = [&n.child_l_aabb(), &n.child_r_aabb()];
        let indices = [n.child_l(), n.child_r()];

        let (first, second) = if are_boxes_correctly_ordered(boxes[0], boxes[1]) {
            (0, 1)
        } else {
            (1, 0)
        };

        convert_bl_bvh(
            indices[first],
            &boxes[first],
            nodes,
            are_boxes_correctly_ordered,
            res,
        );
        convert_bl_bvh(
            indices[second],
            &boxes[second],
            nodes,
            are_boxes_correctly_ordered,
            res,
        );
    }

    if let Some(node_res_idx) = node_res_idx {
        let index_after_subtree = res.len();
        res[node_res_idx].set_exit_idx(index_after_subtree);
    } else {
        // We are back at the root node. Go and change exit pointers to be relative,
        for (i, node) in res.iter_mut().enumerate().skip(initial_node_count) {
            let idx = node.get_exit_idx();
            node.set_exit_idx(idx - i);
        }
    }
}

macro_rules! ordered_flatten_bl_bvh {
    ($order: expr, $bvh:ident, $bvh_nodes:ident) => {{
        convert_bl_bvh(
            0,
            &AABB::default(),
            $bvh.nodes.as_slice(),
            &$order,
            &mut $bvh_nodes,
        );
    }};
}

#[derive(Clone, Abomonation)]
pub struct GpuBlBvh {
    nodes: Vec<GpuBlBvhNode>,
    triangles: Vec<GpuTriangle>,
    aabb: ([f32; 3], [f32; 3]),
}

#[snoozy]
pub fn build_gpu_bvh(ctx: &mut Context, mesh: &SnoozyRef<TriangleMesh>) -> Result<GpuBlBvh> {
    let mesh = ctx.get(mesh)?;
    let aabbs: Vec<AABB> = mesh
        .indices
        .chunks(3)
        .map(|t| {
            AABB::empty()
                .grow(&Point3::from(mesh.positions[t[0] as usize]))
                .grow(&Point3::from(mesh.positions[t[1] as usize]))
                .grow(&Point3::from(mesh.positions[t[2] as usize]))
        })
        .collect();

    //let time0 = std::time::Instant::now();
    let bvh = BVH::build(&aabbs);
    //println!("BVH built in {:?}", time0.elapsed());

    let orderings = (
        |a: &AABB, b: &AABB| a.min.x + a.max.x < b.min.x + b.max.x,
        |a: &AABB, b: &AABB| a.min.x + a.max.x > b.min.x + b.max.x,
        |a: &AABB, b: &AABB| a.min.y + a.max.y < b.min.y + b.max.y,
        |a: &AABB, b: &AABB| a.min.y + a.max.y > b.min.y + b.max.y,
        |a: &AABB, b: &AABB| a.min.z + a.max.z < b.min.z + b.max.z,
        |a: &AABB, b: &AABB| a.min.z + a.max.z > b.min.z + b.max.z,
    );

    let time0 = std::time::Instant::now();

    let mut bvh_nodes: Vec<BlBvhNode> = Vec::with_capacity(bvh.nodes.len() * 6);

    ordered_flatten_bl_bvh!(orderings.0, bvh, bvh_nodes);
    ordered_flatten_bl_bvh!(orderings.1, bvh, bvh_nodes);
    ordered_flatten_bl_bvh!(orderings.2, bvh, bvh_nodes);
    ordered_flatten_bl_bvh!(orderings.3, bvh, bvh_nodes);
    ordered_flatten_bl_bvh!(orderings.4, bvh, bvh_nodes);
    ordered_flatten_bl_bvh!(orderings.5, bvh, bvh_nodes);

    println!("BVH flattened in {:?}", time0.elapsed());
    let time0 = std::time::Instant::now();

    let aabb = bvh_nodes.iter().fold(AABB::empty(), |a, b| {
        a.join(&AABB::with_bounds(b.bbox_min, b.bbox_max))
    });

    let gpu_bvh_nodes: Vec<_> = bvh_nodes.into_iter().map(pack_gpu_bvh_node).collect();

    let bvh_triangles = mesh
        .indices
        .chunks(3)
        .map(|t| {
            let v0 = Point3::from(mesh.positions[t[0] as usize]);
            let v1 = Point3::from(mesh.positions[t[1] as usize]);
            let v2 = Point3::from(mesh.positions[t[2] as usize]);
            let e0 = v1 - v0;
            let e1 = v2 - v0;
            GpuTriangle {
                v: (v0.x, v0.y, v0.z),
                e0_e1: (
                    ((half::f16::from_f32(e0.x).to_bits() as u32) << 16)
                        | (half::f16::from_f32(e1.x).to_bits() as u32),
                    ((half::f16::from_f32(e0.y).to_bits() as u32) << 16)
                        | (half::f16::from_f32(e1.y).to_bits() as u32),
                    ((half::f16::from_f32(e0.z).to_bits() as u32) << 16)
                        | (half::f16::from_f32(e1.z).to_bits() as u32),
                ),
            }
        })
        .collect::<Vec<_>>();

    println!("BVH encoded in {:?}", time0.elapsed());

    Ok(GpuBlBvh {
        nodes: gpu_bvh_nodes,
        triangles: bvh_triangles,
        aabb: (aabb.min.coords.into(), aabb.max.coords.into()),
    })
}

struct BlBvh {
    meta_buf: SnoozyRef<Buffer>,
    tri_buf: SnoozyRef<Buffer>,
    bvh_buf: SnoozyRef<Buffer>,
}

#[snoozy]
fn upload_bl_bvh(ctx: &mut Context, bvh: &SnoozyRef<GpuBlBvh>) -> Result<BlBvh> {
    let bvh = ctx.get(bvh)?;

    let nodes = ArcView::new(&bvh, |n| &n.nodes);
    let triangles = ArcView::new(&bvh, |n| &n.triangles);

    let meta_buf = upload_array_tex_buffer(Box::new(vec![(nodes.len() / 6) as u32]), gl::R32UI);

    let tri_buf = upload_array_tex_buffer(triangles, gl::RG32UI);
    let bvh_buf = upload_array_tex_buffer(nodes, gl::RGBA32UI);

    Ok(BlBvh {
        meta_buf,
        tri_buf,
        bvh_buf,
    })
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Abomonation)]
struct GpuBlBvhHeader {
    // Resident texture handles
    meta_buf: u64,
    tri_buf: u64,
    bvh_buf: u64,
    offset: [f32; 3],
    rotation: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Abomonation)]
struct TlBvhNode {
    bbox_min: [u32; 3], // Hints for visiting order in LSB
    bbox_extent: u32,   // 9e5

    left_idx: u16,  // 0 if leaf
    right_idx: u16, // Also prim idx for leaves

    sibling_idx: u16, // 0 if root
    parent_idx: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Abomonation)]
struct TlBvhNodeB {
    sibling_idx: u16, // 0 if root
    parent_idx: u16,
}

impl TlBvhNode {
    fn new_leaf(bbox_min: Point3, bbox_max: Point3, prim_idx: usize) -> Self {
        let bbox_min_arr: [f32; 3] = bbox_min.coords.into();
        Self {
            bbox_min: [
                bbox_min_arr[0].to_bits(),
                bbox_min_arr[1].to_bits(),
                bbox_min_arr[2].to_bits(),
            ],
            bbox_extent: pack_bbox_extent_9e5(bbox_max - bbox_min),
            left_idx: 0,
            right_idx: u16::try_from(prim_idx).expect("failed to pack primitive index"),
            sibling_idx: 0,
            parent_idx: 0,
        }
    }

    fn new_interior(bbox_min: Point3, bbox_max: Point3) -> Self {
        let bbox_min_arr: [f32; 3] = bbox_min.coords.into();
        Self {
            bbox_min: [
                bbox_min_arr[0].to_bits() & !3u32,
                bbox_min_arr[1].to_bits() & !3u32,
                bbox_min_arr[2].to_bits() & !3u32,
            ],
            bbox_extent: pack_bbox_extent_9e5(bbox_max - bbox_min),
            left_idx: 0,
            right_idx: 0,
            sibling_idx: 0,
            parent_idx: 0,
        }
    }

    fn set_left_idx(&mut self, left_idx: usize) {
        assert!(left_idx != 0);
        self.left_idx = u16::try_from(left_idx).expect("failed to pack left idx");
    }

    fn set_right_idx(&mut self, right_idx: usize) {
        assert!(right_idx != 0);
        self.right_idx = u16::try_from(right_idx).expect("failed to pack right idx");
    }

    fn set_sibling_idx(&mut self, sibling_idx: usize) {
        assert!(sibling_idx != 0);
        self.sibling_idx = u16::try_from(sibling_idx).expect("failed to pack sibling idx");
    }

    fn set_parent_idx(&mut self, parent_idx: usize) {
        self.parent_idx = u16::try_from(parent_idx).expect("failed to pack parent idx");
    }
}

assert_eq_size!(TlBvhNode, [u32; 6]);

fn convert_tl_bvh(node: usize, nbox: &AABB, nodes: &[BVHNode], res: &mut Vec<TlBvhNode>) -> usize {
    let initial_node_count = res.len();
    let n = &nodes[node];

    let node_res_idx = {
        res.push(if let BVHNode::Node { .. } = n {
            TlBvhNode::new_interior(nbox.min, nbox.max)
        } else {
            TlBvhNode::new_leaf(
                nbox.min,
                nbox.max,
                n.shape_index().expect("bvh leaf shape index"),
            )
        });
        initial_node_count
    };

    if let BVHNode::Node { .. } = n {
        let boxes = [&n.child_l_aabb(), &n.child_r_aabb()];
        let indices = [n.child_l(), n.child_r()];

        let orderings = [
            boxes[0].min.x + boxes[0].max.x < boxes[1].min.x + boxes[1].max.x,
            boxes[0].min.x + boxes[0].max.x > boxes[1].min.x + boxes[1].max.x,
            boxes[0].min.y + boxes[0].max.y < boxes[1].min.y + boxes[1].max.y,
            boxes[0].min.y + boxes[0].max.y > boxes[1].min.y + boxes[1].max.y,
            boxes[0].min.z + boxes[0].max.z < boxes[1].min.z + boxes[1].max.z,
            boxes[0].min.z + boxes[0].max.z > boxes[1].min.z + boxes[1].max.z,
        ];

        let ordering_bits_for_axis = |axis: usize| {
            let a = orderings[axis * 2 + 0];
            let b = orderings[axis * 2 + 1];
            (if a { 0 } else { 1 }) | (if b { 0 } else { 2 })
        };

        res[node_res_idx].bbox_min[0] |= ordering_bits_for_axis(0);
        res[node_res_idx].bbox_min[1] |= ordering_bits_for_axis(1);
        res[node_res_idx].bbox_min[2] |= ordering_bits_for_axis(2);

        let left_idx = convert_tl_bvh(indices[0], &boxes[0], nodes, res);
        let right_idx = convert_tl_bvh(indices[1], &boxes[1], nodes, res);

        res[left_idx].set_sibling_idx(right_idx);
        res[left_idx].set_parent_idx(node_res_idx);

        res[right_idx].set_sibling_idx(left_idx);
        res[right_idx].set_parent_idx(node_res_idx);

        res[node_res_idx].set_left_idx(left_idx);
        res[node_res_idx].set_right_idx(right_idx);
    }

    node_res_idx
}

#[snoozy]
pub fn upload_bvh(
    ctx: &mut Context,
    scene: &Vec<(SnoozyRef<TriangleMesh>, Vector3, UnitQuaternion)>,
) -> Result<ShaderUniformBundle> {
    let mut tla_data = Vec::with_capacity(scene.len());
    let mut bl_root_boxes: Vec<AABB> = Vec::with_capacity(scene.len());
    let mut total_aabb = AABB::empty();

    for (mesh, offset, rotation) in scene.iter() {
        let bvh = build_gpu_bvh(mesh.clone());
        let root_aabb = ctx.get(bvh.clone())?.aabb;
        let root_aabb = AABB::with_bounds(root_aabb.0.into(), root_aabb.1.into());

        if !root_aabb.is_empty() {
            let mesh = ctx.get(upload_bl_bvh(bvh))?;

            let meta_buf = ctx.get(&mesh.meta_buf)?;
            let tri_buf = ctx.get(&mesh.tri_buf)?;
            let bvh_buf = ctx.get(&mesh.bvh_buf)?;

            tla_data.push(GpuBlBvhHeader {
                meta_buf: meta_buf.bindless_texture_handle.unwrap(),
                tri_buf: tri_buf.bindless_texture_handle.unwrap(),
                bvh_buf: bvh_buf.bindless_texture_handle.unwrap(),
                offset: (*offset).into(),
                rotation: rotation.quaternion().as_vector().clone().into(),
            });

            let xform_aabb = {
                let center: Vector3 = root_aabb.center().coords;
                let extent: Vector3 = root_aabb.size() * 0.5;

                let rot_matrix: Matrix3 = rotation.to_rotation_matrix().into();
                let rot_center = rot_matrix * center;

                let abs_rot_matrix = rot_matrix.abs();
                let rot_extent = abs_rot_matrix * extent;

                AABB::with_bounds(
                    (offset + rot_center - rot_extent).into(),
                    (offset + rot_center + rot_extent).into(),
                )
            };

            total_aabb.join_mut(&xform_aabb);
            bl_root_boxes.push(xform_aabb);
        }
    }

    let tl_bvh_packed = if !bl_root_boxes.is_empty() {
        //let time0 = std::time::Instant::now();
        let tl_bvh = BVH::build(&bl_root_boxes);
        //println!("TL BVH built in {:?}", time0.elapsed());

        let mut tl_bvh_packed = Vec::with_capacity(tl_bvh.nodes.len());
        convert_tl_bvh(0, &total_aabb, &tl_bvh.nodes, &mut tl_bvh_packed);

        tl_bvh_packed
    } else {
        vec![TlBvhNode::new_interior(
            AABB::empty().min,
            AABB::empty().max,
        )]
    };

    let tl_bvh_packed_b = tl_bvh_packed
        .iter()
        .map(|a| TlBvhNodeB {
            sibling_idx: a.sibling_idx,
            parent_idx: a.parent_idx,
        })
        .collect::<Vec<_>>();

    /*for b in tl_bvh_packed.iter_mut() {
        b.sibling_idx = (((b.bbox_min[0] & 3) << 0)
            | ((b.bbox_min[1] & 3) << 2)
            | ((b.bbox_min[0] & 3) << 4)) as u16;
    }*/

    let bl_root_boxes: Vec<([f32; 3], [f32; 3])> = bl_root_boxes
        .iter()
        .map(|b| (b.min.coords.into(), b.max.coords.into()))
        .collect();

    let bl_count = tla_data.len() as u32;

    Ok(shader_uniforms!(
        "rt_tla_buf": upload_array_buffer(Box::new(tla_data)),
        "rt_tla_meta_buf": upload_buffer(bl_count),
        "rt_tla_root_boxes": upload_array_buffer(Box::new(bl_root_boxes)),
        "rt_tla_nodes": upload_array_tex_buffer(Box::new(tl_bvh_packed), gl::RG32UI),
        "rt_tla_nodes_b": upload_array_tex_buffer(Box::new(tl_bvh_packed_b), gl::R32UI),
    ))
}
