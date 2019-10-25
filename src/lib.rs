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
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Abomonation)]
#[repr(C)]
pub struct GpuBvhNode {
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

fn pack_gpu_bvh_node(node: BvhNode) -> GpuBvhNode {
    let bmin = (
        half::f16::from_bits(f32_to_f16_negbias(node.bbox_min.x)),
        half::f16::from_bits(f32_to_f16_negbias(node.bbox_min.y)),
        half::f16::from_bits(f32_to_f16_negbias(node.bbox_min.z)),
    );

    let box_extent_packed = {
        // The fp16 was rounded-down, so extent will be larger than for fp32
        let extent =
            node.bbox_max - Vector3::new(bmin.0.to_f32(), bmin.1.to_f32(), bmin.2.to_f32());

        pack_rgb9e5_roundup(extent.x, extent.y, extent.z)
    };

    assert!(node.exit_idx < (1u32 << 24));
    assert!(node.prim_idx == std::u32::MAX || node.prim_idx < (1u32 << 24));

    GpuBvhNode {
        packed: (
            box_extent_packed,
            ((bmin.0.to_bits() as u32) << 16) | (bmin.1.to_bits() as u32),
            ((bmin.2.to_bits() as u32) << 16) | ((node.prim_idx >> 8) & 0xffff),
            ((node.prim_idx & 0xff) << 24) | node.exit_idx,
        ),
    }
}

pub struct BvhNode {
    bbox_min: Point3,
    exit_idx: u32,
    bbox_max: Point3,
    prim_idx: u32,
}

impl BvhNode {
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

fn convert_bvh<BoxOrderFn>(
    node: usize,
    nbox: &AABB,
    nodes: &[BVHNode],
    are_boxes_correctly_ordered: &BoxOrderFn,
    res: &mut Vec<BvhNode>,
) where
    BoxOrderFn: Fn(&AABB, &AABB) -> bool,
{
    let initial_node_count = res.len();
    let n = &nodes[node];

    let node_res_idx = if node != 0 {
        res.push(if let BVHNode::Node { .. } = n {
            BvhNode::new_interior(nbox.min, nbox.max)
        } else {
            BvhNode::new_leaf(
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

        convert_bvh(
            indices[first],
            &boxes[first],
            nodes,
            are_boxes_correctly_ordered,
            res,
        );
        convert_bvh(
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

macro_rules! ordered_flatten_bvh {
    ($order: expr, $bvh:ident, $bvh_nodes:ident) => {{
        convert_bvh(
            0,
            &AABB::default(),
            $bvh.nodes.as_slice(),
            &$order,
            &mut $bvh_nodes,
        );
    }};
}

#[derive(Clone, Abomonation)]
pub struct GpuBvh {
    nodes: Vec<GpuBvhNode>,
    triangles: Vec<GpuTriangle>,
}

#[snoozy]
pub fn build_gpu_bvh(ctx: &mut Context, mesh: &SnoozyRef<TriangleMesh>) -> Result<GpuBvh> {
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

    let time0 = std::time::Instant::now();
    let bvh = BVH::build(&aabbs);
    println!("BVH built in {:?}", time0.elapsed());

    let orderings = (
        |a: &AABB, b: &AABB| a.min.x + a.max.x < b.min.x + b.max.x,
        |a: &AABB, b: &AABB| a.min.x + a.max.x > b.min.x + b.max.x,
        |a: &AABB, b: &AABB| a.min.y + a.max.y < b.min.y + b.max.y,
        |a: &AABB, b: &AABB| a.min.y + a.max.y > b.min.y + b.max.y,
        |a: &AABB, b: &AABB| a.min.z + a.max.z < b.min.z + b.max.z,
        |a: &AABB, b: &AABB| a.min.z + a.max.z > b.min.z + b.max.z,
    );

    let time0 = std::time::Instant::now();

    let mut bvh_nodes: Vec<BvhNode> = Vec::with_capacity(bvh.nodes.len() * 6);

    ordered_flatten_bvh!(orderings.0, bvh, bvh_nodes);
    ordered_flatten_bvh!(orderings.1, bvh, bvh_nodes);
    ordered_flatten_bvh!(orderings.2, bvh, bvh_nodes);
    ordered_flatten_bvh!(orderings.3, bvh, bvh_nodes);
    ordered_flatten_bvh!(orderings.4, bvh, bvh_nodes);
    ordered_flatten_bvh!(orderings.5, bvh, bvh_nodes);

    println!("BVH flattened in {:?}", time0.elapsed());
    let time0 = std::time::Instant::now();

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

    Ok(GpuBvh {
        nodes: gpu_bvh_nodes,
        triangles: bvh_triangles,
    })
}

struct BlBvh {
    meta_buf: SnoozyRef<Buffer>,
    tri_buf: SnoozyRef<Buffer>,
    bvh_buf: SnoozyRef<Buffer>,
}

#[snoozy]
fn upload_bl_bvh(ctx: &mut Context, bvh: &SnoozyRef<GpuBvh>) -> Result<BlBvh> {
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
struct GpuBlBvhHeader {
    // Resident texture handles
    meta_buf: u64,
    tri_buf: u64,
    bvh_buf: u64,
    offset_x: f32,
    offset_y: f32,
    offset_z: f32,
}

impl Hash for GpuBlBvhHeader {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.meta_buf.hash(state);
        self.tri_buf.hash(state);
        self.bvh_buf.hash(state);
        self.offset_x.to_bits().hash(state);
        self.offset_y.to_bits().hash(state);
        self.offset_z.to_bits().hash(state);
    }
}

#[snoozy]
pub fn upload_bvh(
    ctx: &mut Context,
    scene: &Vec<(SnoozyRef<TriangleMesh>, Vector3, Quaternion)>,
) -> Result<ShaderUniformBundle> {
    let tla_data: Vec<GpuBlBvhHeader> = scene
        .iter()
        .cloned()
        .map(|(mesh, offset, _rotation)| {
            let mesh = ctx.get(upload_bl_bvh(build_gpu_bvh(mesh)))?;

            let meta_buf = ctx.get(&mesh.meta_buf)?;
            let tri_buf = ctx.get(&mesh.tri_buf)?;
            let bvh_buf = ctx.get(&mesh.bvh_buf)?;

            Ok(GpuBlBvhHeader {
                meta_buf: meta_buf.bindless_texture_handle.unwrap(),
                tri_buf: tri_buf.bindless_texture_handle.unwrap(),
                bvh_buf: bvh_buf.bindless_texture_handle.unwrap(),
                offset_x: offset.x,
                offset_y: offset.y,
                offset_z: offset.z,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let bl_count = tla_data.len() as u32;

    Ok(shader_uniforms!(
        "rt_tla_buf": upload_array_buffer(Box::new(tla_data)),
        "rt_tla_meta_buf": upload_buffer(bl_count),
    ))
}
