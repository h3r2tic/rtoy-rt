#extension GL_EXT_nonuniform_qualifier: require
#define INVALID_TRIANGLE_INDEX 0xffffff

#include "rendertoy::shaders/bindless.inc"

struct PackedBlBvhNode {
    uvec4 data;
};

struct BlBvhNode {
    vec3 box_min;
    uint exit_idx;
    vec3 box_max;
    uint prim_idx;
};

#define RGB9E5_EXPONENT_BITS 5
#define RGB9E5_MANTISSA_BITS 9
#define RGB9E5_EXP_BIAS 15

// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
vec3 rgb9e5_to_float3(uint v) {
    uint biasedexponent = v & ((1u << RGB9E5_EXPONENT_BITS) - 1u);
    int exponent = int(biasedexponent) - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS;
    float scale = exp2(exponent);

    const uint MANTISSA_MASK = ((1u << RGB9E5_MANTISSA_BITS) - 1u);

    uint r = v >> (32 - 9);
    uint g = (v >> (32 - 9 * 2)) & MANTISSA_MASK;
    uint b = (v >> (32 - 9 * 3)) & MANTISSA_MASK;

    return vec3(r * scale, g * scale, b * scale);
}

BlBvhNode unpack_bl_bvh_node(PackedBlBvhNode node) {
    vec3 box_extent = rgb9e5_to_float3(node.data[0]);

    BlBvhNode res;
    res.box_min.xy = unpackHalf2x16(node.data[1]).yx;
    res.box_min.z = unpackHalf2x16(node.data[2]).y;
    res.box_max = res.box_min + box_extent;
    res.prim_idx = ((node.data[2] & 0xffffu) << 8u) | (node.data[3] >> 24u);
    res.exit_idx = node.data[3] & 0xffffffu;
    return res;
}

struct PackedTriangle {
    float v[3];
    uint e[3];
};

struct Triangle {
    vec3 v;
    vec3 e0;
    vec3 e1;
};

Triangle unpack_triangle(PackedTriangle tri) {
    Triangle res;
    res.v = vec3(tri.v[0], tri.v[1], tri.v[2]);
    vec2 ex = unpackHalf2x16(tri.e[0]);
    vec2 ey = unpackHalf2x16(tri.e[1]);
    vec2 ez = unpackHalf2x16(tri.e[2]);    
    res.e0 = vec3(ex.y, ey.y, ez.y);
    res.e1 = vec3(ex.x, ey.x, ez.x);
    return res;
}

struct BlBvhHeader {
    uint meta_buf_sampler;
    uint tri_buf_sampler;
    uint bvh_buf_sampler;
    float offset[3];
    float rotation[4];
};

layout(std430) buffer rt_tla_buf {
    BlBvhHeader bl_headers[];
};

uniform usamplerBuffer bvh_meta_buf;

uint get_bvh_node_count(BlBvhHeader header) {
    return texelFetch(all_buffers[nonuniformEXT(header.meta_buf_sampler)], 0).x;
}

PackedBlBvhNode get_bl_bvh_node(usamplerBuffer bvh_sampler, uint idx) {
    PackedBlBvhNode res;
    res.data = texelFetch(bvh_sampler, int(idx));
    return res;
}

PackedTriangle get_bvh_triangle(usamplerBuffer tri_sampler, uint idx) {
    PackedTriangle res;
    int loc = int(idx * 3);
    uvec2 d0 = texelFetch(tri_sampler, loc+0).xy;
    uvec2 d1 = texelFetch(tri_sampler, loc+1).xy;
    uvec2 d2 = texelFetch(tri_sampler, loc+2).xy;
    res.v[0] = uintBitsToFloat(d0.x);
    res.v[1] = uintBitsToFloat(d0.y);
    res.v[2] = uintBitsToFloat(d1.x);
    res.e[0] = d1.y;
    res.e[1] = d2.x;
    res.e[2] = d2.y;
    return res;
}

/*
// RGBA32UI version; a bit slower in practice

PackedTriangle get_bvh_triangle(usamplerBuffer tri_sampler, uint idx) {
    PackedTriangle res;
    idx *= 3;
    int loc = int(idx >> 1u);
    uvec4 d0 = texelFetch(tri_sampler, loc+0);
    uvec4 d1 = texelFetch(tri_sampler, loc+1);
    uint c = idx & 1;
    res.v[0] = uintBitsToFloat(0 == c ? d0.x : d0.z);
    res.v[1] = uintBitsToFloat(0 == c ? d0.y : d0.w);
    res.v[2] = uintBitsToFloat(0 == c ? d0.z : d1.x);
    res.e[0] = 0 == c ? d0.w : d1.y;
    res.e[1] = 0 == c ? d1.x : d1.z;
    res.e[2] = 0 == c ? d1.y : d1.w;
    return res;
}
*/

struct Ray {
	vec3 o;
	vec3 d;
};

// From https://github.com/tigrazone/glslppm
bool intersect_ray_tri(Ray r, Triangle tri, inout float t, inout vec3 barycentric) {
    vec3 pv = cross(r.d, tri.e1);
    float det = dot(tri.e0, pv);

    const bool cull_backface = true;
    bool passed_face_culling = (cull_backface && det > 1e-10) || !cull_backface;

    {
    	vec3 tv = r.o - tri.v;
    	vec3 qv = cross(tv, tri.e0);

        float hit_t = dot(tri.e1, qv) / det;
        if (hit_t >= 0.0 && hit_t < t && passed_face_culling) {
        	vec3 uvw;
        	uvw.x = dot(tv, pv);
        	uvw.y = dot(r.d, qv);
        	uvw.xy = uvw.xy / det;
        	uvw.z = 1.0 - uvw.x - uvw.y;

            float barycentric_eps = -1e-4;

        	if (all(greaterThanEqual(uvw, vec3(barycentric_eps.xxx))))
    	    {
    		    barycentric = uvw.yzx;
                t = hit_t;
                return true;
    	    }
        }
    }

    return false;
}

// From https://github.com/tigrazone/glslppm
bool intersect_ray_aabb(Ray r, vec3 pmin, vec3 pmax, float t)
{
	vec3 min_interval = (pmax.xyz - r.o.xyz) / r.d;
	vec3 max_interval = (pmin.xyz - r.o.xyz) / r.d;

	vec3 a = min(min_interval, max_interval);
	vec3 b = max(min_interval, max_interval);

    float tmin = max(max(a.x, a.y), a.z);
    float tmax = min(min(b.x, b.y), b.z);

    return tmin <= tmax && tmin < t && tmax >= 0.0;
}

struct RtHit {
    float t;
    vec3 barycentric;
    vec3 normal;
    uint tri_idx;
    uint debug_iter_count;
    uint tri_sampler_idx;
};

vec4 quat_conj(vec4 q) {
    return vec4(-q.x, -q.y, -q.z, q.w); 
}

vec3 qtransform(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
} 

bool raytrace_bl(BlBvhHeader bl_header, usamplerBuffer bvh_sampler, usamplerBuffer tri_sampler, uint tri_sampler_idx, Ray r, uint bl_idx, inout RtHit hit) {
    vec4 rotation = vec4(bl_header.rotation[0], bl_header.rotation[1], bl_header.rotation[2], bl_header.rotation[3]);
    vec4 inv_rotation = quat_conj(rotation);
    r.o -= vec3(bl_header.offset[0], bl_header.offset[1], bl_header.offset[2]);
    r.o = qtransform(inv_rotation, r.o);
    r.d = qtransform(inv_rotation, r.d);

    uint node_idx = 0;
    {
        vec3 absdir = abs(r.d);
        float maxcomp = max(absdir.x, max(absdir.y, absdir.z));
        if (absdir.x == maxcomp) {
            node_idx = r.d.x > 0.0 ? 0 : 1;
        } else if (absdir.y == maxcomp) {
            node_idx = r.d.y > 0.0 ? 2 : 3;
        } else if (absdir.z == maxcomp) {
            node_idx = r.d.z > 0.0 ? 4 : 5;
        }
        node_idx *= get_bvh_node_count(bl_header);
    }

    uint end_idx = node_idx + get_bvh_node_count(bl_header);
    
    float tmin = hit.t;
    vec3 barycentric;
    uint hit_tri = INVALID_TRIANGLE_INDEX;
    vec3 hit_normal;

    uint iter = 0;
    for (; iter < 1024 && node_idx < end_idx; ++iter) {
        BlBvhNode node = unpack_bl_bvh_node(get_bl_bvh_node(bvh_sampler, node_idx));
        bool intersects_box = intersect_ray_aabb(r, node.box_min, node.box_max, tmin);

        bool is_leaf = node.prim_idx != INVALID_TRIANGLE_INDEX;

        if (intersects_box && is_leaf) {
            Triangle tri = unpack_triangle(get_bvh_triangle(tri_sampler, node.prim_idx));
            if (intersect_ray_tri(r, tri, tmin, barycentric)) {
                hit_tri = node.prim_idx;
                hit_normal = cross(tri.e0, tri.e1);
            }
        }

        if (is_leaf || intersects_box) {
            node_idx += 1;
        } else {
            node_idx += node.exit_idx;
        }
    }

    hit.debug_iter_count = iter;

    if (hit_tri != INVALID_TRIANGLE_INDEX) {
        hit.t = tmin;
        hit.barycentric = barycentric;
        hit.tri_idx = hit_tri;
        hit.tri_sampler_idx = tri_sampler_idx;
        hit.normal = normalize(qtransform(rotation, hit_normal));
        return true;
    }

    return false;
}

bool raytrace_bl(Ray r, uint bl_idx, inout RtHit hit) {
    BlBvhHeader bl_header = bl_headers[bl_idx];
    return raytrace_bl(
        bl_header,
        all_buffers[nonuniformEXT(bl_header.bvh_buf_sampler)],
        all_buffers[nonuniformEXT(bl_header.tri_buf_sampler)],
        bl_header.tri_buf_sampler,
        r,
        bl_idx,
        hit);
}

bool raytrace_intersects_any_bl(BlBvhHeader bl_header, usamplerBuffer bvh_sampler, usamplerBuffer tri_sampler, Ray r, uint bl_idx, float tmax) {
    vec4 rotation = vec4(bl_header.rotation[0], bl_header.rotation[1], bl_header.rotation[2], bl_header.rotation[3]);
    vec4 inv_rotation = quat_conj(rotation);
    r.o -= vec3(bl_header.offset[0], bl_header.offset[1], bl_header.offset[2]);
    r.o = qtransform(inv_rotation, r.o);
    r.d = qtransform(inv_rotation, r.d);

    uint node_idx = 0;
    {
        vec3 absdir = abs(r.d);
        float maxcomp = max(absdir.x, max(absdir.y, absdir.z));
        if (absdir.x == maxcomp) {
            node_idx = r.d.x > 0.0 ? 0 : 1;
        } else if (absdir.y == maxcomp) {
            node_idx = r.d.y > 0.0 ? 2 : 3;
        } else if (absdir.z == maxcomp) {
            node_idx = r.d.z > 0.0 ? 4 : 5;
        }
        node_idx *= get_bvh_node_count(bl_header);
    }

    uint end_idx = node_idx + get_bvh_node_count(bl_header);
    
    uint iter = 0;
    for (; iter < 1024 && node_idx < end_idx; ++iter) {
        BlBvhNode node = unpack_bl_bvh_node(get_bl_bvh_node(bvh_sampler, node_idx));
        bool intersects_box = intersect_ray_aabb(r, node.box_min, node.box_max, tmax);
        bool is_leaf = node.prim_idx != INVALID_TRIANGLE_INDEX;

        if (intersects_box && is_leaf) {
            vec3 barycentric;
            if (intersect_ray_tri(r, unpack_triangle(get_bvh_triangle(tri_sampler, node.prim_idx)), tmax, barycentric)) {
                return true;
            }
        }

        if (is_leaf || intersects_box) {
            node_idx += 1;
        } else {
            node_idx += node.exit_idx;
        }
    }

    return false;
}

bool raytrace_intersects_any_bl(Ray r, uint bl_idx, float tmax) {
    BlBvhHeader bl_header = bl_headers[bl_idx];
    return raytrace_intersects_any_bl(
        bl_header,
        all_buffers[nonuniformEXT(bl_header.bvh_buf_sampler)],
        all_buffers[nonuniformEXT(bl_header.tri_buf_sampler)],
        r,
        bl_idx,
        tmax);
}

uniform usamplerBuffer rt_tla_nodes;
uniform usamplerBuffer rt_tla_nodes_b;

struct PackedTlBvhNode {
    uvec2 data[3];
};

PackedTlBvhNode get_tl_bvh_node(uint idx) {
    PackedTlBvhNode res;
    res.data[0] = texelFetch(rt_tla_nodes, int(idx*3+0)).xy;
    res.data[1] = texelFetch(rt_tla_nodes, int(idx*3+1)).xy;
    res.data[2] = texelFetch(rt_tla_nodes, int(idx*3+2)).xy;
    return res;
}

struct TlBvhNode {
    vec3 box_min;
    vec3 box_max;

    uint left_idx;  // 0 if leaf
    uint right_idx; // Also prim idx for leaves

    uint sibling_idx;
    uint parent_idx;

    uint dir_bits;
};

TlBvhNode unpack_tl_bvh_node(PackedTlBvhNode node) {
    TlBvhNode res;
    res.dir_bits = 0;
    res.dir_bits |= (node.data[0].x & 3u);
    res.dir_bits |= (node.data[0].y & 3u) << 2u;
    res.dir_bits |= (node.data[1].x & 3u) << 4u;
    res.box_min = vec3(uintBitsToFloat(node.data[0]), uintBitsToFloat(node.data[1].x));
    vec3 box_extent = rgb9e5_to_float3(node.data[1].y);
    res.box_max = res.box_min + box_extent;

    res.left_idx = node.data[2].x & 0xffffu;
    res.right_idx = node.data[2].x >> 16u;

    res.sibling_idx = node.data[2].y & 0xffffu;
    res.parent_idx = node.data[2].y >> 16u;

    return res;
}

bool raytrace(Ray r, inout RtHit hit) {
    uint node_idx = 0;
    uint stack = 0;
    int result = 0;

    uint dir_bit_mask = 0;
    {
        vec3 absdir = abs(r.d);
        float maxcomp = max(absdir.x, max(absdir.y, absdir.z));
        uint dir_bit_idx = 0;
        if (absdir.x == maxcomp) {
            dir_bit_idx = r.d.x > 0.0 ? 0 : 1;
        } else if (absdir.y == maxcomp) {
            dir_bit_idx = r.d.y > 0.0 ? 2 : 3;
        } else if (absdir.z == maxcomp) {
            dir_bit_idx = r.d.z > 0.0 ? 4 : 5;
        }
        dir_bit_mask = 1u << dir_bit_idx;
    }

    uint iter = 0;
    for (; iter < 1024; ++iter) {
        TlBvhNode node = unpack_tl_bvh_node(get_tl_bvh_node(node_idx));
        uint child_idx = (node.dir_bits & dir_bit_mask) == 0 ? node.left_idx : node.right_idx;
        bool is_leaf = node.left_idx == 0;

        bool intersects_box = intersect_ray_aabb(r, node.box_min, node.box_max, hit.t);

        if (intersects_box && is_leaf) {
            result |= int(raytrace_bl(r, node.right_idx, hit));
        }

        if (!is_leaf && intersects_box)
        {
            // Go down
            node_idx = child_idx;
            stack <<= 1u; // Push 0
        } else {
            // While this is the second child visited at a given level, go up
            uint parent_idx = node.parent_idx;
            uint sibling_idx = node.sibling_idx;

            while ((stack & 1) != 0) {
                node_idx = parent_idx;

                #if 0
                    TlBvhNode partial_node = unpack_tl_bvh_node(get_tl_bvh_node(node_idx));
                    parent_idx = partial_node.parent_idx;
                    sibling_idx = partial_node.sibling_idx;
                #else
                    uint data2 = texelFetch(rt_tla_nodes_b, int(node_idx)).x;
                    sibling_idx = data2 & 0xffffu;
                    parent_idx = data2 >> 16u;
                #endif

                stack >>= 1;    // Pop
            }

            node_idx = sibling_idx;
            stack |= 1;
        }

        if (0 == node_idx) {
            break;
        }
    }

    return bool(result);
}

bool raytrace_intersects_any(Ray r, float tmax) {
    uint node_idx = 0;
    uint stack = 0;

    uint iter = 0;
    for (; iter < 1024; ++iter) {
        TlBvhNode node = unpack_tl_bvh_node(get_tl_bvh_node(node_idx));

        bool intersects_box = intersect_ray_aabb(r, node.box_min, node.box_max, tmax);
        bool is_leaf = node.left_idx == 0;

        if (intersects_box && is_leaf) {
            if (raytrace_intersects_any_bl(r, node.right_idx, tmax)) {
                return true;
            }
        }

        if (!is_leaf && intersects_box)
        {
            // Go down
            node_idx = node.left_idx;
            stack <<= 1u; // Push 0
        } else {
            // While this is the second child visited at a given level, go up
            uint parent_idx = node.parent_idx;
            uint sibling_idx = node.sibling_idx;

            while ((stack & 1) != 0) {
                node_idx = parent_idx;

                #if 0
                    TlBvhNode partial_node = unpack_tl_bvh_node(get_tl_bvh_node(node_idx));
                    parent_idx = partial_node.parent_idx;
                    sibling_idx = partial_node.sibling_idx;
                #else
                    uint data2 = texelFetch(rt_tla_nodes_b, int(node_idx)).x;
                    sibling_idx = data2 & 0xffffu;
                    parent_idx = data2 >> 16u;
                #endif

                stack >>= 1;    // Pop
            }

            node_idx = sibling_idx;
            stack |= 1;
        }

        if (0 == node_idx) {
            break;
        }
    }

    return false;
}

bool raytrace_intersects_any(Ray r) {
    return raytrace_intersects_any(r, 1e10);
}
