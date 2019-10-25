#extension GL_ARB_bindless_texture: enable
#define INVALID_TRIANGLE_INDEX 0xffffff

struct PackedBvhNode {
    uvec4 data;
};

struct BvhNode {
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

BvhNode unpack_bvh_node(PackedBvhNode node) {
    vec3 box_extent = rgb9e5_to_float3(node.data[0]);

    BvhNode res;
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
    uvec2 meta_buf_sampler;
    uvec2 tri_buf_sampler;
    uvec2 bvh_buf_sampler;
    float offset_x;
    float offset_y;
    float offset_z;
};

layout(std430) buffer rt_tla_meta_buf {
    uint bl_count;
};

layout(std430) buffer rt_tla_buf {
    BlBvhHeader bl_headers[];
};

uniform usamplerBuffer bvh_meta_buf;

uint get_bvh_node_count(BlBvhHeader header) {
    return texelFetch(usamplerBuffer(header.meta_buf_sampler), 0).x;
}

PackedBvhNode get_bvh_node(usamplerBuffer bvh_sampler, uint idx) {
    PackedBvhNode res;
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

    const bool cull_backface = false;

    if ((cull_backface && det > 1e-10) || !cull_backface)
    {
    	vec3 tv = r.o - tri.v;
    	vec3 qv = cross(tv, tri.e0);

        float hit_t = dot(tri.e1, qv) / det;
        if (hit_t >= 0.0 && hit_t < t) {
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
    uint tri_idx;
    uint debug_iter_count;
    usamplerBuffer tri_sampler;
};

bool raytrace_bl(Ray r, uint bl_idx, inout RtHit hit) {
    BlBvhHeader bl_header = bl_headers[bl_idx];
    usamplerBuffer bvh_sampler = usamplerBuffer(bl_header.bvh_buf_sampler);
    usamplerBuffer tri_sampler = usamplerBuffer(bl_header.tri_buf_sampler);

    r.o.x -= bl_header.offset_x;
    r.o.y -= bl_header.offset_y;
    r.o.z -= bl_header.offset_z;

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

    uint iter = 0;
    for (; iter < 1024 && node_idx < end_idx; ++iter) {
        BvhNode node = unpack_bvh_node(get_bvh_node(bvh_sampler, node_idx));
        bool intersects_box = intersect_ray_aabb(r, node.box_min, node.box_max, tmin);

        bool is_leaf = node.prim_idx != INVALID_TRIANGLE_INDEX;

        if (intersects_box && is_leaf) {
            if (intersect_ray_tri(r, unpack_triangle(get_bvh_triangle(tri_sampler, node.prim_idx)), tmin, barycentric)) {
                hit_tri = node.prim_idx;
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
        hit.tri_sampler = tri_sampler;
        return true;
    }

    return false;
}

bool raytrace(Ray r, inout RtHit hit) {
    int result = 0;
    for (uint i = 0; i < bl_count; ++i) {
        result = result | int(raytrace_bl(r, i, hit));
    }
    return bool(result);
}

bool raytrace_intersects_any_bl(Ray r, uint bl_idx, float tmax) {
    BlBvhHeader bl_header = bl_headers[bl_idx];
    usamplerBuffer bvh_sampler = usamplerBuffer(bl_header.bvh_buf_sampler);
    usamplerBuffer tri_sampler = usamplerBuffer(bl_header.tri_buf_sampler);

    r.o.x -= bl_header.offset_x;
    r.o.y -= bl_header.offset_y;
    r.o.z -= bl_header.offset_z;

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
        BvhNode node = unpack_bvh_node(get_bvh_node(bvh_sampler, node_idx));
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

bool raytrace_intersects_any(Ray r, float tmax) {
    bool result = false;
    for (uint i = 0; i < bl_count && !result; ++i) {
        result = result || raytrace_intersects_any_bl(r, i, tmax);
    }
    return result;
}

bool raytrace_intersects_any(Ray r) {
    return raytrace_intersects_any(r, 1e10);
}
