@group(0) @binding(0)
var<storage, read> input_left: array<f32>;

@group(0) @binding(1)
var<storage, read> input_right: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size({{ workgroup_x }}, {{ workgroup_y }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    var tmpsum = 0.0;
    for (var k: u32 = 0u; k < {{ i_sizes[0][1] }}u; k = k + 1u) {
        tmpsum = tmpsum + input_left[k + y * {{ i_sizes[0][1] }}u] * input_right[x + k * {{ i_sizes[1][1] }}u];
    }

    output[x + y * {{ o_sizes[0][0] }}u] = tmpsum;
}
