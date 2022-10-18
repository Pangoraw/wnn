@group(0) @binding(0)
var<storage, read> input_left: array<f32>;

@group(0) @binding(1)
var<storage, read> input_right: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    var tmpsum = 0.0;
    for (var k: u32 = 0u; k < 10u; k = k + 1u) {
        tmpsum = tmpsum + input_left[k + y * 10u] * input_right[x + k * 20u];
    }

    output[x + y * 20u] = tmpsum;
}
