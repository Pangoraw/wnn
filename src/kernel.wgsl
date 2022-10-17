@group(0) @binding(0)
var<storage, read> input_left: array<f32>;

@group(0) @binding(1)
var<storage, read> input_right: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let y = global_id.y;
    let x = global_id.x;

    var tmpsum = 0.;
    for (var k: u32 = 0u; k < 1000u; k = k + 1u) {
        tmpsum = tmpsum + input_left[k] * input_right[y + k * 2u];
    }

    output[x * 2u + y] = tmpsum;
}
