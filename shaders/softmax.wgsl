@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gdx = global_id.x; // reduce index

    let base_idx = gdx % {{ i_sizes[0][axis] }}u +
                   {{ i_strides[0][axis] }}u * gdx / {{ i_sizes[0][axis] }}u;

    var sum: f32 = 0.;
    for (var k: u32 = 0u; k < {{ o_sizes[0][axis] }}u; k = k + 1u) {
        let index = base_idx + k * {{ i_strides[0][axis] }}u;
        let e = exp(input[index]);
        output[index] = e;
        sum = sum + e;
    }

    for (var k: u32 = 0u; k < {{ o_sizes[0][axis] }}u; k = k + 1u) {
        let index = base_idx + k * {{ i_strides[0][axis] }}u;

        if index >= {{ o_lens[0] }}u {
            return;
        }

        output[index] = output[index] / sum;
    }
}
