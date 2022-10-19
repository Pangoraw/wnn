@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;

    if x > {{ o_lens[0] }}u {
        return;
    }

    let b = x / {{ o_strides[0][0] }}u;
    var rest = x % {{ o_strides[0][0] }}u;

    let out_chan = rest / {{ o_strides[0][1] }}u;
    rest = rest % {{ o_strides[0][1] }}u;

    let out_x = rest / {{ o_strides[0][2] }}u;
    rest = rest % {{ o_strides[0][2] }}u;

    let out_y = rest / {{ o_strides[0][3] }}u;

    let root_input_index = b * {{ i_strides[0][1] }}u;

    let base_input_index =
        root_input_index
        + out_chan * {{ i_strides[0][1] }}u 
        + out_y * {{ i_strides[0][2] }}u
        + out_x * {{ i_strides[0][3] }}u
        - {{ half_kernel_size[0] * i_strides[0][3] + half_kernel_size[1] * i_strides[0][2] }}u;

    var tmpmax: f32 = input[base_input_index];
    for (var i: u32 = 0u; i < {{ 2 * half_kernel_size[0] }}u; i = i + 1u) {
        for (var j: u32 = 0u; j < {{ 2 * half_kernel_size[1] }}u; j = j + 1u) {
            let input_idx = base_input_index + j * {{ i_strides[0][2] }}u + i * {{ i_strides[0][3] }}u;

            tmpmax = max(tmpmax, input[input_idx]);
        }
    }

    output[x] = tmpmax;
}
