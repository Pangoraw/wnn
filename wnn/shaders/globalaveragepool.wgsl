type T = {{ scalar }};

@group(0) @binding(0)
var<storage, read> input: array<T>;

@group(0) @binding(1)
var<storage, read_write> output: array<T>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;

    if x > {{ o_lens[0] }}u {
        return;
    }

    let b = x / {{ o_strides[0][0] }}u;
    var rest = x % {{ o_strides[0][0] }}u;

    let out_chan = rest / {{ o_strides[0][1] }}u;
    let root_input_index = b * {{ i_strides[0][1] }}u;

    let base_input_index =
        root_input_index
        + out_chan * {{ i_strides[0][1] }}u;

    var sum = T();
    for (var i: u32 = 0u; i < {{ i_sizes[0][3] }}u; i = i + 1u) {
        for (var j: u32 = 0u; j < {{ i_sizes[0][2] }}u; j = j + 1u) {
            let input_idx = base_input_index + j * {{ i_strides[0][2] }}u + i * {{ i_strides[0][3] }}u;

            sum = sum + input[input_idx];
        }
    }
    let mean = sum / {{ i_strides[0][1] }}.;

    output[x] = mean;
}
