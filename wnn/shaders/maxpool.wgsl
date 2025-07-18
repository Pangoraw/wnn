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
    rest = rest % {{ o_strides[0][1] }}u;

    let out_y = rest / {{ o_strides[0][2] }}u;
    rest = rest % {{ o_strides[0][2] }}u;

    let out_x = rest / {{ o_strides[0][3] }}u;

    let root_input_index = b * {{ i_strides[0][0] }}u;

    let base_input_index =
        root_input_index
        + out_chan * {{ i_strides[0][1] }}u;

    // https://gpuweb.github.io/gpuweb/wgsl/#floating-point-types
    var tmpmax: T = -3.40282346638528859812e+38f;
    for (var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
        let tmp_y = out_y * {{ k_strides[0] }}u + i - {{ pads[0] }}u;

        if (tmp_y >= 0u) && (tmp_y < {{ i_sizes[0][2] }}u) {
            for (var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) {
                let tmp_x = out_x * {{ k_strides[1] }}u + j - {{ pads[1] }}u;

                if (tmp_x >= 0u) && (tmp_x < {{ i_sizes[0][3] }}u) {
                    let input_index =
                        base_input_index +
                        tmp_y * {{ i_strides[0][2] }}u +
                        tmp_x * {{ i_strides[0][3] }}u;

                    tmpmax = max(tmpmax, input[input_index]);
                }
            }
        }
    }

    output[x] = tmpmax;
}
