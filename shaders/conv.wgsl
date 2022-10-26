// The code for this kernel is heavily inspired by the code for webonnx/wonnx
// https://github.com/webonnx/wonnx/blob/master/wonnx/templates/pool/conv.wgsl
// TODO: Support dilation
type T = {{ scalar }};


@group(0) @binding(0)
var<storage, read> input: array<T>;

@group(0) @binding(1)
var<storage, read> weight: array<T>;

{% if i_lens | length == 3 %}
    @group(0) @binding(2)
    var<storage, read> bias: array<T>;

    @group(0) @binding(3)
    var<storage, read_write> output: array<T>;
{% else %}
    @group(0) @binding(2)
    var<storage, read_write> output: array<T>;
{% endif %}

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    {% for group in range(end=num_groups) %}

    let x = {{ num_groups }}u * global_id.x{% if group > 0 %} + {{ group }}u{% endif %};

    if x >= {{ o_lens[0] }}u {
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
    let root_weight_index = out_chan * {{ i_strides[1][0] }}u;

    var tmpsum: T = {% if i_lens | length == 3 %}bias[out_chan]{% else %}0.{% endif %};
    for (var input_chan: u32 = 0u; input_chan < {{ i_sizes[0][1] }}u; input_chan = input_chan + 1u) {
        let base_input_index =
            root_input_index
            + input_chan * {{ i_strides[0][1] }}u;
        let base_weight_index =
            root_weight_index +
            input_chan * {{ i_strides[1][1] }}u;

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

                        let weight_index =
                            base_weight_index +
                            i * {{ i_strides[1][2] }}u +
                            j * {{ i_strides[1][3] }}u;

                        tmpsum = tmpsum + input[input_index] * weight[weight_index];
                    }
                }
            }
        }
    }

    output[x] = tmpsum;

    {% endfor %}
}
