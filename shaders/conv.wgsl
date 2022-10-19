@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

{% if i_lens | length == 3 %}
    @group(0) @binding(2)
    var<storage, read> bias: array<f32>;

    @group(0) @binding(3)
    var<storage, read_write> output: array<f32>;
{% else %}
    @group(0) @binding(2)
    var<storage, read_write> output: array<f32>;
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

    let out_x = rest / {{ o_strides[0][2] }}u;
    rest = rest % {{ o_strides[0][2] }}u;

    let out_y = rest / {{ o_strides[0][3] }}u;

    let root_input_index = b * {{ i_strides[0][1] }}u;
    let root_weight_index = out_chan * {{ i_strides[1][0] }}u;

    var tmpsum: f32 = 0.0;
    for (var c: u32 = 0u; c < {{ i_sizes[0][1] }}u; c = c + 1u) {
        let base_input_index =
            root_input_index
            + c * {{ i_strides[0][1] }}u 
            + (out_y * {{ k_strides[0] }}u) * {{ i_strides[0][2] }}u
            + (out_x * {{ k_strides[1] }}u) * {{ i_strides[0][3] }}u
            - {{ half_kernel_size[0] * i_strides[0][2] + half_kernel_size[1] * i_strides[0][3] }}u;
        let base_weight_index = root_weight_index + c * {{ i_strides[1][1] }}u;

        for (var i: u32 = 0u; i < {{ kernel_size[0] }}u; i = i + 1u) {
            for (var j: u32 = 0u; j < {{ kernel_size[1] }}u; j = j + 1u) {
                // FIXME: this is not checking the right thing, we must check if INPUT is out of bounds
                if out_x + j >= {{ i_sizes[0][2] }}u || out_x + j < 0u { continue; }
                if out_y + i >= {{ i_sizes[0][3] }}u || out_x + j < 0u { continue; }

                let input_idx = base_input_index + j * {{ i_strides[0][3] }}u + i * {{ i_strides[0][2] }}u;
                let weight_idx = base_weight_index + j * {{ i_strides[1][3] }}u + i * {{ i_strides[1][2] }}u;

                tmpsum = tmpsum + input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[x] = tmpsum{% if i_lens | length == 3 %} + bias[out_chan]{% endif %};

    {% endfor %}
}
