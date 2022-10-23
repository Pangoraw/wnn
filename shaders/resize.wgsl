@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    {% for group in range(end=num_groups) %}

    let gdx = {{ num_groups }}u * global_id.x{% if group > 0 %} + {{ group }}u{% endif %}; // reduce index

    if gdx >= {{ o_lens[0] }}u {
        return;
    }

    let b = gdx / {{ o_strides[0][0] }}u;

    let rest = gdx % {{ o_strides[0][0] }}u;
    let out_chan = rest / {{ o_strides[0][1] }}u;

    let rest = rest % {{ o_strides[0][1] }}u;
    let out_y = rest / {{ o_strides[0][2] }}u;
    let out_x = rest % {{ o_strides[0][2] }}u;

    // coordinate_transformation_mode == "asymmetric"
    let in_x = u32(f32(out_x) / {{ xy_scales[1] }}.);
    let in_y = u32(f32(out_y) / {{ xy_scales[0] }}.);

    let input_index =
        b * {{ i_strides[0][0] }}u +
        out_chan * {{ i_strides[0][1] }}u +
        in_y * {{ i_strides[0][2] }}u +
        in_x * {{ i_strides[0][3] }}u;

    output[gdx] = input[input_index];

    {% endfor %}
}
