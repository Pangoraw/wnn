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

    var input_index = 0u; // find the corresponding input pixel
    var rest = gdx;
    {% for dim in o_sizes[0] %}
        {% if dim > 1 %}
            let dim_index = rest / {{ o_strides[0][loop.index0] }}u; // a, b, c

            {% if i_strides[0][loop.index0] > 1 %}
                input_index = input_index + dim_index * {{ i_strides[0][loop.index0] }}u;
            {% endif %}

            rest = rest % {{ o_strides[0][loop.index0] }}u;
        {% endif %}
    {% endfor %}

    output[gdx] = input[input_index];

    {% endfor %}
}
