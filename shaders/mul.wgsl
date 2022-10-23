@group(0) @binding(0)
var<storage, read> input_left: array<f32>;

@group(0) @binding(1)
var<storage, read> input_right: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    {% for group in range(end=num_groups) %}
    let gdx = {{ num_groups }}u * global_id.x{% if group > 0 %} + {{ group }}u{% endif %};

    if gdx >= {{ o_lens[0] }}u {
        return;
    }

    var index_left = 0u;
    var index_right = 0u;
    var rest = gdx;
    {% for dim in o_sizes[0] %}
        {% if dim > 1 %}
            let dim_index = rest / {{ o_strides[0][loop.index0] }}u; // a, b, c

            {% if bi_strides[0][loop.index0] > 0 %}
                index_left = index_left + dim_index * {{ bi_strides[0][loop.index0] }}u;
            {% endif %}

            {% if bi_strides[1][loop.index0] > 0 %}
                index_right = index_right + dim_index * {{ bi_strides[1][loop.index0] }}u;
            {% endif %}

            rest = rest % {{ o_strides[0][loop.index0] }}u;
        {% endif %}
    {% endfor %}

    output[gdx] = input_left[index_left] {{ op }} input_right[index_right];
    {% endfor %}
}
