// TODO: Make variadic and optimize...
type T = {{ scalar }};

{% for input in range(end=i_length) %}

@group(0) @binding({{ input }})
var<storage, read> input_{{ input }}: array<T>;

{% endfor %}

@group(0) @binding({{ i_length }})
var<storage, read_write> output: array<T>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    {% for group in range(end=num_groups) %}
    let gx = {{ num_groups }}u * global_id.x{% if group > 0 %} + {{ group }}u {% endif %};

    if gx >= {{ o_lens[0] }}u {
        return;
    }

    var rest = gx;
    {% for i in range(end=axis) %}
        rest = rest % {{ o_strides[0][i] }}u;
    {% endfor %}
    let axis_index = rest / {{ o_strides[0][axis] }}u;

    var index: u32;
    rest = gx;

    if axis_index < {{ i_sizes[0][axis] }}u {

    {% for dim in o_sizes[0] %}
        let dim_index = rest / {{ o_strides[0][loop.index0] }}u;
        index = index + dim_index * {{ i_strides[0][loop.index0] }}u;
        rest = rest % {{ o_strides[0][loop.index0] }}u;
    {% endfor %}

        output[gx] = input_0[index];
    }
    {% for axis_thresh in cum_inputs_size %}
    {% set input_index = loop.index0 + 1 %}
    {% if input_index == i_length - 1 %}
    else {
    {% else %}
    else if axis_index < {{ axis_thresh + i_sizes[input_index][axis] }}u {
    {% endif %}

    {% for dim in o_sizes[0] %}
        {% if loop.index0 == axis %}
        let dim_index = rest / {{ o_strides[0][loop.index0] }}u - {{ axis_thresh }}u;
        {% else %}
        let dim_index = rest / {{ o_strides[0][loop.index0] }}u;
        {% endif %}

        index = index + dim_index * {{ i_strides[1][loop.index0] }}u;
        rest = rest % {{ o_strides[0][loop.index0] }}u;
    {% endfor %}

        output[gx] = input_{{ input_index }}[index];
    }
    {% endfor %}

    {% endfor %}
}
