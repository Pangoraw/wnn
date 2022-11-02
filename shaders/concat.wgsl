// TODO: Make variadic and optimize...
type T = {{ scalar }};

@group(0) @binding(0)
var<storage, read> input_left: array<T>;

@group(0) @binding(1)
var<storage, read> input_right: array<T>;

@group(0) @binding(2)
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

        output[gx] = input_left[index];
    } else {

    {% for dim in o_sizes[0] %}
        {% if loop.index0 == axis %}
        let dim_index = rest / {{ o_strides[0][loop.index0] }}u - {{ i_sizes[0][axis] }}u;
        {% else %}
        let dim_index = rest / {{ o_strides[0][loop.index0] }}u;
        {% endif %}

        index = index + dim_index * {{ i_strides[1][loop.index0] }}u;
        rest = rest % {{ o_strides[0][loop.index0] }}u;
    {% endfor %}

        output[gx] = input_right[index];
    }

    {% endfor %}
}
