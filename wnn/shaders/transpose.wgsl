type T = {{ scalar }};

@group(0) @binding(0)
var<storage, read> input: array<T>;

@group(0) @binding(1)
var<storage, read_write> output: array<T>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gdx = global_id.x;

    if gdx >= {{ o_lens[0] }}u {
        return;
    }

    var input_index = 0u;
    var rest = gdx;
    {% for dim in o_sizes[0] %}
        {
        let dim_index = rest / {{ o_strides[0][loop.index0] }}u;

        {% if pi_strides[loop.index0] > 0 %}
            input_index = input_index + dim_index * {{ pi_strides[loop.index0] }}u;
        {% endif %}

        rest = rest % {{ o_strides[0][loop.index0] }}u;
        }
    {% endfor %}

    output[gdx] = input[input_index];
}
