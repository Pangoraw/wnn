type T = {{ scalar }};
type T2 = {% if scalar_output %}{{ scalar_output }}{% else %}{{ scalar }}{% endif %};

@group(0) @binding(0)
var<storage, read> input: array<T>;

@group(0) @binding(1)
var<storage, read_write> output: array<T2>;

@compute @workgroup_size({{ workgroup_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    {% for group in range(end=num_groups) %}
    let idx = {{ num_groups }}u * global_id.x{% if group > 0 %} + {{ group }}u{% endif %};

    if idx >= {{ o_lens[0] }}u {
        return;
    }

    let x = input[idx];
    output[idx] = {{ activation }};

    {% endfor %}
}
