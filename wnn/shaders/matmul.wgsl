type T = {{ scalar }};

@group(0) @binding(0)
var<storage, read> input_left: array<T>;

@group(0) @binding(1)
var<storage, read> input_right: array<T>;

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
    let gdx = {% if num_groups != 1 %}{{ num_groups }}u * {% endif %}global_id.x{% if group > 0 %} + {{ group }}u {% endif %};

    if gdx >= {{ o_lens[0] }}u {
        return;
    }

    let i = gdx % {{ o_sizes[0][1] }}u;
    let j = gdx / {{ o_sizes[0][1] }}u;

    var tmpsum = T();
    for (var k: u32 = 0u; k < {{ i_sizes[1][trans_b] }}u; k = k + 1u) {
        let left_idx = {% if trans_a == 0 %}
            k + j * {{ i_sizes[0][1] }}u
        {% else %}
            j + k * {{ i_sizes[0][1] }}u
        {% endif %};

        let right_idx = {% if trans_b == 0 %}
            x + k * {{ i_sizes[1][1] }}u
        {% else %}
            k + i * {{ i_sizes[1][1] }}u
        {% endif %};

        tmpsum = tmpsum + input_left[left_idx] * input_right[right_idx];
    }

    let x = {% if alpha != 1.0 %} {{ alpha }} * {% endif %} tmpsum {% if i_lens | length == 3 %}
        + {% if beta != 1.0 %} {{ beta }} * {% endif %} bias[i]
    {% endif %};
    output[gdx] = {% if act %}{{ act }}{% else %}x{% endif %};

    {% endfor %}
}
