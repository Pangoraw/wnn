@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> scale: array<f32>;

@group(0) @binding(2)
var<storage, read> bias: array<f32>;

@group(0) @binding(3)
var<storage, read> input_mean: array<f32>;

@group(0) @binding(4)
var<storage, read> input_var: array<f32>;

@group(0) @binding(5)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    {% for group in range(end=num_groups) %}
    let gidx = {% if num_groups > 1 %}{{ num_groups }} *{% endif %} global_id.x{% if group > 0%} + {{ group }}u{%endif%}; 

    let channel_index = (gidx % {{ i_strides[0][0] }}u) / {{ i_strides[0][1] }}u;

    output[gidx] = scale[channel_index] * (input[gidx] - input_mean[channel_index]) /
                        sqrt(input_var[channel_index] + {{ epsilon }}) +
                        bias[channel_index];
    {% endfor %}
}