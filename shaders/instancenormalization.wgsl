type T = {{ scalar }};

@group(0) @binding(0)
var<storage, read> input: array<T>;

@group(0) @binding(1)
var<storage, read> scale: array<T>;

@group(0) @binding(2)
var<storage, read> bias: array<T>;

@group(0) @binding(3)
var<storage, read_write> output: array<T>;

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; // batch index
    let gidy = global_id.y; // channel_index

    let start = gidx * {{ i_strides[0][0] }}u + gidy * {{ i_strides[0][1] }}u;

    var sum: T = T();
    var sum2: T = T();
    for (var k: u32 = 0u; k < {{ i_strides[0][1] }}u; k = k + 1u) {
        let element = input[start + k];
        sum += element;
        sum2 += element * element;
    }
    let mean = sum / {{ i_strides[0][1] }}.;
    let variance = sum2 / {{ i_strides[0][1] }}. - mean * mean;
    let denom = sqrt({{ epsilon }} + variance);

    for (var k: u32 = 0u; k < {{ i_strides[0][1] }}u; k = k + 1u) {
        let i = start + k;
        let element = input[i];

        if i >= {{ o_lens[0] }}u {
            return;
        }

        output[i] = scale[gidy] * (element - mean) / denom + bias[gidy];
    }
}
