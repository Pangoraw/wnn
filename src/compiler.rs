// This file is very much inspired/copied from webonnx/wonnx

use std::collections::HashMap;

use anyhow::anyhow;
use lazy_static::lazy_static;

use crate::{
    gpu::TensorDesc,
    onnx,
    utils::{get_attr_float, get_attr_int, get_attr_ints},
};

lazy_static! {
    static ref SHADER_FILES: HashMap<&'static str, &'static str> = HashMap::from_iter([
        ("matmul", include_str!("../shaders/matmul.wgsl")),
        ("activation", include_str!("../shaders/activation.wgsl")),
        ("conv", include_str!("../shaders/conv.wgsl")),
        ("maxpool", include_str!("../shaders/maxpool.wgsl")),
        ("transpose", include_str!("../shaders/transpose.wgsl")),
        ("mul", include_str!("../shaders/mul.wgsl")),
        (
            "instancenormalization",
            include_str!("../shaders/instancenormalization.wgsl")
        ),
    ]);
}

/// We should be able to convert to wgsl String
/// and get the workgroup dispatch informations
/// from a compiled shader invocation.
pub(crate) struct ShaderInvocation {
    file_name: &'static str,
    context: tera::Context,
    dispatch: (u32, u32, u32),
}

impl ShaderInvocation {
    pub(crate) fn to_wgsl(&self) -> anyhow::Result<String> {
        let template = SHADER_FILES
            .get(&self.file_name)
            .ok_or_else(|| anyhow!("invalid template {}", self.file_name))?;
        Ok(tera::Tera::one_off(template, &self.context, false)?)
    }

    pub(crate) fn dispatch(&self) -> (u32, u32, u32) {
        self.dispatch
    }
}

// NOTE: This currently assumes that sizes is that of a contiguous tensor.
// TODO: Implement strides in shape inference/kernels.
fn ints_to_strides(sizes: &mut Vec<i64>) {
    let mut current = 1;
    sizes.iter_mut().rev().for_each(|dim| {
        let x = *dim;
        *dim = current;
        current *= x;
    });
}

type TensorDescs<'a> = HashMap<&'a str, TensorDesc>;

fn compute_strides(descs: &TensorDescs, names: &[String]) -> anyhow::Result<Vec<Vec<i64>>> {
    names
        .iter()
        .map(|node| {
            descs[node.as_str()].shape.as_ints().map(|mut sizes| {
                ints_to_strides(&mut sizes);
                sizes
            })
        })
        .collect::<anyhow::Result<Vec<Vec<i64>>>>()
}

fn base_context(node: &onnx::NodeProto, descs: &TensorDescs) -> anyhow::Result<tera::Context> {
    let mut context = tera::Context::new();
    context.insert("i_length", &node.input.len());
    context.insert("o_length", &node.output.len());
    context.insert(
        "i_lens",
        &node
            .input
            .iter()
            .map(|input| descs[input.as_str()].shape.numel().unwrap())
            .collect::<Vec<usize>>(),
    );
    context.insert(
        "o_lens",
        &node
            .output
            .iter()
            .map(|output| descs[output.as_str()].shape.numel().unwrap())
            .collect::<Vec<usize>>(),
    );
    context.insert(
        "i_sizes",
        &node
            .input
            .iter()
            .map(|input| descs[input.as_str()].shape.as_ints())
            .collect::<anyhow::Result<Vec<Vec<i64>>>>()?,
    );
    context.insert("i_strides", &compute_strides(descs, &node.input)?);
    context.insert(
        "o_sizes",
        &node
            .output
            .iter()
            .map(|input| descs[input.as_str()].shape.as_ints())
            .collect::<anyhow::Result<Vec<Vec<i64>>>>()?,
    );
    context.insert("o_strides", &compute_strides(descs, &node.output)?);
    Ok(context)
}

const MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP: u64 = 256;
// const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u64 = 65535;

pub(crate) fn compile_node(
    node: &onnx::NodeProto,
    descs: &TensorDescs,
) -> anyhow::Result<ShaderInvocation> {
    let mut context = base_context(node, descs)?;
    let invoc = match node.op_type() {
        "MatMul" | "Gemm" => {
            let trans_a = get_attr_int(node, "transA").unwrap_or(0);
            let trans_b = get_attr_int(node, "transB").unwrap_or(0);
            context.insert("trans_a", &trans_a);
            context.insert("trans_b", &trans_b);

            let alpha = get_attr_float(node, "alpha").unwrap_or(1.);
            let beta = get_attr_float(node, "beta").unwrap_or(1.);
            context.insert("alpha", &alpha);
            context.insert("beta", &beta);

            let workgroup_x = descs[node.output[0].as_str()].shape.numel().unwrap() as u64;
            let dispatch_x = ceil(workgroup_x, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);

            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(workgroup_x),
            );

            ShaderInvocation {
                file_name: "matmul",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        "InstanceNormalization" => {
            let input_shape = &descs[node.input[0].as_str()].shape;
            let epsilon = get_attr_float(node, "epsilon").unwrap_or(1e-4);
            context.insert("epsilon", &epsilon);

            ShaderInvocation {
                file_name: "instancenormalization",
                context,
                dispatch: (
                    input_shape.concrete_size(0)? as _,
                    input_shape.concrete_size(1)? as _,
                    1,
                ),
            }
        }
        act @ ("Relu" | "Sigmoid" | "Reshape" | "Identity" | "Flatten" | "Squeeze"
        | "Unsqueeze") => {
            let out_shape = descs[node.output[0].as_str()].shape.numel().unwrap() as u64;
            let dispatch_x = ceil(out_shape, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);
            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(out_shape),
            );
            context.insert(
                "activation",
                match act {
                    "Relu" => "max(x, 0.)",
                    "Sigmoid" => "1. / (1. + exp(-x))",
                    "Reshape" | "Identity" | "Flatten" | "Squeeze" | "Unsqueeze" => "x", // identity
                    _ => unreachable!(),
                },
            );

            ShaderInvocation {
                file_name: "activation",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        "Conv" => {
            let weight_shape = &descs[node.input[1].as_str()].shape;
            let out_shape = descs[node.output[0].as_str()].shape.numel().unwrap() as u64;

            let dispatch_x = ceil(out_shape, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);

            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(out_shape),
            );
            context.insert(
                "half_kernel_size",
                &[
                    weight_shape.concrete_size(-2)? / 2,
                    weight_shape.concrete_size(-1)? / 2,
                ],
            );

            ShaderInvocation {
                file_name: "conv",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        "Transpose" => {
            let input_a = &descs[node.input[0].as_str()];
            let perm = get_attr_ints(node, "perm").ok_or_else(|| anyhow!("could not find perm"))?;

            let out_shape = descs[node.output[0].as_str()].shape.numel().unwrap() as u64;
            let dispatch_x = ceil(out_shape, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);
            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(out_shape),
            );
            let mut i_strides = input_a.shape.as_ints()?;
            ints_to_strides(&mut i_strides);
            context.insert(
                "pi_strides",
                &perm
                    .iter()
                    .map(|p| i_strides[*p as usize])
                    .collect::<Vec<i64>>(),
            );

            ShaderInvocation {
                file_name: "transpose",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        op @ ("Mul" | "Add" | "Div" | "Sub") => {
            // TODO: make case with scalar inlining?

            let mut input_a = descs[node.input[0].as_str()].shape.clone();
            let mut input_b = descs[node.input[1].as_str()].shape.clone();
            input_a.pad_left_to(input_b.ndims());
            input_b.pad_left_to(input_a.ndims());

            let out_shape = &descs[node.output[0].as_str()].shape;
            let out_numel = out_shape.numel().unwrap() as u64;
            let dispatch_x = ceil(out_numel, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);
            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(out_numel),
            );

            let i_strides = [&input_a, &input_b]
                .iter()
                .map(|node| {
                    node.as_ints().map(|mut sizes| {
                        ints_to_strides(&mut sizes);
                        sizes
                    })
                })
                .collect::<anyhow::Result<Vec<Vec<i64>>>>()?;
            let bi_strides: Vec<Vec<i64>> = vec![
                (0isize..out_shape.ndims() as isize)
                    .map(|dim| {
                        if input_a.size(dim) == out_shape.size(dim) {
                            i_strides[0][dim as usize]
                        } else {
                            0
                        }
                    })
                    .collect(),
                (0isize..out_shape.ndims() as isize)
                    .map(|dim| {
                        if input_b.size(dim) == out_shape.size(dim) {
                            i_strides[1][dim as usize]
                        } else {
                            0
                        }
                    })
                    .collect(),
            ];
            context.insert("bi_strides", &bi_strides);
            context.insert(
                "op",
                match op {
                    "Add" => "+",
                    "Mul" => "*",
                    "Div" => "/",
                    "Sub" => "-",
                    _ => unreachable!(),
                },
            );

            ShaderInvocation {
                file_name: "mul",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        "MaxPool" => {
            let kernel_size = get_attr_ints(node, "kernel_shape").unwrap();
            let out_shape = descs[node.output[0].as_str()].shape.numel().unwrap() as u64;
            let dispatch_x = ceil(out_shape, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);

            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(out_shape),
            );
            context.insert(
                "half_kernel_size",
                &[kernel_size[0] / 2, kernel_size[1] / 2],
            );

            ShaderInvocation {
                file_name: "maxpool",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        op => unimplemented!("{op}"),
    };

    Ok(invoc)
}

fn ceil(x: u64, factor: u64) -> u64 {
    x / factor + (x % factor != 0) as u64
}
