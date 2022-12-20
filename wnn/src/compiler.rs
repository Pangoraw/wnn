// This file is very much inspired/copied from webonnx/wonnx

use std::collections::HashMap;

use anyhow::{anyhow, bail};
use lazy_static::lazy_static;

use crate::analyzer::{BufferHandle, LogicalGraph, LogicalOp, LogicalOpType, PoolType};

lazy_static! {
    static ref SHADER_FILES: HashMap<&'static str, &'static str> = HashMap::from_iter([
        ("matmul", include_str!("../shaders/matmul.wgsl")),
        ("slice", include_str!("../shaders/slice.wgsl")),
        ("activation", include_str!("../shaders/activation.wgsl")),
        ("conv", include_str!("../shaders/conv.wgsl")),
        ("concat", include_str!("../shaders/concat.wgsl")),
        ("maxpool", include_str!("../shaders/maxpool.wgsl")),
        ("transpose", include_str!("../shaders/transpose.wgsl")),
        ("broadcast", include_str!("../shaders/broadcast.wgsl")),
        ("resize", include_str!("../shaders/resize.wgsl")),
        ("softmax", include_str!("../shaders/softmax.wgsl")),
        ("where", include_str!("../shaders/where.wgsl")),
        (
            "constantofshape",
            include_str!("../shaders/constantofshape.wgsl")
        ),
        (
            "globalaveragepool",
            include_str!("../shaders/globalaveragepool.wgsl")
        ),
        (
            "batchnormalization",
            include_str!("../shaders/batchnormalization.wgsl")
        ),
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
    pub(crate) fn to_wgsl(&self, enable_f16: bool) -> anyhow::Result<String> {
        let template = SHADER_FILES
            .get(&self.file_name)
            .ok_or_else(|| anyhow!("invalid shader template '{}'", self.file_name))?;
        let wgsl = tera::Tera::one_off(template, &self.context, false)?;
        if enable_f16 {
            Ok(format!("enable f16;\n\n{wgsl}"))
        } else {
            Ok(wgsl)
        }
    }

    pub(crate) fn dispatch(&self) -> (u32, u32, u32) {
        self.dispatch
    }
}

// NOTE: This currently assumes that sizes is that of a contiguous tensor.
// TODO: Implement strides in shape inference/kernels.
fn ints_to_strides(sizes: &mut [i64]) {
    let mut current = 1;
    sizes.iter_mut().rev().for_each(|dim| {
        let x = *dim;
        *dim = current;
        current *= x;
    });
}

fn compute_strides(descs: &LogicalGraph, names: &[BufferHandle]) -> anyhow::Result<Vec<Vec<i64>>> {
    names
        .iter()
        .map(|node| {
            descs.get_desc(*node).shape.as_ints().map(|mut sizes| {
                ints_to_strides(&mut sizes);
                sizes
            })
        })
        .collect::<anyhow::Result<Vec<Vec<i64>>>>()
}

fn base_context(op: &LogicalOp, descs: &LogicalGraph) -> anyhow::Result<tera::Context> {
    let mut context = tera::Context::new();
    context.insert("scalar", "f32");
    context.insert("i_length", &op.inputs.len());
    context.insert("o_length", &op.outputs.len());
    context.insert(
        "i_lens",
        &op.inputs
            .iter()
            .map(|input| descs.get_desc(*input).shape.numel().unwrap())
            .collect::<Vec<usize>>(),
    );
    context.insert(
        "o_lens",
        &op.outputs
            .iter()
            .map(|output| descs.get_desc(*output).shape.numel().unwrap())
            .collect::<Vec<usize>>(),
    );
    context.insert(
        "i_sizes",
        &op.inputs
            .iter()
            .map(|input| descs.get_desc(*input).shape.as_ints())
            .collect::<anyhow::Result<Vec<Vec<i64>>>>()?,
    );
    context.insert("i_strides", &compute_strides(descs, &op.inputs)?);
    context.insert(
        "o_sizes",
        &op.outputs
            .iter()
            .map(|input| descs.get_desc(*input).shape.as_ints())
            .collect::<anyhow::Result<Vec<Vec<i64>>>>()?,
    );
    context.insert("o_strides", &compute_strides(descs, &op.outputs)?);
    Ok(context)
}

const MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP: u64 = 256;
const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u64 = 65535;

fn dispatch_invocations(n_invocs: u64) -> (u64, u64, u64) {
    let mut dispatch_x = ceil(n_invocs, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);

    let num_groups = if dispatch_x > MAX_COMPUTE_WORKGROUPS_PER_DIMENSION {
        dispatch_x = MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
        ceil(dispatch_x, MAX_COMPUTE_WORKGROUPS_PER_DIMENSION)
    } else {
        1
    };

    (
        MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(n_invocs),
        dispatch_x,
        num_groups,
    )
}

/// Used to compute the dispatches in the case of `n_invocs`, usually the number
/// of elements in the output. This adds the following to the context which must use them
/// in the shader, otherwise the result may not be what is expected:
///  - `workgroup_x`
///  - `num_groups`
fn add_invocs_to_context(context: &mut tera::Context, n_invocs: u64) -> u64 {
    let (workgroup_x, dispatch_x, num_groups) = dispatch_invocations(n_invocs);
    context.insert("workgroup_x", &workgroup_x);
    context.insert("num_groups", &num_groups);
    dispatch_x
}

/// Some ops have an optional activation function applied at the end, so we add its
/// implementation f(x) -> ... to the context if it is available.
fn maybe_add_activation(
    context: &mut tera::Context,
    activation: &Option<crate::analyzer::UnaryOpType>,
) {
    if let Some(act) = activation {
        context.insert(
            "act",
            match act {
                crate::analyzer::UnaryOpType::Relu => "max(x, T())",
                crate::analyzer::UnaryOpType::Cos => "cos(x)",
                crate::analyzer::UnaryOpType::Sin => "sin(x)",
                crate::analyzer::UnaryOpType::Exp => "exp(x)",
                crate::analyzer::UnaryOpType::Sqrt => "sqrt(x)",
                // other => unimplemented!("{:?}", other),
            },
        );
    }
}

/// Generates the relevant `ShaderInvocation` for a given node using its input descriptions and
/// attributes, the corresponding wgsl code can then be retrieved using `ShaderInvocation::to_wgsl()`.
pub(crate) fn compile_node(
    logical_op: &LogicalOp,
    logical_graph: &LogicalGraph,
) -> anyhow::Result<ShaderInvocation> {
    let mut context = base_context(logical_op, logical_graph)?;
    let invoc = match logical_op.op_type() {
        LogicalOpType::Gemm {
            trans_a,
            trans_b,
            alpha,
            beta,
            activation,
        } => {
            context.insert("trans_a", &i32::from(*trans_a));
            context.insert("trans_b", &i32::from(*trans_b));

            context.insert("alpha", &alpha);
            context.insert("beta", &beta);

            let n_invocs = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);
            maybe_add_activation(&mut context, activation);

            ShaderInvocation {
                file_name: "matmul",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::BatchNormalization { epsilon } => {
            context.insert("epsilon", &epsilon);

            let n_invocs = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);

            ShaderInvocation {
                file_name: "batchnormalization",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::InstanceNormalization { epsilon } => {
            let input_shape = &logical_graph.get_desc(logical_op.inputs[0]).shape;
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
        LogicalOpType::Resize {
            transformation_mode,
        } => {
            /*
            if !matches!(get_attr_string(logical_op, "mode"), Some("nearest")) {
                anyhow::bail!(
                    "unsupported resizing mode {:?}",
                    get_attr_string(logical_op, "mode")
                );
            }*/

            match transformation_mode {
                crate::analyzer::ResizeCoordinateTransformationMode::Asymmetric => {} // other => bail!("unsupported coordinate_transformation_mode '{:?}'", other),
            }

            let input_shape = &logical_graph.get_desc(logical_op.inputs[0]).shape;
            let output_shape = &logical_graph.get_desc(logical_op.outputs[0]).shape;
            let n_invocs = output_shape.numel().unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);

            let xy_scales = [
                output_shape.concrete_size(2)? as f32 / input_shape.concrete_size(2)? as f32,
                output_shape.concrete_size(3)? as f32 / input_shape.concrete_size(3)? as f32,
            ];
            context.insert("xy_scales", &xy_scales);

            ShaderInvocation {
                file_name: "resize",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::Softmax { axis } => {
            let out_shape = &logical_graph.get_desc(logical_op.outputs[0]).shape;
            let real_axis = out_shape.reldim(*axis as isize);
            context.insert("axis", &real_axis);

            let num_reduce =
                (out_shape.numel().unwrap() / out_shape.concrete_size(*axis as _)?) as u64;

            let dispatch_x = ceil(num_reduce, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);
            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(num_reduce),
            );

            ShaderInvocation {
                file_name: "softmax",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::Slice {
            axes,
            starts,
            ends,
            steps,
        } => {
            let out_shape = &logical_graph.get_desc(logical_op.outputs[0]).shape;
            context.insert("axes", axes);
            context.insert("starts", starts);
            context.insert("ends", ends);
            context.insert("steps", steps);

            let n_invocs = out_shape.numel().ok_or_else(|| {
                anyhow!(
                    "could not get concrete shape for %{}",
                    logical_op.outputs[0]
                )
            })?;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs as _);

            ShaderInvocation {
                file_name: "slice",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::ConstantOfShape { constant } => {
            let out_desc = &logical_graph.get_desc(logical_op.outputs[0]);
            let out_shape = &out_desc.shape;
            let n_invocs = out_shape.numel().ok_or_else(|| {
                anyhow!(
                    "could not get concrete shape for %{}",
                    logical_op.outputs[0]
                )
            })?;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs as u64);

            let (value, scalar) = (format!("{}", constant), "f32");
            context.insert("value", &value);
            context.insert("scalar", scalar);

            ShaderInvocation {
                file_name: "constantofshape",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::Concat { axis } => {
            let out_shape = &logical_graph.get_desc(logical_op.outputs[0]).shape;
            let n_invocs = out_shape.numel().ok_or_else(|| {
                anyhow!(
                    "could not get concrete shape for %{}",
                    logical_op.outputs[0]
                )
            })?;

            if logical_op.inputs.len() != 2 {
                bail!(
                    "unsupported number of input {} for Concat",
                    logical_op.inputs.len()
                );
            }

            let dispatch_x = add_invocs_to_context(&mut context, n_invocs as u64);

            let axis = out_shape.reldim(*axis as isize);
            context.insert("axis", &axis);

            ShaderInvocation {
                file_name: "concat",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::Cast { target_type } => {
            let n_invocs = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);

            let target_type = target_type.to_str();
            context.insert("scalar_output", target_type);
            context.insert("activation", &format!("{}(x)", target_type));

            ShaderInvocation {
                file_name: "activation",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        act @ (LogicalOpType::Cos
        | LogicalOpType::Sin
        | LogicalOpType::Relu
        | LogicalOpType::LeakyRelu { .. }
        | LogicalOpType::Sigmoid) => {
            let n_invocs = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);

            match act {
                LogicalOpType::LeakyRelu { alpha } => {
                    context.insert("activation", &format!("{alpha} * x"))
                }
                _ => {
                    context.insert(
                        "activation",
                        match act {
                            // f(x) =
                            LogicalOpType::Relu => "max(x, 0.)",
                            LogicalOpType::Sigmoid => "1. / (1. + exp(-x))",
                            LogicalOpType::Cos => "cos(x)",
                            LogicalOpType::Sin => "sin(x)",
                            _ => unreachable!(),
                        },
                    );
                }
            }

            ShaderInvocation {
                file_name: "activation",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::Transpose { perm } => {
            let input_a = &logical_graph.get_desc(logical_op.inputs[0]);

            let out_shape = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
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
        LogicalOpType::Where => {
            // Where(cond, a, b)
            let n_invocs = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);

            ShaderInvocation {
                file_name: "where",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        op @ (LogicalOpType::Equal
        | LogicalOpType::Mul
        | LogicalOpType::Add
        | LogicalOpType::Div
        | LogicalOpType::Sub) => {
            // TODO: make case with scalar inlining?

            let mut input_a = logical_graph.get_desc(logical_op.inputs[0]).shape.clone();
            let mut input_b = logical_graph.get_desc(logical_op.inputs[1]).shape.clone();
            input_a.pad_left_to(input_b.ndims());
            input_b.pad_left_to(input_a.ndims());

            let out_shape = &logical_graph.get_desc(logical_op.outputs[0]).shape;

            let n_invocs = out_shape.numel().unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);

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
                    LogicalOpType::Add => "+",
                    LogicalOpType::Mul => "*",
                    LogicalOpType::Div => "/",
                    LogicalOpType::Sub => "-",
                    LogicalOpType::Equal => "==",
                    _ => unreachable!(),
                },
            );

            if matches!(op, LogicalOpType::Equal) {
                log::warn!("compiling Equal: probably not supported (bool is not numeric)");
                context.insert("cast", &true);
            }

            ShaderInvocation {
                file_name: "broadcast",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::GlobalAveragePool => {
            let out_shape = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = ceil(out_shape, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);
            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(out_shape),
            );

            ShaderInvocation {
                file_name: "globalaveragepool",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::Pool {
            ptype: PoolType::Conv,
            group: _,
            dilations: _,
            k_strides,
            pads,
            kernel_shape,
            activation,
        } => {
            context.insert("kernel_shape", kernel_shape);
            context.insert("pads", pads);
            context.insert("k_strides", k_strides);

            let n_invocs = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = add_invocs_to_context(&mut context, n_invocs);

            maybe_add_activation(&mut context, activation);

            ShaderInvocation {
                file_name: "conv",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        LogicalOpType::Pool {
            ptype: PoolType::Max,
            group: _,
            dilations: _,
            k_strides,
            pads,
            kernel_shape,
            activation: None,
        } => {
            context.insert("kernel_shape", kernel_shape);
            context.insert("pads", pads);
            context.insert("k_strides", k_strides);

            let out_shape = logical_graph
                .get_desc(logical_op.outputs[0])
                .shape
                .numel()
                .unwrap() as u64;
            let dispatch_x = ceil(out_shape, MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);

            context.insert(
                "workgroup_x",
                &MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP.min(out_shape),
            );

            ShaderInvocation {
                file_name: "maxpool",
                context,
                dispatch: (dispatch_x as _, 1, 1),
            }
        }
        op => bail!("unimplemented {:?}", op),
    };

    Ok(invoc)
}

/// These ops get a special handling and should not result in a ShaderInvocation
/// Basically, we only currently support Shape in static analysis and Constant are
/// added as initializer before any allocation happens.
pub(crate) fn is_untracked_op(op_type: &LogicalOpType) -> bool {
    matches!(op_type, LogicalOpType::Shape | LogicalOpType::Constant)
}

fn ceil(x: u64, factor: u64) -> u64 {
    x / factor + (x % factor != 0) as u64
}
