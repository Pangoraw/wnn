use anyhow::Result;

use crate::analyzer::{
    get_desc, BufferHandle, LogicalGraph, LogicalOp, LogicalOpType, PoolType, UnaryOpType,
};

/// Apply different simplifications passes to the model.
pub(crate) fn simplify(log_graph: &mut LogicalGraph) -> Result<()> {
    remove_identities(log_graph);
    fold_activations(log_graph)?;
    Ok(())
}

fn replace_all_uses(log_graph: &mut LogicalGraph, mapping: (BufferHandle, BufferHandle)) {
    let (new_value, old_value) = mapping;
    for op in log_graph.ops.iter_mut() {
        for input in op.inputs.iter_mut() {
            if *input == old_value {
                *input = new_value;
            }
        }
    }
}

/// Remove "Identity" nodes from the graph and replace their outputs by its input in all nodes.
fn remove_identities(log_graph: &mut LogicalGraph) {
    let bufs = &log_graph.buffers;
    let ops = &mut log_graph.ops;

    let mut identity_mappings = Vec::new();

    ops.retain(|op| {
        if !matches!(op.op_type(), LogicalOpType::ReshapeOnly) {
            return true;
        }

        if op.inputs.len() != 1 || op.outputs.len() != 1 {
            return true;
        }

        let input = op.inputs[0];
        let output = op.outputs[0];
        if get_desc(bufs, input).shape != get_desc(bufs, output).shape {
            return true;
        }

        identity_mappings.push((input, output));

        false
    });

    // TODO: also clear entries in log_graph.buffers?
    for mapping in identity_mappings {
        replace_all_uses(log_graph, mapping);
    }
}

/// Apply a simple folding of unary activations in compatible layers for layer which currently
/// supports it (Conv, Gemm).
fn fold_activations(log_graph: &mut LogicalGraph) -> Result<()> {
    let mut index = -1isize;
    while index < log_graph.ops.len() as isize - 1 {
        index += 1;

        let op = &log_graph.ops[index as usize];

        if op.outputs.len() != 1 {
            continue;
        }

        let output = op.outputs[0];
        let usages = log_graph
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.inputs.iter().any(|input| *input == output))
            .collect::<Vec<(usize, &LogicalOp)>>();

        if usages.len() != 1 {
            continue;
        }

        let (op_to_fold_index, op_to_fold) = usages.first().unwrap();
        let op_to_fold_index = *op_to_fold_index;

        let act_to_fold = match op_to_fold.op_type() {
            LogicalOpType::Relu => Some(UnaryOpType::Relu),
            LogicalOpType::Cos => Some(UnaryOpType::Cos),
            LogicalOpType::Sin => Some(UnaryOpType::Sin),
            LogicalOpType::Exp => Some(UnaryOpType::Exp),
            LogicalOpType::Sqrt => Some(UnaryOpType::Sqrt),
            _ => continue,
        };
        let new_output = op_to_fold.outputs[0];

        {
            let op = &mut log_graph.ops[index as usize];
            match op.op_type_mut() {
                LogicalOpType::Pool {
                    ptype: PoolType::Conv,
                    activation: act @ None,
                    ..
                }
                | LogicalOpType::Gemm {
                    activation: act @ None,
                    ..
                } => {
                    *act = act_to_fold;
                    op.outputs[0] = new_output;
                }
                _ => continue,
            }
        }

        log_graph.ops.remove(op_to_fold_index);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::simplify;
    use crate::{
        analyzer::{builder::LogicalGraphBuilder, LogicalOpType},
        shape::Shape,
        tensor::TensorDesc,
    };

    #[test]
    fn test_remove_identities() -> anyhow::Result<()> {
        let mut builder = LogicalGraphBuilder::empty();
        let desc = TensorDesc::new(Shape::from(&[10]), crate::tensor::DataType::F32);

        let input = builder.add_input("input", desc.clone());
        let inter = builder.add_buffer("inter", desc.clone());
        let output = builder.add_output("output", desc);

        builder.add_op(
            "identity",
            LogicalOpType::ReshapeOnly,
            vec![input],
            vec![inter],
        );
        builder.add_op("cos", LogicalOpType::Cos, vec![inter], vec![output]);

        let mut graph = builder.build();
        simplify(&mut graph)?;

        if graph
            .ops
            .iter()
            .any(|op| matches!(op.op_type(), LogicalOpType::ReshapeOnly))
        {
            anyhow::bail!(
                "remove_identities pass did not work ({} ops)",
                graph.ops.len()
            );
        }

        if graph.ops[0].inputs[0] != input {
            anyhow::bail!("failed to replace use of {inter} by {input}");
        }

        Ok(())
    }
}
