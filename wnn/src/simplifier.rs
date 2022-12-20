use anyhow::Result;

use crate::analyzer::{LogicalGraph, LogicalOp, LogicalOpType, PoolType, UnaryOpType};

/// Apply a simple folding of unary activations in compatible layers for layer which currently
/// supports it (Conv, Gemm).
pub(crate) fn simplify(log_graph: &mut LogicalGraph) -> Result<()> {
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
