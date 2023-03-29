use anyhow::{anyhow, Context};
use protobuf::Message;
use structopt::StructOpt;

use wnn::{onnx, tensor::DataType, CompiledModel, EvalOutput, InitMode};

mod npy;

#[derive(StructOpt, Debug)]
struct Args {
    #[structopt(default_value = "./sd-v1-5-onnx/vae_decoder_sim.onnx")]
    input_model: std::path::PathBuf,
    dump_folder: Option<std::path::PathBuf>,

    #[structopt(long, short, default_value = "ones")]
    init: String,

    #[structopt(long, short, help = "seq:10,N:10")]
    dim_mappings: Option<String>,
}

async fn compile_and_eval(
    graph: &onnx::GraphProto,
    inputs: Vec<&[u8]>,
    dim_mappings: Option<std::collections::HashMap<&str, i64>>,
) -> anyhow::Result<EvalOutput> {
    let compiled_model = CompiledModel::new(graph, dim_mappings).await?;
    compiled_model.eval_graph(&inputs).await
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::from_args();
    let filename = args.input_model;

    let mut onnx_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&filename)
        .with_context(|| format!("while opening file {}", filename.display()))?;
    let model = onnx::ModelProto::parse_from_reader(&mut onnx_file)?;

    let init_mode = match args.init.as_str() {
        "ones" => InitMode::Ones,
        "range" => InitMode::Range,
        other => InitMode::File(other),
    };

    let dim_mappings: Option<std::collections::HashMap<&str, i64>> = args
        .dim_mappings
        .as_ref()
        .map(|map| wnn::parse_mappings(map));

    let inputs = model
        .graph
        .input
        .iter()
        .map(|input| -> anyhow::Result<Vec<u8>> {
            let dtype = DataType::from_int(input.type_.tensor_type().elem_type())?;
            let numel = input
                .type_
                .tensor_type()
                .shape
                .dim
                .iter()
                .map(|dim| match (&dim.value, &dim_mappings) {
                    (Some(onnx::tensor_shape_proto::dimension::Value::DimValue(val)), _) => *val,
                    (
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(dim)),
                        Some(dim_mappings),
                    ) => {
                        let Some(val) = dim_mappings.get(dim.as_str()) else {
                            panic!("dim {dim} is not present in dimension mappings.");
                        };
                        *val
                    }
                    (Some(onnx::tensor_shape_proto::dimension::Value::DimParam(dim)), None) => {
                        panic!(
                            "dim {dim} is not concrete and no dimension mappings were provided."
                        );
                    }
                    _ => unreachable!(),
                })
                .reduce(|a, b| a * b)
                .unwrap();

            let bytes: Vec<u8> = match (&init_mode, dtype) {
                (InitMode::Ones, DataType::F32) => bytemuck::cast_slice(
                    &std::iter::repeat(1.0)
                        .take(numel as _)
                        .collect::<Vec<f32>>(),
                )
                .to_vec(),
                (InitMode::Ones, DataType::F64) => bytemuck::cast_slice(
                    &std::iter::repeat(1.0)
                        .take(numel as _)
                        .collect::<Vec<f64>>(),
                )
                .to_vec(),
                (InitMode::Ones, DataType::I64) => bytemuck::cast_slice(
                    &std::iter::repeat(1).take(numel as _).collect::<Vec<i64>>(),
                )
                .to_vec(),
                (InitMode::Range, DataType::F32) => bytemuck::cast_slice(
                    &(0..numel)
                        .map(|i| i as f32)
                        .take(numel as _)
                        .collect::<Vec<f32>>(),
                )
                .to_vec(),
                (InitMode::Range, DataType::F64) => bytemuck::cast_slice(
                    &(0..numel)
                        .map(|i| i as f64)
                        .take(numel as _)
                        .collect::<Vec<f64>>(),
                )
                .to_vec(),
                (InitMode::SliceF32(slice), DataType::F32) => bytemuck::cast_slice(slice).to_vec(),
                (InitMode::File(path), _) if path.ends_with(".npy") => {
                    let (_, data) = npy::read_from_file(path)
                        .with_context(|| anyhow!("when open file {}", path))?;
                    data
                }
                (init, dtype) => {
                    anyhow::bail!("invalid initialization '{:?}' for type {dtype}", init)
                }
            };
            Ok(bytes)
        })
        .collect::<anyhow::Result<Vec<Vec<u8>>>>()?;

    let outputs = pollster::block_on(compile_and_eval(
        &model.graph,
        inputs.iter().map(|slice| slice.as_slice()).collect(),
        dim_mappings,
    ))?;

    let dump_folder = args
        .dump_folder
        .unwrap_or_else(|| std::path::PathBuf::from("activations"));

    for (name, tensor) in outputs.iter() {
        let filename = format!("{}.npy", name.replace('/', "."));
        let filepath = dump_folder.join(&filename);
        let data = tensor.raw_data();
        let desc = &tensor.desc;

        npy::save_to_file(filepath.to_str().unwrap(), data, desc)
            .with_context(|| anyhow!("Saving to {filename}"))?;
    }

    Ok(())
}
