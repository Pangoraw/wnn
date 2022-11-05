use anyhow::Context;
use protobuf::Message;
use structopt::StructOpt;

use wnn::{eval_graph, npy, onnx, InitMode};

#[derive(StructOpt, Debug)]
struct Args {
    #[structopt(default_value = "./sd-v1-5-onnx/vae_decoder_sim.onnx")]
    input_model: std::path::PathBuf,
    dump_folder: Option<std::path::PathBuf>,

    #[structopt(long, short, default_value = "ones")]
    init: String,
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

    let outputs = eval_graph(&model.graph, init_mode, args.dump_folder)?;

    for (name, tensor) in outputs.iter() {
        let filename = format!("activations/{name}.npy");
        let data = tensor.raw_data();
        let desc = &tensor.desc;

        npy::save_to_file(&filename, data, desc)?;
    }

    Ok(())
}
