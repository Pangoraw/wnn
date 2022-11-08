use protobuf::Message;
use wasm_bindgen::prelude::*;
use wasm_bindgen_console_logger::DEFAULT_LOGGER;
use wnn::{
    onnx,
    tensor::{CPUTensor, CPUTensorData},
};

#[wasm_bindgen(js_name = initModule)]
pub fn init_module() {
    console_error_panic_hook::set_once();
    log::set_logger(&DEFAULT_LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Trace);
}

#[wasm_bindgen]
pub struct ONNXModel {
    model: onnx::ModelProto,
}

pub fn parse_model_(bytes: &[u8]) -> anyhow::Result<ONNXModel> {
    let model = onnx::ModelProto::parse_from_bytes(bytes)?;
    let onnx_model = ONNXModel { model };
    Ok(onnx_model)
}

#[wasm_bindgen(js_name = parseModel)]
pub fn parse_model(raw_bytes: *const u8, size: usize) -> Result<ONNXModel, JsValue> {
    log::info!("parsing model at {:?}, {size}", raw_bytes);
    let bytes = unsafe { std::slice::from_raw_parts(raw_bytes, size) };
    parse_model_(bytes).map_err(|err| JsValue::from(err.to_string()))
}

pub async fn eval_graph_(model: &ONNXModel, input: &[f32]) -> anyhow::Result<Vec<f32>> {
    let outputs = wnn::eval_graph(&model.model.graph, wnn::InitMode::SliceF32(input), None).await?;

    let Some((_, first)) = outputs.iter().next() else {
        anyhow::bail!("model has not output");
    };

    let CPUTensor {
        desc: _,
        data: CPUTensorData::F32(v),
    } = first else {
        anyhow::bail!("invalid output format");
    };

    Ok(v.to_vec()) // NOTE: how to prevent copy ?
}

#[wasm_bindgen(js_name = evalGraph)]
pub async fn eval_graph(
    model: &ONNXModel,
    in_ptr: *const f32,
    in_size: usize,
    out_ptr: *mut f32,
    out_size: usize,
) -> Result<(), JsValue> {
    let in_slice = unsafe { std::slice::from_raw_parts(in_ptr, in_size) };
    let model_out = eval_graph_(model, in_slice)
        .await
        .map_err(|err| JsValue::from(err.to_string()))?;
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_size) };
    out_slice.clone_from_slice(&model_out);

    Ok(())
}
