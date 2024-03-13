use std::{collections::HashMap, num::NonZeroU64};

use anyhow::{anyhow, bail, Context};
use wgpu::util::DeviceExt;

use crate::{
    analyzer::{self, effective_inputs, BufferHandle, LogicalGraph, LogicalOp, LogicalOpType},
    compiler::{compile_node, is_untracked_op},
    onnx,
    tensor::{DataType, TensorDesc},
    utils::external_data,
};

pub(crate) struct TensorStorage {
    desc: TensorDesc,
    buffer: wgpu::Buffer,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BufferType {
    /// Rewritable buffers
    Input,
    /// Readable buffers
    Output,
    /// Regular buffer
    Intermediary,
}

impl TensorStorage {
    pub(crate) fn size(&self) -> u64 {
        self.buffer.size()
    }

    fn new(
        device: &wgpu::Device,
        desc: TensorDesc,
        label: Option<&str>,
        node_type: BufferType,
    ) -> Self {
        let usage = match node_type {
            BufferType::Output => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            BufferType::Input => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            BufferType::Intermediary => wgpu::BufferUsages::STORAGE,
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            mapped_at_creation: false,
            size: desc.size_of() as _,
            usage,
        });
        Self { desc, buffer }
    }

    fn new_with_init(
        device: &wgpu::Device,
        data: &[u8],
        desc: TensorDesc,
        label: Option<&str>,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self { desc, buffer }
    }
}

/// A GPU operation to be run with inputs and outputs buffers.
pub(crate) struct Op {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch: (u32, u32, u32),
}

impl Op {
    pub(crate) fn new(
        runner: &Runner,
        inputs: Vec<&TensorStorage>,
        outputs: Vec<&TensorStorage>,
        logical_op: &LogicalOp,
        logical_graph: &LogicalGraph,
    ) -> anyhow::Result<Self> {
        let shader = compile_node(logical_op, logical_graph)
            .with_context(|| anyhow!("compiling shader for {}", logical_op.name()))?;
        let enable_f16 = logical_op
            .inputs
            .iter()
            .take(effective_inputs(logical_op))
            .any(|input| matches!(logical_graph.get_desc(*input).dtype, DataType::F16))
            || logical_op
                .outputs
                .iter()
                .any(|output| matches!(logical_graph.get_desc(*output).dtype, DataType::F16));
        if enable_f16 {
            bail!("f16 is currently not supported in naga");
        }
        let shader_source = shader
            .to_wgsl(enable_f16)
            .with_context(|| anyhow!("expanding shader for {}", logical_op.name()))?;

        // std::fs::write(
        //     &format!("kernels/{}.wgsl", logical_op.name()),
        //     shader_source.as_bytes(),
        // )?;
        // println!("{shader_source}");

        let kernel = runner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!(
                    "Shader {}[{}]",
                    logical_op.name(),
                    logical_op.op_type()
                )),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::from(shader_source)),
            });
        let pipeline = runner
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!(
                    "Compute Pipeline {}[{:?}]",
                    logical_op.name(),
                    logical_op.op_type()
                )),
                layout: None,
                module: &kernel,
                entry_point: "main",
            });
        let bind_group = runner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!(
                "Bind Group Compute {}[{:?}]",
                logical_op.name(),
                logical_op.op_type()
            )),
            entries: &inputs
                .iter()
                .chain(&outputs)
                .enumerate()
                .map(|(i, tensor)| wgpu::BindGroupEntry {
                    binding: i as _,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &tensor.buffer,
                        offset: 0,
                        size: NonZeroU64::new(tensor.size()),
                    }),
                })
                .collect::<Vec<wgpu::BindGroupEntry>>(),
            layout: &pipeline.get_bind_group_layout(0),
        });

        Ok(Self {
            pipeline,
            bind_group,
            dispatch: shader.dispatch(),
        })
    }

    pub(crate) fn run<'a>(&'a self, compute_pass: &mut wgpu::ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        let (x, y, z) = self.dispatch;
        compute_pass.dispatch_workgroups(x, y, z);
    }
}

pub(crate) type TensorHandle = usize;
const NO_TENSOR_HANDLE: TensorHandle = TensorHandle::MAX;

pub(crate) struct Runner {
    device: wgpu::Device,
    queue: wgpu::Queue,

    inputs: HashMap<TensorHandle, wgpu::Buffer>,
    outputs: HashMap<TensorHandle, wgpu::Buffer>,
    slots: Vec<TensorStorage>,

    // A map BufferHandle -> TensorHandle,
    // using semaphore NO_TENSOR_HANDLE for no
    // Buffer.
    which_slots: Vec<TensorHandle>,
}

const MAX_ALLOC_LIMIT: u64 = 7_500_000_000;

impl Runner {
    pub(crate) async fn new(
        max_buffer_size: Option<u32>,
        enable_f16: bool,
    ) -> anyhow::Result<Runner> {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow!("failed to request adapter"))?;

        let info = adapter.get_info();
        log::info!("adapter: {}", info.name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: if enable_f16 {
                        wgpu::Features::SHADER_FLOAT16
                    } else {
                        wgpu::Features::empty()
                    },
                    limits: match max_buffer_size {
                        Some(max_storage_buffer_binding_size) => wgpu::Limits {
                            max_storage_buffer_binding_size,
                            ..Default::default()
                        },
                        None => wgpu::Limits::default(),
                    },
                    label: Some("Compute Device"),
                },
                None,
            )
            .await?;

        Ok(Self {
            device,
            queue,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            slots: Vec::new(),
            which_slots: Vec::new(),
        })
    }

    pub(crate) fn total_allocated_size(&self) -> u64 {
        self.slots.iter().map(|tensor| tensor.size()).sum::<u64>()
            + self
                .outputs
                .values()
                .map(|buffer| buffer.size())
                .sum::<u64>()
    }

    fn has_buffer(&self, buf: &BufferHandle) -> bool {
        !matches!(self.which_slots.get(*buf), None | Some(&NO_TENSOR_HANDLE))
    }

    /// Performs tensor allocation by reusing unused allocations.
    /// TODO: put inits, inputs and output buffers inside the allocation line too?
    pub(crate) fn allocate_tensors(
        &mut self,
        log_graph: &analyzer::LogicalGraph,
        force_readable: bool,
        allow_not_exact_size_buffers: bool,
    ) -> anyhow::Result<()> {
        let mut current_size = self.total_allocated_size();

        #[derive(Clone)]
        struct Slot {
            free: bool,
        }

        let pre_allocated_slots = self.slots.len();
        let mut slots_availability: Vec<Slot> = std::iter::repeat(Slot { free: false })
            .take(pre_allocated_slots)
            .collect();
        debug_assert!(slots_availability.len() == self.slots.len());

        log::debug!(
            "starting allocation with size {} on {} slots",
            human_bytes::human_bytes(current_size as f64),
            self.slots.len(),
        );

        let aliases: HashMap<BufferHandle, BufferHandle> = HashMap::from_iter(
            log_graph
                .ops
                .iter()
                .filter(|node| matches!(node.op_type(), LogicalOpType::ReshapeOnly))
                .map(|node| (node.outputs[0], node.inputs[0])),
        );

        // Iterate the nodes in reverse to get liveness ranges for free.
        // See https://www.mattkeeter.com/blog/2022-10-04-ssra/ for more details.
        // If `force_readable` == true, then we never free the slots so that each value has its own
        // buffer.
        'nodes: for node in log_graph.ops.iter().rev() {
            if is_untracked_op(node.op_type())
                || matches!(node.op_type(), LogicalOpType::ReshapeOnly)
            {
                continue 'nodes;
            }

            // We allocate inputs first, so that no ops can run the risk of having the same buffer
            // as inputs and outputs because output buffers are only freed once input buffers have
            // been allocated for this particular node.
            'inputs: for input in node
                .inputs
                .iter()
                .take(analyzer::effective_inputs(node))
                .map(|input| match aliases.get(input) {
                    Some(s) => s,
                    None => input,
                })
            {
                // Input already has a buffer, skip it..
                if self.has_buffer(input) {
                    continue 'inputs;
                }

                // When node appears for the first time,
                // reserve a free slot for it (1.1) if available (1.)
                // or allocate one (1.2).

                // 1. Find free slot
                if let Some((slot_index, slot)) = slots_availability
                    .iter_mut()
                    .enumerate()
                    .skip(pre_allocated_slots) // <- Don't use pre-allocated slots
                    .find(|(slot_index, slot)| {
                        slot.free
                            && match self.slots[*slot_index]
                                .desc
                                .size_of()
                                .cmp(&log_graph.get_desc(*input).size_of())
                            {
                                std::cmp::Ordering::Greater if allow_not_exact_size_buffers => true,
                                std::cmp::Ordering::Equal => true,
                                _ => false,
                            }
                    })
                {
                    log::debug!("reusing slot {slot_index} for {input}");

                    // 1.1) Reserve free slot if available
                    slot.free = false;
                    self.register_slot(*input, slot_index);
                } else {
                    let desc = log_graph.get_desc(*input);

                    current_size += desc.size_of() as u64;
                    if current_size > MAX_ALLOC_LIMIT {
                        bail!(
                            "out-of-memory error when allocating {} (currently at {}) for {}",
                            human_bytes::human_bytes(desc.size_of() as f64),
                            human_bytes::human_bytes(current_size as f64),
                            input
                        );
                    }

                    // 1.2) Allocate slot otherwise
                    let slot_index = self.slots.len();

                    log::debug!(
                        "no slot found, allocating {} for {input} ({slot_index})",
                        human_bytes::human_bytes(desc.size_of() as f64),
                    );

                    self.add_node(
                        input,
                        desc.clone(),
                        if force_readable {
                            BufferType::Output
                        } else {
                            BufferType::Intermediary
                        },
                    )?;
                    slots_availability.push(Slot { free: false });

                    debug_assert!(self.slots.len() == slots_availability.len());
                }
            }

            'outputs: for output in node.outputs.iter().map(|output| match aliases.get(output) {
                Some(s) => s,
                None => output,
            }) {
                if !self.has_buffer(output) {
                    log::warn!("found output which was not used before (%{output})");
                    continue 'outputs;
                }

                debug_assert!(self.slots.len() == slots_availability.len());

                let slot_index = self.which_slots[*output];

                // If the slot was pre-allocated or we want every activation to have its own buffer,
                // then we don't free it.
                if slot_index < pre_allocated_slots || force_readable {
                    continue 'outputs;
                }

                log::debug!(
                    "freeing slot {} for output {output}",
                    self.which_slots[*output]
                );

                // 2) When output appears mark slot as free
                let slot: &mut Slot = &mut slots_availability[slot_index];
                slot.free = true;
            }
        }

        for (output, input) in aliases {
            if self.has_buffer(&output) {
                continue;
            }
            self.register_slot(output, self.which_slots[input]);
        }

        if &std::env::var_os("DUMP_ALLOCS")
            .map(|s| s.to_str().unwrap().to_owned())
            .unwrap_or_else(|| String::from("1"))
            == "1"
        {
            println!("=== ALLOCATIONS");
            for node in &log_graph.ops {
                if is_untracked_op(node.op_type()) {
                    continue;
                }

                for (i, output) in node.outputs.iter().enumerate() {
                    print!(
                        "%{}(#{})",
                        output,
                        match self.which_slots.get(*output) {
                            Some(s) => *s as isize,
                            None => -1,
                        }
                    );
                    if i != node.outputs.len() - 1 {
                        print!(", ");
                    }
                }
                print!("\t= {}[{}](", node.name(), node.op_type());
                let input_len = effective_inputs(node);
                for (i, input) in node.inputs.iter().take(input_len).enumerate() {
                    print!(
                        "%{}(#{})",
                        input,
                        match self.which_slots.get(*input) {
                            Some(s) => *s as isize,
                            None => -1,
                        }
                    );
                    if i != input_len - 1 {
                        print!(", ");
                    }
                }
                println!(")")
            }
            println!("===============");
        }

        Ok(())
    }

    pub(crate) fn get_storage(&self, handle: &BufferHandle) -> anyhow::Result<&TensorStorage> {
        match self.which_slots.get(*handle) {
            Some(i) if *i < self.slots.len() => Ok(&self.slots[*i]),
            _ => Err(anyhow!("failed to find tensor for %{handle}")),
        }
    }

    pub(crate) fn add_node(
        &mut self,
        handle: &BufferHandle,
        desc: TensorDesc,
        node_type: BufferType,
    ) -> anyhow::Result<()> {
        let buf_size = desc.size_of();
        let storage = TensorStorage::new(&self.device, desc, None, node_type);
        if self.has_buffer(handle) {
            anyhow::bail!("node {} was already inserted in runner", handle);
        }

        let tensor_idx = self.slots.len();
        self.slots.push(storage);
        self.register_slot(*handle, tensor_idx);

        match node_type {
            BufferType::Output => {
                let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Read buffer {handle}")),
                    mapped_at_creation: false,
                    size: buf_size as _,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                });
                self.outputs.insert(tensor_idx, read_buffer);
            }
            BufferType::Input => {
                let write_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Read buffer {handle}")),
                    mapped_at_creation: false,
                    size: buf_size as _,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
                });
                self.inputs.insert(tensor_idx, write_buffer);
            }
            BufferType::Intermediary => {}
        }

        Ok(())
    }

    pub(crate) fn add_init(
        &mut self,
        tensor: &onnx::TensorProto,
        handle: &BufferHandle,
        desc: TensorDesc,
    ) -> anyhow::Result<()> {
        // TODO: Move this function out of gpu::Runner.
        if tensor.data_location() == crate::onnx::tensor_proto::DataLocation::EXTERNAL {
            return self.add_node_with_init(handle, desc, &external_data(tensor)?);
        }

        let raw_data = match desc.dtype {
            DataType::F32 if !tensor.float_data.is_empty() => {
                bytemuck::cast_slice(&tensor.float_data)
            }
            DataType::I64 if !tensor.int64_data.is_empty() => {
                bytemuck::cast_slice(&tensor.int64_data)
            }
            DataType::F64 if !tensor.double_data.is_empty() => {
                bytemuck::cast_slice(&tensor.double_data)
            }
            _ => tensor.raw_data(),
        };
        self.add_node_with_init(handle, desc, raw_data)
    }

    fn register_slot(&mut self, node: BufferHandle, handle: TensorHandle) {
        if node + 1 > self.which_slots.len() {
            self.which_slots.resize(node + 1, NO_TENSOR_HANDLE);
        }
        self.which_slots[node] = handle;
    }

    pub(crate) fn add_node_with_init(
        &mut self,
        name: &BufferHandle,
        desc: TensorDesc,
        raw_data: &[u8],
    ) -> anyhow::Result<()> {
        if raw_data.len() != desc.size_of() {
            bail!(
                "invalid raw_data (len {}) for tensor {} of size {}, type {}",
                raw_data.len(),
                name,
                desc.shape,
                desc.dtype,
            );
        }

        if self.has_buffer(name) {
            bail!("tensor {} was already inserted in runner", name);
        }

        let storage = TensorStorage::new_with_init(&self.device, raw_data, desc, None);

        let tensor_idx = self.slots.len();
        self.register_slot(*name, tensor_idx);
        self.slots.push(storage);

        Ok(())
    }

    pub(crate) async fn read_bytes(&self, handle: &BufferHandle) -> anyhow::Result<Vec<u8>> {
        if !self.outputs.contains_key(&self.which_slots[*handle]) {
            bail!(
                "output buffer not found for %{handle}(#{})",
                self.which_slots[*handle]
            );
        }
        let read_buf = &self.outputs[&self.which_slots[*handle]];

        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        match rx.receive().await {
            Some(Ok(())) => {
                let out = slice.get_mapped_range().to_owned();
                read_buf.unmap();
                Ok(out)
            }
            Some(Err(err)) => Err(anyhow!("error: reading buffer: {err}")),
            _ => Err(anyhow!("error: reading buffer")),
        }
    }

    pub(crate) fn submit_ops(&self, ops: &[Op]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Command Encoder"),
            });

        for (input, content_buf) in self.inputs.iter() {
            let dest_buffer = &self.slots[*input];
            encoder.copy_buffer_to_buffer(
                content_buf,
                0,
                &dest_buffer.buffer,
                0,
                dest_buffer.size(),
            );
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            for op in ops {
                op.run(&mut compute_pass);
            }
        }

        for (output, target_buffer) in self.outputs.iter() {
            let source_buffer = &self.slots[*output];
            encoder.copy_buffer_to_buffer(
                &source_buffer.buffer,
                0,
                target_buffer,
                0,
                target_buffer.size(),
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub(crate) async fn write_node(
        &self,
        input_handle: BufferHandle,
        input: &[u8],
    ) -> anyhow::Result<()> {
        let tensor_handle = self.which_slots[input_handle];
        let src_buf = &self.inputs[&tensor_handle];

        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        let slice = src_buf.slice(..);
        slice.map_async(wgpu::MapMode::Write, move |res| {
            tx.send(res).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        match rx.receive().await {
            Some(Ok(())) => {
                {
                    let mut range = slice.get_mapped_range_mut();
                    range.copy_from_slice(input);
                }
                src_buf.unmap();
                Ok(())
            }
            Some(Err(err)) => Err(anyhow!("error: reading buffer: {err}")),
            _ => Err(anyhow!("error: reading buffer")),
        }
    }
}
