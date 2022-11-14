use std::{collections::HashMap, num::NonZeroU64};

use anyhow::{anyhow, bail, Context};
use wgpu::util::DeviceExt;

use crate::{
    compiler::{compile_node, effective_inputs, is_reshape_op, is_untracked_op},
    onnx,
    tensor::{DataType, TensorDesc},
    utils::external_data,
};

pub(crate) struct TensorStorage {
    desc: TensorDesc,
    buffer: wgpu::Buffer,
}

impl TensorStorage {
    pub(crate) fn size(&self) -> u64 {
        self.buffer.size()
    }

    fn new(device: &wgpu::Device, desc: TensorDesc, label: Option<&str>, is_output: bool) -> Self {
        let usage = if is_output {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE
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
        device: &wgpu::Device,
        inputs: Vec<&TensorStorage>,
        outputs: Vec<&TensorStorage>,
        node: &onnx::NodeProto,
        descs: &HashMap<&str, TensorDesc>,
    ) -> anyhow::Result<Self> {
        let shader = compile_node(node, descs)
            .with_context(|| anyhow!("compiling shader for {}", node.name()))?;
        let enable_f16 = node
            .input
            .iter()
            .take(effective_inputs(node))
            .any(|input| matches!(descs[input.as_str()].dtype, DataType::F16))
            || node
                .output
                .iter()
                .any(|output| matches!(descs[output.as_str()].dtype, DataType::F16));
        if enable_f16 {
            bail!("f16 is currently not supported in naga");
        }
        let shader_source = shader
            .to_wgsl(enable_f16)
            .with_context(|| anyhow!("expanding shader for {}", node.name()))?;

        // println!("{shader_source}");

        let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Shader {}[{}]", node.name(), node.op_type())),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::from(shader_source)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!(
                "Compute Pipeline {}[{}]",
                node.name(),
                node.op_type()
            )),
            layout: None,
            module: &kernel,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!(
                "Bind Group Compute {}[{}]",
                node.name(),
                node.op_type()
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

pub(crate) struct Runner<'a> {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    slots: Vec<TensorStorage>,
    which_slots: HashMap<&'a str, usize>,
}

const MAX_ALLOC_LIMIT: u64 = 7_500_000_000;

impl<'a> Runner<'a> {
    pub(crate) async fn new(
        max_buffer_size: Option<u32>,
        enable_f16: bool,
    ) -> anyhow::Result<Runner<'a>> {
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
            slots: Vec::new(),
            which_slots: HashMap::new(),
        })
    }

    pub(crate) fn total_allocated_size(&self) -> u64 {
        self.slots.iter().map(|tensor| tensor.size()).sum::<u64>()
    }

    /// Performs tensor allocation by grouping allocations of the same size together.
    /// TODO: put inits, inputs and output buffers inside the allocation line too?
    pub(crate) fn allocate_tensors(
        &mut self,
        nodes: &'a [onnx::NodeProto],
        descs: &HashMap<&str, TensorDesc>,
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

        let aliases: HashMap<&str, &str> = HashMap::from_iter(
            nodes
                .iter()
                .filter(|node| is_reshape_op(node.op_type()))
                .map(|node| (node.output[0].as_str(), node.input[0].as_str())),
        );

        // Iterate the nodes in reverse to get liveness ranges for free.
        // See https://www.mattkeeter.com/blog/2022-10-04-ssra/ for more details.
        // If `force_readable` == true, then we never free the slots so that each value has its own
        // buffer.
        'nodes: for node in nodes.iter().rev() {
            if is_untracked_op(node.op_type()) || is_reshape_op(node.op_type()) {
                continue 'nodes;
            }

            // We allocate inputs first, so that no ops can run the risk of having the same buffer
            // as inputs and outputs because output buffers are only freed once input buffers have
            // been allocated for this particular node.
            'inputs: for input in node.input.iter().take(effective_inputs(node)).map(|input| {
                match aliases.get(input.as_str()) {
                    Some(s) => s,
                    None => input.as_str(),
                }
            }) {
                // Input already has a buffer, skip it..
                if self.which_slots.contains_key(input) {
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
                                .cmp(&descs[input].size_of())
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
                    self.which_slots.insert(input, slot_index);
                } else {
                    let desc = descs[input].clone();

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

                    self.slots
                        .push(TensorStorage::new(&self.device, desc, None, force_readable));
                    slots_availability.push(Slot { free: false });

                    debug_assert!(self.slots.len() == slots_availability.len());

                    self.which_slots.insert(input, slot_index);
                }
            }

            'outputs: for output in
                node.output
                    .iter()
                    .map(|output| match aliases.get(output.as_str()) {
                        Some(s) => s,
                        None => output.as_str(),
                    })
            {
                if !self.which_slots.contains_key(output) {
                    log::warn!("found output which was not used before {output}");
                    continue 'outputs;
                }

                debug_assert!(self.slots.len() == slots_availability.len());

                let slot_index = self.which_slots[output];

                // If the slot was pre-allocated or we want every activation to have its own buffer,
                // then we don't free it.
                if slot_index < pre_allocated_slots || force_readable {
                    continue 'outputs;
                }

                log::debug!(
                    "freeing slot {} for output {output}",
                    self.which_slots[output]
                );

                // 2) When output appears mark slot as free
                let slot: &mut Slot = &mut slots_availability[slot_index];
                slot.free = true;
            }
        }

        for (output, input) in aliases {
            if self.which_slots.contains_key(output) {
                continue;
            }
            self.which_slots.insert(output, self.which_slots[input]);
        }

        if &std::env::var_os("DUMP_ALLOCS")
            .map(|s| s.to_str().unwrap().to_owned())
            .unwrap_or_else(|| String::from("1"))
            == "1"
        {
            println!("=== ALLOCATIONS");
            for node in nodes {
                if is_untracked_op(node.op_type()) {
                    continue;
                }

                for (i, output) in node.output.iter().enumerate() {
                    print!(
                        "{}(#{})",
                        output,
                        match self.which_slots.get(output.as_str()) {
                            Some(s) => *s as isize,
                            None => -1,
                        }
                    );
                    if i != node.output.len() - 1 {
                        print!(", ");
                    }
                }
                print!("\t= {}[{}](", node.name(), node.op_type());
                let input_len = if node.op_type() == "Resize" {
                    1
                } else {
                    node.input.len()
                };
                for (i, input) in node.input.iter().take(input_len).enumerate() {
                    print!(
                        "{}(#{})",
                        input,
                        match self.which_slots.get(input.as_str()) {
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

    pub(crate) fn get_storage(&self, name: &str) -> anyhow::Result<&TensorStorage> {
        match self.which_slots.get(name) {
            Some(i) if *i < self.slots.len() => Ok(&self.slots[*i]),
            _ => Err(anyhow!("failed to find tensor for {name}")),
        }
    }

    pub(crate) fn add_node(
        &mut self,
        name: &'a str,
        desc: TensorDesc,
        is_output: bool,
    ) -> anyhow::Result<()> {
        let storage = TensorStorage::new(&self.device, desc, Some(name), is_output);
        if self.which_slots.contains_key(name) {
            anyhow::bail!("node {} was already inserted in runner", name);
        }

        let tensor_idx = self.slots.len();
        self.slots.push(storage);
        self.which_slots.insert(name, tensor_idx);
        Ok(())
    }

    pub(crate) fn add_init(
        &mut self,
        tensor: &'a onnx::TensorProto,
        desc: TensorDesc,
    ) -> anyhow::Result<()> {
        // TODO: Move this function out of gpu::Runner.
        if tensor.data_location() == crate::onnx::tensor_proto::DataLocation::EXTERNAL {
            return self.add_node_with_init(tensor.name(), desc, &external_data(tensor)?);
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
        return self.add_node_with_init(tensor.name(), desc, raw_data);
    }

    pub(crate) fn add_node_with_init(
        &mut self,
        name: &'a str,
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

        if self.which_slots.contains_key(name) {
            log::warn!("tensor {} was already inserted in runner", name);
            return Ok(());
            // bail!("tensor {} was already inserted in runner", name);
        }
        let storage = TensorStorage::new_with_init(&self.device, raw_data, desc, Some(name));
        let tensor_idx = self.which_slots.len();
        self.which_slots.insert(name, tensor_idx);
        self.slots.push(storage);
        Ok(())
    }

    pub(crate) async fn read_bytes_from_name(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let tensor = self.get_storage(name)?;
        self.read_bytes(tensor).await
    }

    pub(crate) async fn read_bytes(&self, tensor: &TensorStorage) -> anyhow::Result<Vec<u8>> {
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: tensor.size(),
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &read_buf, 0, tensor.size());
        self.queue.submit(std::iter::once(encoder.finish()));

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
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            for op in ops {
                op.run(&mut compute_pass);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
