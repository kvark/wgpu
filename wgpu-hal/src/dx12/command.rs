use super::{conv, HResult as _};
use std::{mem, ops::Range, ptr};
use winapi::um::d3d12;

fn make_box(origin: &wgt::Origin3d, size: &crate::CopyExtent) -> d3d12::D3D12_BOX {
    d3d12::D3D12_BOX {
        left: origin.x,
        top: origin.y,
        right: origin.x + size.width,
        bottom: origin.y + size.height,
        front: origin.z,
        back: origin.z + size.depth,
    }
}

impl super::Temp {
    fn prepare_marker(&mut self, marker: &str) -> (&[u16], u32) {
        self.marker.clear();
        self.marker.extend(marker.encode_utf16());
        self.marker.push(0);
        (&self.marker, self.marker.len() as u32 * 2)
    }
}

impl super::CommandEncoder {
    unsafe fn begin_pass(&mut self, kind: super::PassKind, label: crate::Label) {
        let list = self.list.unwrap();
        self.pass.kind = kind;
        if let Some(label) = label {
            let (wide_label, size) = self.temp.prepare_marker(label);
            list.BeginEvent(0, wide_label.as_ptr() as *const _, size);
            self.pass.has_label = true;
        }
        list.set_descriptor_heaps(&[self.shared.heap_views.raw, self.shared.heap_samplers.raw]);
    }

    unsafe fn end_pass(&mut self) {
        let list = self.list.unwrap();
        list.set_descriptor_heaps(&[]);
        if self.pass.has_label {
            list.EndEvent();
        }
        self.pass.clear();
    }

    unsafe fn prepare_draw(&mut self) {
        let list = self.list.unwrap();
        while self.pass.dirty_vertex_buffers != 0 {
            let index = self.pass.dirty_vertex_buffers.trailing_zeros();
            self.pass.dirty_vertex_buffers ^= 1 << index;
            list.IASetVertexBuffers(
                index,
                1,
                self.pass.vertex_buffers.as_ptr().offset(index as isize),
            );
        }
    }

    fn update_root_elements(&self, range: Range<super::RootIndex>) {
        use super::{BufferViewKind as Bvk, PassKind as Pk};

        let list = self.list.unwrap();
        for index in range {
            match self.pass.root_elements[index as usize] {
                super::RootElement::Empty => {}
                super::RootElement::Table(descriptor) => match self.pass.kind {
                    Pk::Render => list.set_graphics_root_descriptor_table(index, descriptor),
                    Pk::Compute => list.set_compute_root_descriptor_table(index, descriptor),
                    Pk::Transfer => (),
                },
                super::RootElement::DynamicOffsetBuffer { kind, address } => {
                    match (self.pass.kind, kind) {
                        (Pk::Render, Bvk::Constant) => {
                            list.set_graphics_root_constant_buffer_view(index, address)
                        }
                        (Pk::Compute, Bvk::Constant) => {
                            list.set_compute_root_constant_buffer_view(index, address)
                        }
                        (Pk::Render, Bvk::ShaderResource) => {
                            list.set_graphics_root_shader_resource_view(index, address)
                        }
                        (Pk::Compute, Bvk::ShaderResource) => {
                            list.set_compute_root_shader_resource_view(index, address)
                        }
                        (Pk::Render, Bvk::UnorderedAccess) => {
                            list.set_graphics_root_unordered_access_view(index, address)
                        }
                        (Pk::Compute, Bvk::UnorderedAccess) => {
                            list.set_compute_root_unordered_access_view(index, address)
                        }
                        (Pk::Transfer, _) => (),
                    }
                }
            }
        }
    }
}

impl crate::CommandEncoder<super::Api> for super::CommandEncoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        let list = match self.free_lists.pop() {
            Some(list) => {
                list.reset(self.allocator, native::PipelineState::null());
                list
            }
            None => self
                .device
                .create_graphics_command_list(
                    native::CmdListType::Direct,
                    self.allocator,
                    native::PipelineState::null(),
                    0,
                )
                .into_device_result("Create command list")?,
        };

        if let Some(label) = label {
            let cwstr = conv::map_label(label);
            list.SetName(cwstr.as_ptr());
        }

        self.list = Some(list);
        self.temp.clear();
        self.pass.clear();
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {
        if let Some(list) = self.list.take() {
            list.close();
            self.free_lists.push(list);
        }
    }
    unsafe fn end_encoding(&mut self) -> Result<super::CommandBuffer, crate::DeviceError> {
        let raw = self.list.take().unwrap();
        raw.close();
        Ok(super::CommandBuffer { raw })
    }
    unsafe fn reset_all<I: Iterator<Item = super::CommandBuffer>>(&mut self, command_buffers: I) {
        for cmd_buf in command_buffers {
            self.free_lists.push(cmd_buf.raw);
        }
    }

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, super::Api>>,
    {
        self.temp.barriers.clear();

        for barrier in barriers {
            let s0 = conv::map_buffer_usage_to_state(barrier.usage.start);
            let s1 = conv::map_buffer_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.Transition_mut() = d3d12::D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: barrier.buffer.resource.as_mut_ptr(),
                    Subresource: d3d12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: s0,
                    StateAfter: s1,
                };
                self.temp.barriers.push(raw);
            } else if barrier.usage.start == crate::BufferUses::STORAGE_WRITE {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.UAV_mut() = d3d12::D3D12_RESOURCE_UAV_BARRIER {
                    pResource: barrier.buffer.resource.as_mut_ptr(),
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            self.list
                .unwrap()
                .ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr());
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Api>>,
    {
        self.temp.barriers.clear();

        for barrier in barriers {
            let s0 = conv::map_texture_usage_to_state(barrier.usage.start);
            let s1 = conv::map_texture_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.Transition_mut() = d3d12::D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: barrier.texture.resource.as_mut_ptr(),
                    Subresource: d3d12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: s0,
                    StateAfter: s1,
                };

                let mip_level_count = match barrier.range.mip_level_count {
                    Some(count) => count.get(),
                    None => barrier.texture.mip_level_count - barrier.range.base_mip_level,
                };
                let array_layer_count = match barrier.range.array_layer_count {
                    Some(count) => count.get(),
                    None => barrier.texture.array_layer_count() - barrier.range.base_array_layer,
                };

                if barrier.range.aspect == wgt::TextureAspect::All
                    && barrier.range.base_mip_level + mip_level_count
                        == barrier.texture.mip_level_count
                    && barrier.range.base_array_layer + array_layer_count
                        == barrier.texture.array_layer_count()
                {
                    // Only one barrier if it affects the whole image.
                    self.temp.barriers.push(raw);
                } else {
                    // Generate barrier for each layer/level combination.
                    for rel_mip_level in 0..mip_level_count {
                        for rel_array_layer in 0..array_layer_count {
                            raw.u.Transition_mut().Subresource = barrier.texture.calc_subresource(
                                barrier.range.base_mip_level + rel_mip_level,
                                barrier.range.base_array_layer + rel_array_layer,
                                0,
                            );
                            self.temp.barriers.push(raw);
                        }
                    }
                }
            } else if barrier.usage.start == crate::TextureUses::STORAGE_WRITE {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.UAV_mut() = d3d12::D3D12_RESOURCE_UAV_BARRIER {
                    pResource: barrier.texture.resource.as_mut_ptr(),
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            self.list
                .unwrap()
                .ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr());
        }
    }

    unsafe fn fill_buffer(&mut self, buffer: &super::Buffer, range: crate::MemoryRange, value: u8) {
        assert_eq!(value, 0, "Only zero is supported!");
        let list = self.list.unwrap();
        let mut offset = range.start;
        while offset < range.end {
            let size = super::ZERO_BUFFER_SIZE.min(range.end - offset);
            list.CopyBufferRegion(
                buffer.resource.as_mut_ptr(),
                offset,
                self.shared.zero_buffer.as_mut_ptr(),
                0,
                size,
            );
            offset += size;
        }
    }

    unsafe fn copy_buffer_to_buffer<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferCopy>,
    {
        let list = self.list.unwrap();
        for r in regions {
            list.CopyBufferRegion(
                dst.resource.as_mut_ptr(),
                r.dst_offset,
                src.resource.as_mut_ptr(),
                r.src_offset,
                r.size.get(),
            );
        }
    }

    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &super::Texture,
        _src_usage: crate::TextureUses,
        dst: &super::Texture,
        regions: T,
    ) where
        T: Iterator<Item = crate::TextureCopy>,
    {
        let list = self.list.unwrap();
        let mut src_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: src.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: mem::zeroed(),
        };
        let mut dst_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: dst.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: mem::zeroed(),
        };

        for r in regions {
            let src_box = make_box(&r.src_base.origin, &r.size);
            *src_location.u.SubresourceIndex_mut() = src.calc_subresource_for_copy(&r.src_base);
            *dst_location.u.SubresourceIndex_mut() = dst.calc_subresource_for_copy(&r.dst_base);

            list.CopyTextureRegion(
                &dst_location,
                r.dst_base.origin.x,
                r.dst_base.origin.y,
                r.dst_base.origin.z,
                &src_location,
                &src_box,
            );
        }
    }

    unsafe fn copy_buffer_to_texture<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Texture,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let list = self.list.unwrap();
        let mut src_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: src.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            u: mem::zeroed(),
        };
        let mut dst_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: dst.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: mem::zeroed(),
        };
        let raw_format = conv::map_texture_format(dst.format);

        for r in regions {
            let src_box = make_box(&wgt::Origin3d::ZERO, &r.size);
            *src_location.u.PlacedFootprint_mut() = d3d12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: r.buffer_layout.offset,
                Footprint: d3d12::D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: raw_format,
                    Width: r.size.width,
                    Height: r
                        .buffer_layout
                        .rows_per_image
                        .map_or(r.size.height, |count| count.get()),
                    Depth: r.size.depth,
                    RowPitch: r.buffer_layout.bytes_per_row.map_or(0, |count| {
                        count.get().max(d3d12::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
                    }),
                },
            };
            *dst_location.u.SubresourceIndex_mut() = dst.calc_subresource_for_copy(&r.texture_base);

            list.CopyTextureRegion(
                &dst_location,
                r.texture_base.origin.x,
                r.texture_base.origin.y,
                r.texture_base.origin.z,
                &src_location,
                &src_box,
            );
        }
    }

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &super::Texture,
        _src_usage: crate::TextureUses,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let list = self.list.unwrap();
        let mut src_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: src.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: mem::zeroed(),
        };
        let mut dst_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: dst.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            u: mem::zeroed(),
        };
        let raw_format = conv::map_texture_format(src.format);

        for r in regions {
            let src_box = make_box(&r.texture_base.origin, &r.size);
            *src_location.u.SubresourceIndex_mut() = src.calc_subresource_for_copy(&r.texture_base);
            *dst_location.u.PlacedFootprint_mut() = d3d12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: r.buffer_layout.offset,
                Footprint: d3d12::D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: raw_format,
                    Width: r.size.width,
                    Height: r
                        .buffer_layout
                        .rows_per_image
                        .map_or(r.size.height, |count| count.get()),
                    Depth: r.size.depth,
                    RowPitch: r.buffer_layout.bytes_per_row.map_or(0, |count| count.get()),
                },
            };

            list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, &src_box);
        }
    }

    unsafe fn begin_query(&mut self, set: &super::QuerySet, index: u32) {
        self.list
            .unwrap()
            .BeginQuery(set.raw.as_mut_ptr(), set.raw_ty, index);
    }
    unsafe fn end_query(&mut self, set: &super::QuerySet, index: u32) {
        self.list
            .unwrap()
            .EndQuery(set.raw.as_mut_ptr(), set.raw_ty, index);
    }
    unsafe fn write_timestamp(&mut self, set: &super::QuerySet, index: u32) {
        self.list.unwrap().EndQuery(
            set.raw.as_mut_ptr(),
            d3d12::D3D12_QUERY_TYPE_TIMESTAMP,
            index,
        );
    }
    unsafe fn reset_queries(&mut self, _set: &super::QuerySet, _range: Range<u32>) {
        // nothing to do here
    }
    unsafe fn copy_query_results(
        &mut self,
        set: &super::QuerySet,
        range: Range<u32>,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        _stride: wgt::BufferSize,
    ) {
        self.list.unwrap().ResolveQueryData(
            set.raw.as_mut_ptr(),
            set.raw_ty,
            range.start,
            range.end - range.start,
            buffer.resource.as_mut_ptr(),
            offset,
        );
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<super::Api>) {
        self.begin_pass(super::PassKind::Render, desc.label);

        let mut color_views = [native::CpuDescriptor { ptr: 0 }; crate::MAX_COLOR_TARGETS];
        for (rtv, cat) in color_views.iter_mut().zip(desc.color_attachments.iter()) {
            *rtv = cat.target.view.handle_rtv.unwrap().raw;
        }
        let ds_view = match desc.depth_stencil_attachment {
            None => ptr::null(),
            Some(ref ds) => {
                if ds.target.usage == crate::TextureUses::DEPTH_STENCIL_WRITE {
                    &ds.target.view.handle_dsv_rw.as_ref().unwrap().raw
                } else {
                    &ds.target.view.handle_dsv_ro.as_ref().unwrap().raw
                }
            }
        };

        let list = self.list.unwrap();
        list.OMSetRenderTargets(
            desc.color_attachments.len() as u32,
            color_views.as_ptr(),
            0,
            ds_view,
        );

        self.pass.resolves.clear();
        for (rtv, cat) in color_views.iter().zip(desc.color_attachments.iter()) {
            if !cat.ops.contains(crate::AttachmentOps::LOAD) {
                let value = [
                    cat.clear_value.r as f32,
                    cat.clear_value.g as f32,
                    cat.clear_value.b as f32,
                    cat.clear_value.a as f32,
                ];
                list.clear_render_target_view(*rtv, value, &[]);
            }
            if let Some(ref target) = cat.resolve_target {
                self.pass.resolves.push(super::PassResolve {
                    src: cat.target.view.target_base,
                    dst: target.view.target_base,
                    format: target.view.raw_format,
                });
            }
        }
        if let Some(ref ds) = desc.depth_stencil_attachment {
            let mut flags = native::ClearFlags::empty();
            if !ds.depth_ops.contains(crate::AttachmentOps::LOAD) {
                flags |= native::ClearFlags::DEPTH;
            }
            if !ds.stencil_ops.contains(crate::AttachmentOps::LOAD) {
                flags |= native::ClearFlags::STENCIL;
            }

            if !ds_view.is_null() {
                list.clear_depth_stencil_view(
                    *ds_view,
                    flags,
                    ds.clear_value.0,
                    ds.clear_value.1 as u8,
                    &[],
                );
            }
        }
    }
    unsafe fn end_render_pass(&mut self) {
        if !self.pass.resolves.is_empty() {
            let list = self.list.unwrap();
            self.temp.barriers.clear();

            // All the targets are expected to be in `COLOR_TARGET` state,
            // but D3D12 has special source/destination states for the resolves.
            for resolve in self.pass.resolves.iter() {
                let mut barrier = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                //Note: this assumes `D3D12_RESOURCE_STATE_RENDER_TARGET`.
                // If it's not the case, we can include the `TextureUses` in `PassResove`.
                *barrier.u.Transition_mut() = d3d12::D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: resolve.src.0.as_mut_ptr(),
                    Subresource: resolve.src.1,
                    StateBefore: d3d12::D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: d3d12::D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                };
                self.temp.barriers.push(barrier);
                *barrier.u.Transition_mut() = d3d12::D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: resolve.dst.0.as_mut_ptr(),
                    Subresource: resolve.dst.1,
                    StateBefore: d3d12::D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: d3d12::D3D12_RESOURCE_STATE_RESOLVE_DEST,
                };
                self.temp.barriers.push(barrier);
            }
            list.ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr());

            for resolve in self.pass.resolves.iter() {
                list.ResolveSubresource(
                    resolve.dst.0.as_mut_ptr(),
                    resolve.dst.1,
                    resolve.src.0.as_mut_ptr(),
                    resolve.src.1,
                    resolve.format,
                );
            }

            // Flip all the barriers to reverse, back into `COLOR_TARGET`.
            for barrier in self.temp.barriers.iter_mut() {
                let transition = barrier.u.Transition_mut();
                mem::swap(&mut transition.StateBefore, &mut transition.StateAfter);
            }
            list.ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr());
        }

        self.end_pass();
    }

    unsafe fn set_bind_group(
        &mut self,
        layout: &super::PipelineLayout,
        index: u32,
        group: &super::BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
        invalidation: crate::BindingInvalidation,
    ) {
        let info = &layout.bind_group_infos[index as usize];
        let mut root_index = info.base_root_index as usize;

        let base_update_index = match invalidation {
            crate::BindingInvalidation::All => {
                // Bind CBV/SRC/UAV descriptor tables
                if info.tables.contains(super::TableTypes::SRV_CBV_UAV) {
                    self.pass.root_elements[root_index] =
                        super::RootElement::Table(group.handle_views.unwrap().gpu);
                    root_index += 1;
                }

                // Bind Sampler descriptor tables.
                if info.tables.contains(super::TableTypes::SAMPLERS) {
                    self.pass.root_elements[root_index] =
                        super::RootElement::Table(group.handle_samplers.unwrap().gpu);
                    root_index += 1;
                }
                info.base_root_index
            }
            crate::BindingInvalidation::DynamicOffsetsOnly => {
                // skip the tables
                root_index += info.tables.bits().count_ones() as usize;
                root_index as super::RootIndex
            }
        };

        // Bind root descriptors
        for ((&kind, &gpu_base), &offset) in info
            .dynamic_buffers
            .iter()
            .zip(group.dynamic_buffers.iter())
            .zip(dynamic_offsets)
        {
            self.pass.root_elements[root_index] = super::RootElement::DynamicOffsetBuffer {
                kind,
                address: gpu_base + offset as native::GpuAddress,
            };
            root_index += 1;
        }

        let update_range = if self.pass.signature != layout.raw {
            // D3D12 requires full reset on signature change
            self.pass.signature = layout.raw;
            0..layout.total_root_elements
        } else {
            base_update_index..root_index as super::RootIndex
        };
        self.update_root_elements(update_range);
    }
    unsafe fn set_push_constants(
        &mut self,
        _layout: &super::PipelineLayout,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u32],
    ) {
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {
        let (wide_label, size) = self.temp.prepare_marker(label);
        self.list
            .unwrap()
            .SetMarker(0, wide_label.as_ptr() as *const _, size);
    }
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        let (wide_label, size) = self.temp.prepare_marker(group_label);
        self.list
            .unwrap()
            .BeginEvent(0, wide_label.as_ptr() as *const _, size);
    }
    unsafe fn end_debug_marker(&mut self) {
        self.list.unwrap().EndEvent()
    }

    unsafe fn set_render_pipeline(&mut self, pipeline: &super::RenderPipeline) {
        let list = self.list.unwrap();

        if self.pass.signature != pipeline.signature {
            // D3D12 requires full reset on signature change
            list.set_graphics_root_signature(pipeline.signature);
            self.pass.signature = pipeline.signature;
            self.update_root_elements(0..pipeline.total_root_elements);
        };

        list.set_pipeline_state(pipeline.raw);
        list.IASetPrimitiveTopology(pipeline.topology);

        for (index, (vb, &stride)) in self
            .pass
            .vertex_buffers
            .iter_mut()
            .zip(pipeline.vertex_strides.iter())
            .enumerate()
        {
            if let Some(stride) = stride {
                if vb.StrideInBytes != stride.get() {
                    vb.StrideInBytes = stride.get();
                    self.pass.dirty_vertex_buffers |= 1 << index;
                }
            }
        }
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, super::Api>,
        format: wgt::IndexFormat,
    ) {
        self.list.unwrap().set_index_buffer(
            binding.resolve_address(),
            binding.resolve_size() as u32,
            conv::map_index_format(format),
        );
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, super::Api>,
    ) {
        let vb = &mut self.pass.vertex_buffers[index as usize];
        vb.BufferLocation = binding.resolve_address();
        vb.SizeInBytes = binding.resolve_size() as u32;
        self.pass.dirty_vertex_buffers |= 1 << index;
    }

    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {
        let raw_vp = d3d12::D3D12_VIEWPORT {
            TopLeftX: rect.x,
            TopLeftY: rect.y,
            Width: rect.w,
            Height: rect.h,
            MinDepth: depth_range.start,
            MaxDepth: depth_range.end,
        };
        self.list.unwrap().RSSetViewports(1, &raw_vp);
    }
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        let raw_rect = d3d12::D3D12_RECT {
            left: rect.x as i32,
            top: rect.y as i32,
            right: (rect.x + rect.w) as i32,
            bottom: (rect.y + rect.h) as i32,
        };
        self.list.unwrap().RSSetScissorRects(1, &raw_rect);
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        self.list.unwrap().set_stencil_reference(value);
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        self.list.unwrap().set_blend_factor(*color);
    }

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        self.prepare_draw();
        self.list
            .unwrap()
            .draw(vertex_count, instance_count, start_vertex, start_instance);
    }
    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        self.prepare_draw();
        self.list.unwrap().draw_indexed(
            index_count,
            instance_count,
            start_index,
            base_vertex,
            start_instance,
        );
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        self.prepare_draw();
        self.list.unwrap().ExecuteIndirect(
            self.shared.cmd_signatures.draw.as_mut_ptr(),
            draw_count,
            buffer.resource.as_mut_ptr(),
            offset,
            ptr::null_mut(),
            0,
        );
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        self.prepare_draw();
        self.list.unwrap().ExecuteIndirect(
            self.shared.cmd_signatures.draw_indexed.as_mut_ptr(),
            draw_count,
            buffer.resource.as_mut_ptr(),
            offset,
            ptr::null_mut(),
            0,
        );
    }
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        self.prepare_draw();
        self.list.unwrap().ExecuteIndirect(
            self.shared.cmd_signatures.draw.as_mut_ptr(),
            max_count,
            buffer.resource.as_mut_ptr(),
            offset,
            count_buffer.resource.as_mut_ptr(),
            count_offset,
        );
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        self.prepare_draw();
        self.list.unwrap().ExecuteIndirect(
            self.shared.cmd_signatures.draw_indexed.as_mut_ptr(),
            max_count,
            buffer.resource.as_mut_ptr(),
            offset,
            count_buffer.resource.as_mut_ptr(),
            count_offset,
        );
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor) {
        self.begin_pass(super::PassKind::Compute, desc.label);
    }
    unsafe fn end_compute_pass(&mut self) {
        self.end_pass();
    }

    unsafe fn set_compute_pipeline(&mut self, pipeline: &super::ComputePipeline) {
        let list = self.list.unwrap();

        if self.pass.signature != pipeline.signature {
            // D3D12 requires full reset on signature change
            list.set_compute_root_signature(pipeline.signature);
            self.pass.signature = pipeline.signature;
            self.update_root_elements(0..pipeline.total_root_elements);
        };

        list.set_pipeline_state(pipeline.raw);
    }

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        self.list.unwrap().dispatch(count);
    }
    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {
        self.list.unwrap().ExecuteIndirect(
            self.shared.cmd_signatures.dispatch.as_mut_ptr(),
            1,
            buffer.resource.as_mut_ptr(),
            offset,
            ptr::null_mut(),
            0,
        );
    }
}