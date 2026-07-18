#![allow(non_camel_case_types)]

use core::ffi::c_void;

use super::buffer::clipped_len;
use super::{DefaultExecutor, Error, Executor, Result, TensorRange};
use log;
pub const MAX_BINDINGS: usize = 32;

pub type iree_hal_executable_import_v0_t =
    unsafe extern "C" fn(params: *mut c_void, context: *mut c_void, reserved: *mut c_void) -> i32;
pub type iree_hal_executable_import_thunk_v0_t = unsafe extern "C" fn(
    fn_ptr: iree_hal_executable_import_v0_t,
    params: *mut c_void,
    context: *mut c_void,
    reserved: *mut c_void,
) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct iree_hal_processor_v0_t {
    pub data: [u64; 8],
}

#[repr(C)]
pub struct iree_hal_executable_environment_v0_t {
    pub constants: *const u32,
    pub import_thunk: Option<iree_hal_executable_import_thunk_v0_t>,
    pub import_funcs: *const iree_hal_executable_import_v0_t,
    pub import_contexts: *const *const c_void,
    pub processor: iree_hal_processor_v0_t,
}

#[repr(C)]
pub struct iree_hal_executable_dispatch_state_v0_t {
    pub workgroup_size_x: u32,
    pub workgroup_size_y: u32,
    pub workgroup_size_z: u16,
    pub constant_count: u16,
    pub workgroup_count_x: u32,
    pub workgroup_count_y: u32,
    pub workgroup_count_z: u16,
    pub max_concurrency: u8,
    pub binding_count: u8,
    pub constants: *const u32,
    pub binding_ptrs: *const *mut c_void,
    pub binding_lengths: *const usize,
}

#[repr(C)]
pub struct iree_hal_executable_workgroup_state_v0_t {
    pub workgroup_id_x: u32,
    pub workgroup_id_y: u32,
    pub workgroup_id_z: u16,
    pub reserved: u16,
    pub processor_id: u32,
    pub local_memory: *mut c_void,
    pub local_memory_size: u32,
}

pub type DispatchFn = unsafe extern "C" fn(
    environment: *const iree_hal_executable_environment_v0_t,
    dispatch_state: *const iree_hal_executable_dispatch_state_v0_t,
    workgroup_state: *const iree_hal_executable_workgroup_state_v0_t,
) -> i32;

pub const IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST: u32 = 6;

#[repr(C)]
pub struct iree_hal_executable_library_header_t {
    pub version: u32,
    pub name: *const u8,
    pub features: u32,
    pub sanitizer: i32,
}

pub type iree_hal_executable_library_query_fn_t =
    unsafe extern "C" fn(
        max_version: u32,
        environment: *const iree_hal_executable_environment_v0_t,
    ) -> *const *const iree_hal_executable_library_header_t;

#[repr(C)]
struct iree_hal_executable_import_table_v0_t {
    count: u32,
    symbols: *const *const u8,
}

#[repr(C)]
struct iree_hal_executable_dispatch_attrs_v0_t {
    flags: u64,
    local_memory_pages: u16,
    constant_count: u8,
    binding_count: u8,
    workgroup_size_x: u32,
    workgroup_size_y: u32,
    workgroup_size_z: u16,
    parameter_count: u16,
    reserved_1: [u64; 5],
}

#[repr(C)]
struct iree_hal_executable_dispatch_parameter_v0_t {
    parameter_type: u8,
    size: u8,
    flags: u16,
    name: u16,
    offset: u16,
}

#[repr(C)]
struct iree_hal_executable_dispatch_occupancy_v0_t {
    reserved: i32,
}

#[repr(C)]
struct iree_hal_executable_source_location_v0_t {
    line: u32,
    path_length: u32,
    path: *const u8,
}

#[repr(C)]
struct iree_hal_executable_stage_location_table_v0_t {
    count: u32,
    names: *const *const u8,
    locations: *const iree_hal_executable_source_location_v0_t,
}

#[repr(C)]
struct iree_hal_executable_export_table_v0_t {
    count: u32,
    ptrs: *const DispatchFn,
    attrs: *const iree_hal_executable_dispatch_attrs_v0_t,
    params: *const *const iree_hal_executable_dispatch_parameter_v0_t,
    occupancy: *const iree_hal_executable_dispatch_occupancy_v0_t,
    names: *const *const u8,
    tags: *const *const u8,
    parameter_names: *const *const u8,
    source_locations: *const iree_hal_executable_source_location_v0_t,
    stage_locations: *const iree_hal_executable_stage_location_table_v0_t,
}

#[repr(C)]
struct iree_hal_executable_constant_table_v0_t {
    count: u32,
}

#[repr(C)]
struct iree_hal_executable_source_file_v0_t {
    path_length: u32,
    path: *const u8,
    content_length: u32,
    content: *const u8,
}

#[repr(C)]
struct iree_hal_executable_source_file_table_v0_t {
    count: u32,
    files: *const iree_hal_executable_source_file_v0_t,
}

#[repr(C)]
struct iree_hal_executable_library_v0_t {
    header: *const iree_hal_executable_library_header_t,
    imports: iree_hal_executable_import_table_v0_t,
    exports: iree_hal_executable_export_table_v0_t,
    constants: iree_hal_executable_constant_table_v0_t,
    sources: iree_hal_executable_source_file_table_v0_t,
}

/// Resolves an IREE static-library export to a dispatch function.
///
/// Input: generated `*_library_query` symbol and export ordinal.
/// Output: dispatch function pointer, if the library exposes that ordinal.
pub fn dispatch_fn_from_library(
    query: iree_hal_executable_library_query_fn_t,
    ordinal: usize,
) -> Option<DispatchFn> {
    let environment = iree_hal_executable_environment_v0_t {
        constants: core::ptr::null(),
        import_thunk: None,
        import_funcs: core::ptr::null(),
        import_contexts: core::ptr::null(),
        processor: iree_hal_processor_v0_t { data: [0; 8] },
    };

    let library = unsafe { query(IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST, &environment) }
        as *const iree_hal_executable_library_v0_t;
    if library.is_null() {
        return None;
    }

    let exports = unsafe { &(*library).exports };
    if ordinal >= exports.count as usize || exports.ptrs.is_null() {
        return None;
    }

    // for index in 0..exports.count as usize {
    //     let func_ptr = unsafe { *exports.ptrs.add(index) };
    //     let name_ptr = unsafe { *exports.names.add(index) };
    //     let name = if !name_ptr.is_null() {
    //         unsafe { core::ffi::CStr::from_ptr(name_ptr as *const i8) }
    //             .to_str()
    //             .unwrap_or("<invalid utf-8>")
    //     } else {
    //         "<null>"
    //     };
    //     log::info!("Export {}: name = {}, ptr = {}", index, name, func_ptr as usize);
    // }

    unsafe {
        log::trace!(
            "Resolved dispatch function for ordinal {} at address {:#x}",
            ordinal,
            *exports.ptrs.add(ordinal) as usize
        )
    };
    Some(unsafe { *exports.ptrs.add(ordinal) })
}

/// Dispatches an IREE workgroup and panics if it fails.
///
/// Input: optional dispatch function, constants, workload, and tensor ranges.
/// Output: no value; backend side effects are written into bound tensors.
pub fn dispatch(
    function: Option<DispatchFn>,
    params: &[u32],
    workload: &[u32],
    ranges: &[TensorRange],
) {
    try_dispatch(function, params, workload, ranges).expect("backend dispatch failed");
}

/// Dispatches an IREE workgroup and returns runtime errors.
///
/// Input: optional dispatch function, constants, workload, and tensor ranges.
/// Output: `Ok(())` after dispatch or an `Error` if bindings/status fail.
pub fn try_dispatch(
    function: Option<DispatchFn>,
    params: &[u32],
    workload: &[u32],
    ranges: &[TensorRange],
) -> Result<()> {
    let mut executor = DefaultExecutor::default();
    try_dispatch_with_executor(&mut executor, function, params, workload, ranges)
}

/// Dispatches an IREE workgroup through the provided executor.
///
/// Input: executor, optional dispatch function, constants, workload, and tensor ranges.
/// Output: `Ok(())` after dispatch or an `Error` if bindings/status fail.
pub fn try_dispatch_with_executor<E>(
    executor: &mut E,
    function: Option<DispatchFn>,
    params: &[u32],
    workload: &[u32],
    ranges: &[TensorRange],
) -> Result<()>
where
    E: Executor,
{
    let Some(function) = function else {
        return Ok(());
    };
    if ranges.len() > MAX_BINDINGS {
        return Err(Error::TooManyBindings {
            provided: ranges.len(),
            capacity: MAX_BINDINGS,
        });
    }

    let mut binding_ptrs = [core::ptr::null_mut(); MAX_BINDINGS];
    let mut binding_lengths = [0usize; MAX_BINDINGS];
    for (index, range) in ranges.iter().enumerate() {
        let len = clipped_len(*range);
        unsafe {
            binding_ptrs[index] = range.tensor.ptr.add(range.offset) as *mut c_void;
        }
        binding_lengths[index] = len;
        log::trace!(
            "Binding {}: ptr = {:#x}, length = {}, align16 = {}",
            index,
            binding_ptrs[index] as usize,
            binding_lengths[index],
            (binding_ptrs[index] as usize) % 16
        );
    }

    let environment = iree_hal_executable_environment_v0_t {
        constants: core::ptr::null(),
        import_thunk: None,
        import_funcs: core::ptr::null(),
        import_contexts: core::ptr::null(),
        processor: iree_hal_processor_v0_t { data: [0; 8] },
    };
    let workload_x = workload.first().copied().unwrap_or(1);
    let workload_y = workload.get(1).copied().unwrap_or(1);
    let workload_z = workload.get(2).copied().unwrap_or(1) as u16;

    let dispatch_state = iree_hal_executable_dispatch_state_v0_t {
        workgroup_size_x: 1,
        workgroup_size_y: 1,
        workgroup_size_z: 1,
        constant_count: params.len() as u16,
        workgroup_count_x: workload_x,
        workgroup_count_y: workload_y,
        workgroup_count_z: workload_z,
        max_concurrency: 1,
        binding_count: ranges.len() as u8,
        constants: params.as_ptr(),
        binding_ptrs: binding_ptrs.as_ptr(),
        binding_lengths: binding_lengths.as_ptr(),
    };
    let environment_ptr = core::ptr::addr_of!(environment) as usize;
    let dispatch_state_ptr = core::ptr::addr_of!(dispatch_state) as usize;

    for z in 0..workload_z {
        for y in 0..workload_y {
            for x in 0..workload_x {
                log::trace!("Dispatching workgroup (x={}, y={}, z={})", x, y, z);
                executor.schedule(move || {
                    let workgroup_state = iree_hal_executable_workgroup_state_v0_t {
                        workgroup_id_x: x,
                        workgroup_id_y: y,
                        workgroup_id_z: z,
                        reserved: 0,
                        processor_id: 0,
                        local_memory: core::ptr::null_mut(),
                        local_memory_size: 0,
                    };
                    let environment =
                        environment_ptr as *const iree_hal_executable_environment_v0_t;
                    let dispatch_state =
                        dispatch_state_ptr as *const iree_hal_executable_dispatch_state_v0_t;
                    let status = unsafe { function(environment, dispatch_state, &workgroup_state) };
                    if status != 0 {
                        return Err(Error::DispatchFailed { status });
                    }
                    Ok(())
                })?;
            }
        }
    }
    Ok(())
}
