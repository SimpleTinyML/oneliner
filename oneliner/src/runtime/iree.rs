mod abi;

use core::ffi::c_void;

#[cfg(feature = "ariel-os")]
use ariel_os::log;
#[cfg(not(feature = "ariel-os"))]
use log;

use super::buffer::checked_len;
use super::{DefaultExecutor, Error, Executor, Result, TensorRange, WorkItem};
use abi::{iree_hal_executable_library_v0_t, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST};
use portable_atomic::{AtomicI32, Ordering};

pub use abi::{
    iree_hal_executable_dispatch_state_v0_t, iree_hal_executable_environment_v0_t,
    iree_hal_executable_import_thunk_v0_t, iree_hal_executable_import_v0_t,
    iree_hal_executable_library_header_t, iree_hal_executable_library_query_fn_t,
    iree_hal_executable_workgroup_state_v0_t, iree_hal_processor_v0_t, DispatchFn,
};

pub const MAX_BINDINGS: usize = 32;

/// Resolves an export ordinal from an IREE static library.
///
/// # Safety
///
/// `query` must return a valid IREE v0 executable library for the duration of
/// the call, including a valid export pointer table.
pub unsafe fn dispatch_fn_from_library(
    query: iree_hal_executable_library_query_fn_t,
    ordinal: usize,
) -> Result<DispatchFn> {
    let environment = empty_environment();
    let library = unsafe { query(IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST, &environment) }
        as *const iree_hal_executable_library_v0_t;
    if library.is_null() {
        return Err(Error::MissingDispatchFunction { ordinal });
    }

    let exports = unsafe { &(*library).exports };
    if ordinal >= exports.count as usize || exports.ptrs.is_null() {
        return Err(Error::MissingDispatchFunction { ordinal });
    }

    let function = unsafe { *exports.ptrs.add(ordinal) };
    log::trace!(
        "Resolved dispatch function for ordinal {} at address {:#x}",
        ordinal,
        function as usize
    );
    Ok(function)
}

/// Dispatches an IREE workload and panics on failure.
///
/// # Safety
///
/// `function` and every tensor range must satisfy the same requirements as
/// [`try_dispatch`].
pub unsafe fn dispatch(
    function: DispatchFn,
    params: &[u32],
    workload: &[u32],
    ranges: &[TensorRange],
) {
    unsafe { try_dispatch(function, params, workload, ranges) }.expect("backend dispatch failed");
}

/// Dispatches an IREE workload with the default executor.
///
/// # Safety
///
/// `function` must be a valid IREE dispatch function. Every tensor pointer must
/// remain valid for its declared range until all scheduled work completes.
pub unsafe fn try_dispatch(
    function: DispatchFn,
    params: &[u32],
    workload: &[u32],
    ranges: &[TensorRange],
) -> Result<()> {
    let mut executor = DefaultExecutor::default();
    unsafe { try_dispatch_with_executor(&mut executor, function, params, workload, ranges) }
}

/// Dispatches an IREE workload through a caller-provided executor.
///
/// # Safety
///
/// `function` must be a valid IREE dispatch function. Every tensor pointer must
/// remain valid for its declared range until the executor has completed all
/// submitted work.
pub unsafe fn try_dispatch_with_executor<E>(
    executor: &mut E,
    function: DispatchFn,
    params: &[u32],
    workload: &[u32],
    ranges: &[TensorRange],
) -> Result<()>
where
    E: Executor,
{
    if ranges.len() > MAX_BINDINGS {
        return Err(Error::TooManyBindings {
            provided: ranges.len(),
            capacity: MAX_BINDINGS,
        });
    }
    let constant_count = u16::try_from(params.len()).map_err(|_| Error::TooManyConstants {
        provided: params.len(),
        capacity: u16::MAX as usize,
    })?;
    let workload_z = u16::try_from(workload.get(2).copied().unwrap_or(1)).map_err(|_| {
        Error::WorkgroupCountTooLarge {
            dimension: 'z',
            value: workload[2],
        }
    })?;

    let mut binding_ptrs = [core::ptr::null_mut(); MAX_BINDINGS];
    let mut binding_lengths = [0usize; MAX_BINDINGS];
    for (index, range) in ranges.iter().enumerate() {
        let len = checked_len(*range)?;
        binding_ptrs[index] = unsafe { range.tensor.ptr.add(range.offset) as *mut c_void };
        binding_lengths[index] = len;
        log::trace!(
            "Binding {}: ptr = {:#x}, length = {}, align16 = {}",
            index,
            binding_ptrs[index] as usize,
            binding_lengths[index],
            (binding_ptrs[index] as usize) % 16
        );
    }

    let environment = empty_environment();
    let workload_x = workload.first().copied().unwrap_or(1);
    let workload_y = workload.get(1).copied().unwrap_or(1);
    let dispatch_state = iree_hal_executable_dispatch_state_v0_t {
        workgroup_size_x: 1,
        workgroup_size_y: 1,
        workgroup_size_z: 1,
        constant_count,
        workgroup_count_x: workload_x,
        workgroup_count_y: workload_y,
        workgroup_count_z: workload_z,
        max_concurrency: 1,
        binding_count: ranges.len() as u8,
        constants: params.as_ptr(),
        binding_ptrs: binding_ptrs.as_ptr(),
        binding_lengths: binding_lengths.as_ptr(),
    };
    let dispatch_status = AtomicI32::new(0);

    for z in 0..workload_z {
        for y in 0..workload_y {
            for x in 0..workload_x {
                log::trace!("Dispatching workgroup (x={}, y={}, z={})", x, y, z);
                let work_item = unsafe {
                    WorkItem::iree(
                        function,
                        core::ptr::addr_of!(environment)
                            as *mut iree_hal_executable_environment_v0_t,
                        core::ptr::addr_of!(dispatch_state)
                            as *mut iree_hal_executable_dispatch_state_v0_t,
                        iree_hal_executable_workgroup_state_v0_t {
                            workgroup_id_x: x,
                            workgroup_id_y: y,
                            workgroup_id_z: z,
                            reserved: 0,
                            processor_id: 0,
                            local_memory: core::ptr::null_mut(),
                            local_memory_size: 0,
                        },
                        core::ptr::addr_of!(dispatch_status),
                    )
                };
                executor.schedule(work_item);
            }
        }
    }
    executor.wait_job_completion();
    let status = dispatch_status.load(Ordering::Acquire);
    if status != 0 {
        return Err(Error::DispatchFailed { status });
    }
    Ok(())
}

fn empty_environment() -> iree_hal_executable_environment_v0_t {
    iree_hal_executable_environment_v0_t {
        constants: core::ptr::null(),
        import_thunk: None,
        import_funcs: core::ptr::null(),
        import_contexts: core::ptr::null(),
        processor: iree_hal_processor_v0_t { data: [0; 8] },
    }
}
