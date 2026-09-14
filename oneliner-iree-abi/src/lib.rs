//! Rust representations of IREE's local executable-library ABI.
//!
//! These `repr(C)` representations interact with IREE executabe libraries compiled by IREE.
//! The upstream IREE executabe library API version is **0.6**. the `v0`
//! suffix in type names does not imply a version constant of zero.
//!
//! The source specification is IREE's
//! [executable_library.h](https://github.com/iree-org/iree/blob/9acea6ac572f30a28b19c70118d94b4457f87420/runtime/src/iree/hal/local/executable_library.h).
//! Field order, widths and pointer indirection
//! must remain ABI-compatible; Rust target layout determines pointer size.
//!
//! # Pointer contracts
//!
//! Before invoking a function pointer,
//! callers must provide compatible structures and live, aligned nested arrays
//! of the stated lengths. Buffer writes must satisfy the kernel's access and
//! aliasing requirements. Null is allowed only where the corresponding ABI
//! field permits it. All borrowed dispatch data must survive before finishing workitem execution.

#![no_std]
#![allow(non_camel_case_types)]

use core::ffi::c_void;

/// Imported function taking opaque parameters and context; zero means success.
pub type iree_hal_executable_import_v0_t =
    unsafe extern "C" fn(params: *mut c_void, context: *mut c_void, reserved: *mut c_void) -> i32;
/// Import-call adapter receiving the target function and its opaque arguments.
pub type iree_hal_executable_import_thunk_v0_t = unsafe extern "C" fn(
    fn_ptr: iree_hal_executable_import_v0_t,
    params: *mut c_void,
    context: *mut c_void,
    reserved: *mut c_void,
) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
/// Opaque CPU information.
pub struct iree_hal_processor_v0_t {
    /// Eight architecture-specific 64-bit words; zero when unavailable.
    pub data: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Execution environment shared with kernels.
pub struct iree_hal_executable_environment_v0_t {
    /// Executable specialization constants, or null when absent.
    pub constants: *const u32,
    /// Optional dispatcher used to call imported functions.
    pub import_thunk: Option<iree_hal_executable_import_thunk_v0_t>,
    /// Import function table in import-ordinal order.
    pub import_funcs: *const iree_hal_executable_import_v0_t,
    /// Context pointer table corresponding to imports.
    pub import_contexts: *const *const c_void,
    /// CPU information available to the executable.
    pub processor: iree_hal_processor_v0_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Read-only dispatch state shared by all workgroups.
pub struct iree_hal_executable_dispatch_state_v0_t {
    /// Chosen workgroup size on the x axis.
    pub workgroup_size_x: u32,
    /// Chosen workgroup size on the y axis.
    pub workgroup_size_y: u32,
    /// Chosen workgroup size on the z axis.
    pub workgroup_size_z: u16,
    /// Number of 32-bit push-constant words.
    pub constant_count: u16,
    /// Number of workgroups on the x axis.
    pub workgroup_count_x: u32,
    /// Number of workgroups on the y axis.
    pub workgroup_count_y: u32,
    /// Number of workgroups on the z axis.
    pub workgroup_count_z: u16,
    /// Estimated maximum simultaneous workgroups.
    pub max_concurrency: u8,
    /// Number of entries in both binding arrays.
    pub binding_count: u8,
    /// Pointer to `constant_count` 32-bit words.
    pub constants: *const u32,
    /// Binding base addresses, indexed by binding ordinal.
    pub binding_ptrs: *const *mut c_void,
    /// Byte lengths corresponding one-to-one with binding pointers.
    pub binding_lengths: *const usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// State for one workgroup invocation.
pub struct iree_hal_executable_workgroup_state_v0_t {
    /// Zero-based x coordinate below the dispatch count.
    pub workgroup_id_x: u32,
    /// Zero-based y coordinate below the dispatch count.
    pub workgroup_id_y: u32,
    /// Zero-based z coordinate below the dispatch count.
    pub workgroup_id_z: u16,
    /// Reserved field; initialize to zero.
    pub reserved: u16,
    /// Logical execution processor identifier.
    pub processor_id: u32,
    /// Exclusive workgroup scratch; null if no scratch is required.
    pub local_memory: *mut c_void,
    /// Available workgroup scratch bytes.
    pub local_memory_size: u32,
}

/// Kernel entry point; returns zero on success and a nonzero backend status on failure.
pub type DispatchFn = unsafe extern "C" fn(
    environment: *const iree_hal_executable_environment_v0_t,
    dispatch_state: *const iree_hal_executable_dispatch_state_v0_t,
    workgroup_state: *const iree_hal_executable_workgroup_state_v0_t,
) -> i32;

/// Maximum executable-library API version understood by these bindings (0.6).
pub const IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST: u32 = 6;

#[repr(C)]
/// Common executable-library header.
pub struct iree_hal_executable_library_header_t {
    /// Executable-library API version required by this object.
    pub version: u32,
    /// NUL-terminated diagnostic name.
    pub name: *const u8,
    /// Required/supported executable feature bits.
    pub features: u32,
    /// Sanitizer kind expected by the compiled object.
    pub sanitizer: i32,
}

/// Queries a library compatible with the supplied maximum API version.
///
/// The returned pointer addresses a versioned library record whose first field
/// is a header pointer. It may be null when no compatible version is available.
pub type iree_hal_executable_library_query_fn_t =
    unsafe extern "C" fn(
        max_version: u32,
        environment: *const iree_hal_executable_environment_v0_t,
    ) -> *const *const iree_hal_executable_library_header_t;

#[repr(C)]
/// Imports declared by an executable.
pub struct iree_hal_executable_import_table_v0_t {
    /// Number of import entries.
    pub count: u32,
    /// Array of NUL-terminated import names; names encode optionality.
    pub symbols: *const *const u8,
}

#[repr(C)]
/// Static requirements for one exported dispatch.
pub struct iree_hal_executable_dispatch_attrs_v0_t {
    /// Dispatch behavior flags.
    pub flags: u64,
    /// Required workgroup scratch in 4096-byte pages.
    pub local_memory_pages: u16,
    /// Number of required 32-bit push constants.
    pub constant_count: u8,
    /// Number of required buffer bindings.
    pub binding_count: u8,
    /// Compiled workgroup size on the x axis.
    pub workgroup_size_x: u32,
    /// Compiled workgroup size on the y axis.
    pub workgroup_size_y: u32,
    /// Compiled workgroup size on the z axis.
    pub workgroup_size_z: u16,
    /// Number of logical parameter declarations.
    pub parameter_count: u16,
    /// Reserved words; initialize all entries to zero.
    pub reserved_1: [u64; 5],
}

#[repr(C)]
/// One logical dispatch parameter declaration.
pub struct iree_hal_executable_dispatch_parameter_v0_t {
    /// ABI parameter kind: constant, binding or raw buffer pointer.
    pub parameter_type: u8,
    /// Payload bytes without padding; ignored for binding parameters.
    pub size: u8,
    /// Parameter behavior flags.
    pub flags: u16,
    /// Parameter-name ordinal, or `u16::MAX` if unnamed.
    pub name: u16,
    /// Byte offset in constants, or binding ordinal, according to kind.
    pub offset: u16,
}

#[repr(C)]
/// Placeholder for optional occupancy metadata.
pub struct iree_hal_executable_dispatch_occupancy_v0_t {
    /// Reserved field.
    pub reserved: i32,
}

#[repr(C)]
/// Source location associated with generated code.
pub struct iree_hal_executable_source_location_v0_t {
    /// Source line recorded by the compiler.
    pub line: u32,
    /// Length of the source path in bytes.
    pub path_length: u32,
    /// NUL-terminated source path on the build host.
    pub path: *const u8,
}

#[repr(C)]
/// Source locations associated with named compilation stages.
pub struct iree_hal_executable_stage_location_table_v0_t {
    /// Number of stage/location pairs.
    pub count: u32,
    /// NUL-terminated stage names corresponding to locations.
    pub names: *const *const u8,
    /// Array of source locations in stage-name order.
    pub locations: *const iree_hal_executable_source_location_v0_t,
}

#[repr(C)]
/// Export metadata arrays indexed by dispatch ordinal.
pub struct iree_hal_executable_export_table_v0_t {
    /// Number of exported dispatches.
    pub count: u32,
    /// Array of callable dispatch entry points.
    pub ptrs: *const DispatchFn,
    /// Dispatch requirements corresponding to entry points.
    pub attrs: *const iree_hal_executable_dispatch_attrs_v0_t,
    /// Optional per-dispatch parameter declaration arrays.
    pub params: *const *const iree_hal_executable_dispatch_parameter_v0_t,
    /// Optional occupancy metadata.
    pub occupancy: *const iree_hal_executable_dispatch_occupancy_v0_t,
    /// Optional diagnostic dispatch names.
    pub names: *const *const u8,
    /// Optional per-dispatch diagnostic tags.
    pub tags: *const *const u8,
    /// Optional parameter-name lookup table.
    pub parameter_names: *const *const u8,
    /// Optional per-dispatch source locations.
    pub source_locations: *const iree_hal_executable_source_location_v0_t,
    /// Optional per-dispatch compilation-stage location tables.
    pub stage_locations: *const iree_hal_executable_stage_location_table_v0_t,
}

#[repr(C)]
/// Executable specialization-constant declaration.
pub struct iree_hal_executable_constant_table_v0_t {
    /// Number of executable specialization constants.
    pub count: u32,
}

#[repr(C)]
/// Source file embedded for debugging.
pub struct iree_hal_executable_source_file_v0_t {
    /// Length of the embedded source path in bytes.
    pub path_length: u32,
    /// Pointer to the source path.
    pub path: *const u8,
    /// Length of the source contents in bytes.
    pub content_length: u32,
    /// Pointer to the source contents.
    pub content: *const u8,
}

#[repr(C)]
/// Embedded source file table.
pub struct iree_hal_executable_source_file_table_v0_t {
    /// Number of embedded files.
    pub count: u32,
    /// Array of embedded source file records.
    pub files: *const iree_hal_executable_source_file_v0_t,
}

#[repr(C)]
/// Versioned executable library record returned by the query entry point.
pub struct iree_hal_executable_library_v0_t {
    /// Pointer to the common version and feature header.
    pub header: *const iree_hal_executable_library_header_t,
    /// Imports requested by the library.
    pub imports: iree_hal_executable_import_table_v0_t,
    /// Exported dispatch functions and their requirements.
    pub exports: iree_hal_executable_export_table_v0_t,
    /// Specialization-constant declaration.
    pub constants: iree_hal_executable_constant_table_v0_t,
    /// Optional embedded source metadata.
    pub sources: iree_hal_executable_source_file_table_v0_t,
}
