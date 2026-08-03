#![no_std]
#![allow(non_camel_case_types)]

use core::ffi::c_void;

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
#[derive(Clone, Copy)]
pub struct iree_hal_executable_environment_v0_t {
    pub constants: *const u32,
    pub import_thunk: Option<iree_hal_executable_import_thunk_v0_t>,
    pub import_funcs: *const iree_hal_executable_import_v0_t,
    pub import_contexts: *const *const c_void,
    pub processor: iree_hal_processor_v0_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
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
#[derive(Clone, Copy)]
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
pub struct iree_hal_executable_import_table_v0_t {
    pub count: u32,
    pub symbols: *const *const u8,
}

#[repr(C)]
pub struct iree_hal_executable_dispatch_attrs_v0_t {
    pub flags: u64,
    pub local_memory_pages: u16,
    pub constant_count: u8,
    pub binding_count: u8,
    pub workgroup_size_x: u32,
    pub workgroup_size_y: u32,
    pub workgroup_size_z: u16,
    pub parameter_count: u16,
    pub reserved_1: [u64; 5],
}

#[repr(C)]
pub struct iree_hal_executable_dispatch_parameter_v0_t {
    pub parameter_type: u8,
    pub size: u8,
    pub flags: u16,
    pub name: u16,
    pub offset: u16,
}

#[repr(C)]
pub struct iree_hal_executable_dispatch_occupancy_v0_t {
    pub reserved: i32,
}

#[repr(C)]
pub struct iree_hal_executable_source_location_v0_t {
    pub line: u32,
    pub path_length: u32,
    pub path: *const u8,
}

#[repr(C)]
pub struct iree_hal_executable_stage_location_table_v0_t {
    pub count: u32,
    pub names: *const *const u8,
    pub locations: *const iree_hal_executable_source_location_v0_t,
}

#[repr(C)]
pub struct iree_hal_executable_export_table_v0_t {
    pub count: u32,
    pub ptrs: *const DispatchFn,
    pub attrs: *const iree_hal_executable_dispatch_attrs_v0_t,
    pub params: *const *const iree_hal_executable_dispatch_parameter_v0_t,
    pub occupancy: *const iree_hal_executable_dispatch_occupancy_v0_t,
    pub names: *const *const u8,
    pub tags: *const *const u8,
    pub parameter_names: *const *const u8,
    pub source_locations: *const iree_hal_executable_source_location_v0_t,
    pub stage_locations: *const iree_hal_executable_stage_location_table_v0_t,
}

#[repr(C)]
pub struct iree_hal_executable_constant_table_v0_t {
    pub count: u32,
}

#[repr(C)]
pub struct iree_hal_executable_source_file_v0_t {
    pub path_length: u32,
    pub path: *const u8,
    pub content_length: u32,
    pub content: *const u8,
}

#[repr(C)]
pub struct iree_hal_executable_source_file_table_v0_t {
    pub count: u32,
    pub files: *const iree_hal_executable_source_file_v0_t,
}

#[repr(C)]
pub struct iree_hal_executable_library_v0_t {
    pub header: *const iree_hal_executable_library_header_t,
    pub imports: iree_hal_executable_import_table_v0_t,
    pub exports: iree_hal_executable_export_table_v0_t,
    pub constants: iree_hal_executable_constant_table_v0_t,
    pub sources: iree_hal_executable_source_file_table_v0_t,
}
