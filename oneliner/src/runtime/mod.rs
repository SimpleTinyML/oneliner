mod arena;
mod buffer;
mod executor;
mod interface;

#[cfg(feature = "iree-runtime")]
mod iree;

pub use buffer::{
    concurrent, fill, Access,
    BufferRange, Buffer, BufferMut, BufferSource, AnyBufferRange, AnyBuffer
};
pub use executor::{DefaultExecutor, Executor, WorkItem};
pub use interface::{
    Error, ModelArtifacts, ModelInference, ModelSource, Shape, Tensor, Tensor4D,
    TensorArray,
};
#[cfg(feature = "iree-runtime")]
pub use iree::{
    dispatch, dispatch_fn_from_library, iree_hal_executable_dispatch_state_v0_t,
    iree_hal_executable_environment_v0_t, iree_hal_executable_import_thunk_v0_t,
    iree_hal_executable_import_v0_t, iree_hal_executable_library_header_t,
    iree_hal_executable_library_query_fn_t, iree_hal_executable_workgroup_state_v0_t,
    iree_hal_processor_v0_t, try_dispatch, try_dispatch_with_executor, DispatchFn,
};

pub use aligned::{Aligned, A16, A2, A32, A4, A64};
pub type AlignedType = A64;


pub use arena::{OwnedArena, SharedArena, ArenaStorage};
