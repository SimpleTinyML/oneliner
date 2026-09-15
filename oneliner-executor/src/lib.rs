//! Onliner's interface for IREE's (parallelizable) workitem (model inference workload) execution. [`SequentialExecutor`] is the sequential scheduling implementation.
//!
//! [`SequentialExecutor`] runs each workitem during [`Executor::schedule`]. A
//! platform-specific executor may queue workitems and emits them cocurrently.
//!
//! ```
//! use oneliner_executor::{Executor, SequentialExecutor};
//! let mut executor = SequentialExecutor::new();
//! executor.wait_job_completion(); // Nothing is pending. Returns immediately.
//! ```
#![no_std]

use oneliner_iree_abi::{
    iree_hal_executable_dispatch_state_v0_t, iree_hal_executable_environment_v0_t,
    iree_hal_executable_workgroup_state_v0_t, DispatchFn,
};
use portable_atomic::{AtomicI32, Ordering};

/// A backend workload descriptor.
///
/// Copying a descriptor does not copy its buffers or extend their lifetimes.
/// Executors must not retain or replay it beyond the submission's completion
/// barrier.
#[derive(Clone, Copy)]
pub struct WorkItem {
    kind: WorkItemKind,
}

#[derive(Clone, Copy)]
enum WorkItemKind {
    IreeWorkload {
        dispatch_fn: DispatchFn,
        environment: *mut iree_hal_executable_environment_v0_t,
        dispatch_state: *mut iree_hal_executable_dispatch_state_v0_t,
        workgroup_state: iree_hal_executable_workgroup_state_v0_t,
        status: *const AtomicI32,
    },
}

impl WorkItem {
    /// Creates an IREE work item borrowing dispatch state owned by the caller.
    ///
    /// # Safety
    ///
    /// The function must match the IREE executable-library ABI. All pointers,
    /// including nested binding pointers, must remain valid until execution
    /// completes. Buffers must meet the kernel's alignment, size and access
    /// requirements; any parallel work must have race-free access. `status`
    /// must point to a live, aligned atomic value initialized by the dispatcher.
    #[doc(hidden)]
    pub unsafe fn iree(
        dispatch_fn: DispatchFn,
        environment: *mut iree_hal_executable_environment_v0_t,
        dispatch_state: *mut iree_hal_executable_dispatch_state_v0_t,
        workgroup_state: iree_hal_executable_workgroup_state_v0_t,
        status: *const AtomicI32,
    ) -> Self {
        Self {
            kind: WorkItemKind::IreeWorkload {
                dispatch_fn,
                environment,
                dispatch_state,
                workgroup_state,
                status,
            },
        }
    }

    /// Executes the workload once and records the status.
    ///
    /// Backend/executor must preserve the pointer's
    /// validity until this method returns.
    #[doc(hidden)]
    pub fn run(self) {
        match self.kind {
            WorkItemKind::IreeWorkload {
                dispatch_fn,
                environment,
                dispatch_state,
                workgroup_state,
                status,
            } => {
                let dispatch_status =
                    // SAFETY: the backend constructor and executor submission
                    // contract keep this ABI state alive until completion.
                    unsafe { dispatch_fn(environment, dispatch_state, &workgroup_state) };
                if dispatch_status != 0 {
                    let status = unsafe { &*status };
                    let _ = status.compare_exchange(
                        0,
                        dispatch_status,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
        }
    }
}

/// Schedules work items and provides a completion barrier.
pub trait Executor {
    /// Submits one work item; execution may finish immediately or be deferred.
    ///
    /// Implementations must execute each submitted item exactly once and retain
    /// no borrowed work after the next completion barrier returns.
    fn schedule(&mut self, item: WorkItem);

    /// Waits until all previously scheduled work has finished.
    ///
    /// All buffer writes and status updates must be visible to the caller on
    /// return. A dispatcher may then release its stack-allocated state.
    fn wait_job_completion(&mut self);
}

/// Executor that runs work items sequentially in submission order.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Creates a stateless executor that runs work in the calling context.
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for SequentialExecutor {
    fn schedule(&mut self, item: WorkItem) {
        item.run();
    }

    fn wait_job_completion(&mut self) {}
}
