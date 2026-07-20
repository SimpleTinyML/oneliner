#[cfg(feature = "ariel-os")]
mod ariel_os;

#[cfg(feature = "ariel-os")]
pub use ariel_os::ArielOsExecutor;

use super::{DispatchFn, iree_hal_executable_environment_v0_t, iree_hal_executable_dispatch_state_v0_t, iree_hal_executable_workgroup_state_v0_t};

#[derive(Clone, Copy)]
pub enum WorkItem {
    IREEWorkload {
        dispatch_fn: DispatchFn,
        environment: *mut iree_hal_executable_environment_v0_t,
        dispatch_state: *mut iree_hal_executable_dispatch_state_v0_t,
        workgroup_state: iree_hal_executable_workgroup_state_v0_t,

    },
}

impl WorkItem {
    pub fn run(self) {
        match self {
            WorkItem::IREEWorkload {
                dispatch_fn,
                environment,
                dispatch_state,
                workgroup_state,
            } => unsafe {
                dispatch_fn(
                    environment,
                    dispatch_state,
                    &workgroup_state,
                );
            },
        }
    }
}


/// Synchronously schedules work items for execution.
///
/// Input: work items implementing `WorkItem`.
/// Output: each work item's execution result.
pub trait Executor {
    /// Schedules and executes one work item.
    ///
    /// Input: work item to run.
    /// Output: result returned by the work item.
    fn schedule(&mut self, item: WorkItem);

    fn wait_job_completion(&mut self);

}

/// Default executor that runs work items immediately in submission order.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Creates a sequential executor.
    ///
    /// Input: none.
    /// Output: executor instance with no internal state.
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for SequentialExecutor {
    /// Runs the work item immediately on the current thread.
    ///
    /// Input: work item to execute.
    /// Output: result returned by the work item.
    fn schedule(&mut self, item: WorkItem)

    {
        item.run();
    }

    fn wait_job_completion(&mut self) {
        
    }
    
}

#[cfg(feature = "ariel-os")]
pub type DefaultExecutor = ArielOsExecutor;

#[cfg(not(feature = "ariel-os"))]
pub type DefaultExecutor = SequentialExecutor;