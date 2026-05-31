#[cfg(feature = "ariel-os")]
mod ariel_os;

#[cfg(feature = "ariel-os")]
pub use ariel_os::ArielOsExecutor;

/// A unit of work that can be scheduled by an executor.
///
/// Input: owned work item state.
/// Output: backend-defined result produced by the work item.
pub trait WorkItem: Send {
    type Output: Send;

    /// Runs this work item.
    ///
    /// Input: owned work item.
    /// Output: value produced by execution.
    fn run(self) -> Self::Output;
}

impl<F, Output> WorkItem for F
where
    F: FnOnce() -> Output + Send,
    Output: Send,
{
    type Output = Output;

    /// Runs a closure-backed work item.
    ///
    /// Input: owned closure.
    /// Output: closure return value.
    fn run(self) -> Self::Output {
        self()
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
    fn schedule<W>(&mut self, item: W) -> W::Output
    where
        W: WorkItem;
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
    fn schedule<W>(&mut self, item: W) -> W::Output
    where
        W: WorkItem,
    {
        item.run()
    }
}

#[cfg(feature = "ariel-os")]
pub type DefaultExecutor = ArielOsExecutor;

#[cfg(not(feature = "ariel-os"))]
pub type DefaultExecutor = SequentialExecutor;
