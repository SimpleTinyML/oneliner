use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use ariel_os::thread::sync::Channel;

use portable_atomic::{fence, AtomicUsize, Ordering};

static JOB_REMAINING: AtomicUsize = AtomicUsize::new(0);

use super::{Executor, WorkItem};

use ariel_os::log::{debug, error, trace};

pub const ARIEL_OS_EXECUTOR_WORKER_STACK_SIZE: usize = 2048;
pub const ARIEL_OS_EXECUTOR_WORKER_PRIORITY: u8 = 1;

const TASK_SLOT_COUNT: usize = 4;
const TASK_SLOT_FREE: usize = 0;
const TASK_SLOT_BUSY: usize = 1;

#[derive(Clone, Copy)]
struct TaskMessage {
    run: unsafe fn(WorkItem, usize),
    item: WorkItem,
}

// #[derive(Clone, Copy)]
struct TaskSlot {
    state: AtomicUsize,
    message: UnsafeCell<MaybeUninit<TaskMessage>>,
}

unsafe impl Sync for TaskSlot {}

impl TaskSlot {
    const fn new() -> Self {
        Self {
            state: AtomicUsize::new(TASK_SLOT_FREE),
            message: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    fn write(&self, task: TaskMessage) {
        unsafe {
            (*self.message.get()).write(task);
        }
    }

    fn read(&self) -> TaskMessage {
        unsafe { (*self.message.get()).assume_init_read() }
    }
}

static TASKS: Channel<usize> = Channel::new();
static TASK_SLOTS: [TaskSlot; TASK_SLOT_COUNT] = [
    TaskSlot::new(),
    TaskSlot::new(),
    TaskSlot::new(),
    TaskSlot::new(),
];

fn acquire_task_slot() -> usize {
    loop {
        for (id, slot) in TASK_SLOTS.iter().enumerate() {
            if slot
                .state
                .compare_exchange(
                    TASK_SLOT_FREE,
                    TASK_SLOT_BUSY,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return id;
            }
        }

        ariel_os::thread::yield_same();
    }
}

fn release_task_slot(id: usize) {
    TASK_SLOTS[id]
        .state
        .store(TASK_SLOT_FREE, Ordering::Release);
}


/// Ariel OS executor backed by a small fixed worker pool.
///
/// Work items are sent to Ariel OS worker threads and `schedule` waits until
/// the selected worker has written the result back to the caller's stack frame.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArielOsExecutor;

impl ArielOsExecutor {
    /// Creates an Ariel OS executor handle.
    ///
    /// Input: none.
    /// Output: stateless executor handle using the global worker pool.
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for ArielOsExecutor {
    /// Runs the work item on an Ariel OS worker thread.
    ///
    /// Input: work item to execute.
    /// Output: result returned by the work item.
    fn schedule(&mut self, item: WorkItem)
    {
        unsafe fn take_and_run(item: WorkItem, accepted: usize)

        {
            item.run();
        }

        // let accepted: Channel<()> = Channel::new();
        let task = TaskMessage {
            run: take_and_run,
            item: item,
        };
        let slot_id = acquire_task_slot();

        if slot_id >= TASK_SLOT_COUNT {
            error!("Scheduler acquired invalid task slot: {}", slot_id);
            panic!("Invalid task slot");
        }

        TASK_SLOTS[slot_id].write(task);

        JOB_REMAINING.fetch_add(1, Ordering::Relaxed);
        let my_id = ariel_os::thread::current_tid().unwrap();
        // fence(Ordering::Release);
        trace!(
            "[{:?}]Scheduling task: slot={}, run={:?}, item={:?}",
            my_id, slot_id, task.run, &task.item as *const WorkItem, 
        );
        // let slot_ptr = &slot as *const TaskSlot as usize;
        TASKS.send(&slot_id);
        trace!("[{:?}]Finish Scheduling Task: slot={}, slot_id_addr = {:?}", my_id, slot_id, &slot_id as *const usize);
        // accepted.recv();
    }

    fn wait_job_completion(&mut self) {
        while JOB_REMAINING.load(Ordering::Relaxed) > 0 {
            ariel_os::thread::yield_same();
        }
    }
}

fn worker_loop() -> () {
    let my_id = ariel_os::thread::current_tid().unwrap();
    let core = ariel_os::thread::core_id();
    debug!("[{:?}] Runining at [{:?}] ...", my_id, core);
    loop {
        trace!("[{:?}] Wating for Task...", my_id);
        let slot_id = TASKS.recv();
        trace!(
            "[{:?}] Worker received task: slot={}, slot_id_addr = {:?}",
            my_id, slot_id, &slot_id as *const usize
        );

        if slot_id >= TASK_SLOT_COUNT {
            error!("[{:?}] Worker received invalid task slot: {}", my_id, slot_id);
            panic!("[{:?}] Invalid task slot", my_id);
        }

        // fence(Ordering::Acquire);
        let task = TASK_SLOTS[slot_id].read();
        trace!(
            "[{:?}] Worker received task: slot={}, run={:?}, item={:?}, ",
            my_id, slot_id, task.run, &task.item as *const WorkItem,
        );
        unsafe {
            task.item.run();
            trace!(
            "[{:?}] Worker Finished task: slot={}, slot_id_addr = {:?}",
            my_id, slot_id, &slot_id as *const usize
        );
            release_task_slot(slot_id);
            JOB_REMAINING.fetch_sub(1, Ordering::Relaxed);
        }
    }
}


#[ariel_os::thread(autostart)]
fn oneliner_ariel_os_worker_0() {

    worker_loop();
}


#[ariel_os::thread(autostart)]
fn oneliner_ariel_os_worker_1() {

    worker_loop();
}
