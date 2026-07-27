use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use ariel_os::thread::sync::Channel;

use portable_atomic::{AtomicUsize, Ordering};

use heapless::Deque;

use super::{Executor, WorkItem};

use ariel_os::log::{debug, error, trace};

static JOB_REMAINING: AtomicUsize = AtomicUsize::new(0);

const TASK_SLOT_COUNT: usize = 4;

use ariel_os::thread::sync::Mutex;

static TASKS: Channel<usize> = Channel::new();

static TASK_QUEUE: Mutex<Deque<WorkItem, TASK_SLOT_COUNT>> =
    Mutex::new(Deque::new());

/// Ariel OS executor backed by a small fixed worker pool.
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
    fn schedule(&mut self, item: WorkItem) {

        loop {

            let result = {
                let mut queue = TASK_QUEUE.lock();
                queue.push_back(item)
            };
            match result {
                Ok(_) => {
                    JOB_REMAINING.fetch_add(1, Ordering::Relaxed);
                    break;
                }

                Err(_) => {
                    ariel_os::thread::yield_same();
                }
            }   

        }
        
        TASKS.send(&42);

    }

    fn wait_job_completion(&mut self) {
        while JOB_REMAINING.load(Ordering::Acquire) > 0 {
            ariel_os::thread::yield_same();
        }
    }
}

fn worker_loop() {
    let my_id = ariel_os::thread::current_tid().unwrap();
    let core = ariel_os::thread::core_id();
    debug!("[{:?}] Running on core {:?}", my_id, core);
    loop {

        let slot_id = TASKS.recv();

        let task = {
            let mut queue = TASK_QUEUE.lock();
            queue.pop_front()
        };
        match task {
            Some(t) => {
                t.run();
                JOB_REMAINING.fetch_sub(1, Ordering::Release);
            }

            None => {
                continue;
            }
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
