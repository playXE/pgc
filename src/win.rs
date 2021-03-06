use processthreadsapi::*;
use std::sync::atomic::{AtomicBool, AtomicI8, Ordering};
use winapi::um::handleapi::*;
use winapi::um::synchapi::*;
use winapi::um::winnt::*;
use winapi::um::*;
use winapi::*;

pub struct StkRoot {
    pub thread_handle: *mut u8,
    pub suspend_ack: AtomicBool,
}

unsafe impl Send for StkRoot {}
unsafe impl Sync for StkRoot {}

static mut GC_INSIDE_COLLECT: AtomicI8 = AtomicI8::new(-1);

use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    pub static THREAD: RefCell<Arc<StkRoot>> = unsafe {
        let handle = GetCurrentThread();
        let process = GetCurrentProcess();
        let mut thread = StkRoot {
            thread_handle: std::ptr::null_mut(),
            suspend_ack: AtomicBool::new(false)
        };
        DuplicateHandle(process, handle, process, (&mut thread.thread_handle) as *mut *mut u8 as *mut *mut _, 0,0, DUPLICATE_SAME_ACCESS);
        let r = Arc::new(thread);
        THREADS.write().push(r.clone());
        RefCell::new(r)
    };
}

lazy_static::lazy_static! {
    pub static ref THREADS: parking_lot::RwLock<Vec<Arc<StkRoot>>> = parking_lot::RwLock::new(vec![]);
}

pub unsafe fn mutator_suspend() {
    GC_INSIDE_COLLECT.store(1, Ordering::Relaxed);
    let lock = THREADS.read();
    for root in lock.iter() {
        if !THREAD.with(|t| Arc::ptr_eq(root, &t.borrow())) {
            if SuspendThread(root.thread_handle as *mut _) == !0 {
                panic!("GC: Cannot suspend thread");
            }
        }
    }

    for root in lock.iter() {
        root.suspend_ack.store(true, Ordering::Relaxed);
    }
}

pub unsafe fn mutator_resume() {
    let lock = THREADS.read();
    for root in lock.iter() {
        if !THREAD.with(|t| Arc::ptr_eq(root, &t.borrow())) {
            if ResumeThread(root.thread_handle as *mut _) == !0 {
                panic!("GC: Cannot resume thread");
            }
        }
    }
    GC_INSIDE_COLLECT.store(0, Ordering::Relaxed);
}
