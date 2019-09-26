extern crate pgc;
use pgc::*;

enum Foo {
    None,
    Node(i32, Gc<Foo>),
}

unsafe impl GcObject for Foo {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut v: Vec<Gc<dyn GcObject>> = vec![];
        match self {
            Foo::Node(_, n) => v.push(*n),
            _ => (),
        }
        v
    }
}

fn main() {
    unsafe {
        use std::io::Write;
        use std::sync::atomic::*;
        use win::*;
        std::thread::spawn(|| {});
        mutator_suspend();

        mutator_resume();
    }
}
