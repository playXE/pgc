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

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Drop");
    }
}

fn main() {
    {
        let r = Rooted::new(Foo::None);
    }
    gc_collect();
}
