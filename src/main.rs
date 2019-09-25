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
    //enable_incremental();
    {
        let v = Rooted::new(Foo::Node(0, Gc::new(Foo::None)));
        match v.get_mut() {
            Foo::Node(_, n) => *n = v.inner(),
            _ => (),
        }
    }
    gc_collect();
    //gc_collect_medium();
}
