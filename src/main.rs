extern crate pgc;

use pgc::*;
struct Node {
    i: i32,
    j: i32,
    left: Option<Gc<Node>>,
    right: Option<Gc<Node>>,
}

impl Node {
    pub const fn new(left: Option<Gc<Node>>, right: Option<Gc<Node>>) -> Self {
        Self {
            i: 0,
            j: 0,
            left,
            right,
        }
    }

    pub const fn leaf() -> Self {
        Self::new(None, None)
    }
}

const ARRAY_SIZE: usize = 500000;
const MIN_THREE_DEPTH: usize = 4;
const MAX_THREE_DEPTH: usize = 16;

unsafe impl GcObject for Node {
    fn references(&self) -> Vec<Gc<dyn GcObject>> {
        let mut v: Vec<Gc<dyn GcObject>> = vec![];
        v.extend(self.left.references());
        v.extend(self.right.references());

        v
    }
}

const fn tree_size(i: i32) -> i32 {
    (1 << (i + 1)) - 1
}

const fn num_iters(depth: i32, i: i32) -> i32 {
    return 4 * tree_size(depth) / tree_size(i);
}

fn populate(mut depth: i32, this_node: Gc<Node>) {
    if depth <= 0 {
        return;
    } else {
        depth -= 1;
        this_node.get_mut().left = Some(Gc::new(Node::leaf()));
        this_node.get_mut().right = Some(Gc::new(Node::leaf()));
        populate(depth, this_node.get().left.unwrap());
        populate(depth, this_node.get().right.unwrap());
    }
}

fn make_tree(depth: i32) -> Gc<Node> {
    if depth <= 0 {
        return Gc::new(Node::leaf());
    } else {
        return Gc::new(Node::new(
            Some(make_tree(depth - 1)),
            Some(make_tree(depth - 1)),
        ));
    }
}

fn time_construction(s: i32, depth: i32) {
    let num_iters = num_iters(s, depth);

    let start = time::PreciseTime::now();

    let mut i = 0;
    while i < num_iters {
        let temp_tree = Rooted::new(Node::leaf());
        populate(depth, temp_tree.inner());
        i += 1;
    }
    let finish = time::PreciseTime::now();
    println!(
        "Top down construction took {}ms",
        start.to(finish).num_milliseconds()
    );
    let start = finish;

    for _ in 0..num_iters {
        let _ = make_tree(depth);
    }
    let end = time::PreciseTime::now();
    println!(
        "Bottom up construction took {}ms",
        start.to(end).num_milliseconds()
    );
}

fn main() {
    let depth = 10;

    let start = time::PreciseTime::now();
    make_tree(depth + 1);
    let long_lived = Rooted::new(Node::leaf());
    populate(depth, long_lived.inner());
    let _ = Gc::new(vec![0.0f64; ARRAY_SIZE]);

    let mut d = MIN_THREE_DEPTH;
    while d <= depth as usize {
        time_construction(depth, d as _);
        d = d + 2;
    }
    let finish = time::PreciseTime::now();
    gc_collect();
    println!("Completed in {}ms", start.to(finish).num_milliseconds());
}
