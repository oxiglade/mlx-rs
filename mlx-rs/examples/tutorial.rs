use mlx_rs::transforms::grad;
use mlx_rs::{Array, Dtype};

fn scalar_basics() {
    // create a scalar array
    let x: Array = 1.0.into();

    // the datatype is .float32
    let dtype = x.dtype();
    assert_eq!(dtype, Dtype::Float32);

    // get the value
    let s = x.item_exact::<f32>();
    assert_eq!(s, 1.0);

    // reading the value with a different type is a fatal error
    // let i = x.item_exact::<i32>();

    // scalars have a size of 1
    let size = x.size();
    assert_eq!(size, 1);

    // scalars have 0 dimensions
    let ndim = x.ndim();
    assert_eq!(ndim, 0);

    // scalar shapes are empty arrays
    let shape = x.shape();
    assert!(shape.is_empty());
}

#[allow(unused_variables)]
fn array_basics() {
    // make a multidimensional array.
    let x: Array = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // mlx is row-major by default so the first row of this array
    // is [1.0, 2.0] and the second row is [3.0, 4.0]

    // Make an array of shape {2, 2} filled with ones:
    let y = Array::ones::<f32>(&[2, 2]).unwrap();

    // Pointwise add x and y:
    let z = x.add(&y);

    // Same thing:
    let mut z = &x + &y;

    // mlx is lazy by default. At this point `z` only
    // has a shape and a type but no actual data:
    assert_eq!(z.dtype(), Dtype::Float32);
    assert_eq!(z.shape(), vec![2, 2]);

    // To actually run the computation you must evaluate `z`.
    // Under the hood, mlx records operations in a graph.
    // The variable `z` is a node in the graph which points to its operation
    // and inputs. When `eval` is called on an array (or arrays), the array and
    // all of its dependencies are recursively evaluated to produce the result.
    // Once an array is evaluated, it has data and is detached from its inputs.
    z.eval().unwrap();

    // Of course the array can still be an input to other operations. You can even
    // call eval on the array again, this will just be a no-op:
    z.eval().unwrap(); // no-op

    // Some functions or methods on arrays implicitly evaluate them. For example
    // accessing a value in an array or printing the array implicitly evaluate it:
    z = Array::ones::<f32>(&[1]).unwrap();
    z.item_exact::<f32>(); // implicit evaluation

    z = Array::ones::<f32>(&[2, 2]).unwrap();
    println!("{z}"); // implicit evaluation
}

fn automatic_differentiation() {
    use mlx_rs::error::Result;

    fn f(x: &Array) -> Result<Array> {
        x.square()
    }

    fn calculate_grad(func: impl Fn(&Array) -> Result<Array>, arg: &Array) -> Result<Array> {
        grad(&func)(arg)
    }

    let x = Array::from(1.5);

    let dfdx = calculate_grad(f, &x).unwrap();
    assert_eq!(dfdx.item_exact::<f32>(), 2.0 * 1.5);

    let dfdx2 = calculate_grad(|args| calculate_grad(f, args), &x).unwrap();
    assert_eq!(dfdx2.item_exact::<f32>(), 2.0);
}

fn main() {
    scalar_basics();
    array_basics();
    automatic_differentiation();
}
