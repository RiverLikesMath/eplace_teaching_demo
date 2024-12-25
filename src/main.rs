use ndarray::array;

mod bad_tests;
mod dct;
mod density;
mod eplace;
mod ref_dct;
mod util;
mod wirelength;
mod wl_grad;

///in this oversimplified example, there will be a small number of  logic elements placed on an mxm grid
///each logic element will be 1.5 pixel wide and tall so we don't have to worry about eplace's
///smoothing.
///for simplicities sake, we'll assume all the elements are on a single global net. this is very
///silly, but it'll make the demonstration a bit easier
#[allow(unused_variables)] //we're allowing unused variables in main here, at least for now.
fn main() {
    let cell_centers = array![
        [28. / 8., 28. / 8.], //x,y, initial placement
        [56. / 8., 58. / 8.],
        [9. / 8., 19. / 8.],
        [99. / 8., 101. / 8.0],
        [10.8, 6.7],
        [1.27, 2.04],
        [14.81, 14.25],
        [14.22, 14.44],
        [1.5, 1.5],
        [1.9, 12.9],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 4.0],
        [6.0, 6.0],
        [5.5, 5.5],
        [4.75, 4.75],
        [5.25, 12.25],
    ];

    let m: usize = 16; // m == sqrt(number of bins), max 1024, must be power of 2

    //before we start the loop, we need an initial set of parameters to feed to
    //the nl solver - those are all calculated and grabbed from here
    let initial_loop_params = eplace::calc_initial_params(&cell_centers, m);

    //my heart is telling me to do this recursively, but the closest thing to the paper
    //would be a for loop
    let mut curr_eplace_iteration = initial_loop_params;
    for i in 1..10 {
        println!("Beginning new loop");
        println!("current iteration is: {i}");

        let prev = curr_eplace_iteration;
        curr_eplace_iteration = eplace::eplace(prev, m);

        println!("current objective function is: ");
        dbg!(curr_eplace_iteration.f_k);
    }
}
