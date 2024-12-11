use ndarray::{array, Array1, Axis};

//use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

mod bad_tests;
mod dct;
mod density;
mod ref_dct;
mod util;
mod wirelength;
mod wl_grad;

///in this oversimplified example, there will be 4 logic elements placed on a 16 x 16 grid
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

    let density = density::calc_density(&cell_centers, m); // mxm density calculation, overlaps of
                                                           // cells with bin , this is what we'll run
                                                           // our first DCT on
                                                           //fine, we'll make this mutable later so rust stops complaining
                                                           //minimum density overflowtau_min is 0.1 in their benchmarks, but we're only running for 10 or so
                                                           //iterations. Todo: precise definition of minimum density overflow. we'll only define if we use
                                                           //let minimum_overflow_density = 0.10;

    // the next step is lambda_0, that's equation 35, remember that the the equation we're
    // optimizing for is W(v) + lamdba * N(v), where v is the placement solution
    // lambda_0 requires that we have the gradient of the wirelength estimator as well as the
    // initial electric field calculatedq, so we'll have to build up to that. Getting the electric
    // field will requires us to run a DCT, calculate psi, etc, so we're going to do all that for
    // the first time here:

    //first up, wirelength estimation! we're not even going to calculate the wirelength yet, just
    //all the gradients. Remember that wirelength is a scalar function: you give it the placement
    //vector / list of cell centers and then it gives you a single number.
    //
    //However, the gradient of the wirelength will be a an array with cell count *2
    //  elements -> partial x and partial y for each of our logic elements. The really interesting thing is that we may
    //not need the wirelength estimation in order to calculate its gradient. The gradient formula is
    //very well defined, and while it's related to the wirelength function it doesn't necessarily
    //depend on it.

    //wirelength gradient, the gradient of the wirelength estimator from equation 6
    //going to be a 1d array of values, even values are x, odd values are y

    let wl_gradient_0 = wl_grad::calc_wl_grad(&cell_centers);

    //we'll also need the electric field, which requires some cosine/sine transforms of the density matrix

    let charges = array![2.25, 2.25, 2.25, 2.25];

    //mxm matrix of a_uvs, calculated via 2D DCT, but ultimately coming from eq 21)
    let coeffs = dct::calc_coeffs(&density, m);

    //these functions calculation the electric field for each bin
    let elec_field_x = dct::elec_field_x(&coeffs, m);
    let elec_field_y = dct::elec_field_y(&coeffs, m);

    // the denominator of equation 35 is depends on the electric field, so we'll use our elec_field_x's and
    //our elec_field_y's to get the electric field for each cell. We do this by multipyling the overlap of
    // of the cell with each bin with the corresponding electric field in the given direction in a bin
    // the key line from the called functions is  lec_field += cell_overlap * bins_elec_field[[u, v]];

    //for these, we are not yet multiplying anything by q_i (the charge amount, fixed 2.25 here)

    //Axis(0) is columns, Axis(1) is rows! I think.
    let cell_fields_x =
        cell_centers.map_axis(Axis(1), |x| dct::apply_bins_to_cell(&x, &elec_field_x, m));

    let cell_fields_y =
        cell_centers.map_axis(Axis(1), |x| dct::apply_bins_to_cell(&x, &elec_field_y, m));

    //the numerator of equation 35 is the absolute values of the x and y components of each gradient,
    //all summmed together

    let lambda_0_upper = wl_gradient_0.map(|x| x.abs()).sum(); //from eq 35

    //the denominator is equal to the absolute value of each component of the electric field times the charge of the cell (fixed at 1.5*1.5=2.25, here, since that's our area)
    let lambda_lower_x = cell_fields_x.map(|x| 2.25 * x.abs()).sum();
    let lambda_lower_y = cell_fields_y.map(|y| 2.25 * y.abs()).sum();

    let lambda_0_lower = lambda_lower_x + lambda_lower_y;

    let lambda_0 = lambda_0_upper / lambda_0_lower;

    //inital alpha_0^max from the eplace algorithm (alg 3) on page 24 is 0.044 * bin width, so we'll
    //just set it to 0.044
    let alpha_0 = 0.044 * 1.;

    //we still have to initialize some other parameters, but once we do:
    let total_pot = dct::total_potential(&coeffs, &cell_centers, m);

    let iter_max = 1; //maximum number of iterations - will be 10 once the loop
                      //is done

    for k in 0..iter_max {
        let wl = wirelength::wl(&cell_centers, 0.2);
        println!("estimated wirelength for the current iteration: ");
        dbg!(&wl);

        println!("estimated potential/penalty function times lambda");
        let lambda_pot = lambda_0 * dct::total_potential(&coeffs, &cell_centers, m);
        dbg!(&lambda_pot);

        let f_k = wl + lambda_pot;
        println!("total objective function f_k for this iteration:");
        dbg!(f_k);

        //each of these arrays is of length (#of cells)*2. This is the gradient of N, our penalty function
        let grad_penalty_k = penalty_grad(&cell_fields_x, &cell_fields_y);

        //technically running this function twice on the same data, but once we're actually looping
        //it should make sense.
        let wl_gradient = wl_grad::calc_wl_grad(&cell_centers);

        //the gradient of f_k is the gradient of the wirelength + lambda * the gradient of the penalty function
        //taking the gradient is (fortunately!) a very linear operation, if f_k = wl + lambda * N,
        //grad f_k = grad(wl) + lambda * grad(N). This works very nicely even though f_k is a scalar and grad f_k is a
        //vector.

        //note that this gradient should be preconditioned before being fed to the solver! that's up next 
        let grad_f_k: Array1<f64> = wl_gradient + lambda_0 * grad_penalty_k;

        //let new_placement = NL_Solver( &cell_centers, f_k, grad_f_k, alpha);
        println!("latest thing calculated: grad_f_k. Next up: preconditioning and NL_Solver");
    }
}

//interleaves/zips two arrays. I'd do this with list comprehensions and a flatten in python or haskell, unsure how
// to do here.
fn penalty_grad(cell_fields_x: &Array1<f64>, cell_fields_y: &Array1<f64>) -> Array1<f64> {
    let mut grad_penalty_k = Array1::<f64>::zeros(2 * cell_fields_x.len());
    for i in 0..cell_fields_x.len() {
        grad_penalty_k[i] = 2.25 * cell_fields_x[i];
        grad_penalty_k[i + 1] = 2.25 * cell_fields_y[i];
    }
    grad_penalty_k
}
