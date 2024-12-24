use ndarray::array;

//use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

mod bad_tests;
mod dct;
mod density;
mod eplace;
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

    let initial_loop_params = eplace::calc_initial_params(&cell_centers, m);

    //ok, this will eventually be in a loop somehow
    let new_placement = eplace::eplace(initial_loop_params, m);

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

    //we'll also need the electric field, which requires some cosine/sine transforms of the density matrix

    //mxm matrix of a_uvs, calculated via 2D DCT, but ultimately coming from eq 21)

    //these functions calculation the electric field for each bin

    // the denominator of equation 35 is depends on the electric field, so we'll use our elec_field_x's and
    //our elec_field_y's to get the electric field for each cell. We do this by multipyling the overlap of
    // of the cell with each bin with the corresponding electric field in the given direction in a bin
    // the key line from the called functions is  lec_field += cell_overlap * bins_elec_field[[u, v]];

    //for these, we are not yet multiplying anything by q_i (the charge amount, fixed 2.25 here)

    //Axis(0) is columns, Axis(1) is rows! I think.

    //inital alpha_0^max from the eplace algorithm (alg 3) on page 24 is 0.044 * bin width, so we'll
    //just set it to 0.044

    //gamma is defined in equation 38 on page 23, it's based on the density overflow tau
    //the paper's assumption is that starting density overflow is equal to 1.0 - we'll throw in the actual formula
    // in a bit

    //again, assuming initial_density_overflow is 1

    //each of these arrays is of length (#of cells)*2. This is the gradient of N, our penalty function

    //the gradient of f_k is the gradient of the wirelength + lambda * the gradient of the penalty function
    //taking the gradient is (fortunately!) a very linear operation, if f_k = wl + lambda * N,
    //grad f_k = grad(wl) + lambda * grad(N). This works very nicely even though f_k is a scalar and grad f_k is a
    //vector.

    //note that this gradient should be preconditioned before being fed to the solver! that's up next

    //for the initial loop, placement and reference placement are the same. they'll
    //be different in later loops (hence the clone here, which won't exist later)
}

//interleaves/zips two arrays. I'd do this with list comprehensions and a flatten in python or haskell, unsure how
// to do here.
