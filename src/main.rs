use ndarray::array;

//use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

mod dct;
mod density;
mod ref_dct;
mod util;
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
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 4.0],
        [6.0, 6.0],
        [5.5, 5.5],
        [4.75, 4.75],
        [5.25, 5.25],
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
    // initial electric field calculated, so we'll have to build up to that. Getting the electric
    // field will requires us to run a DCT, calculate psi, etc, so we're going to do all that for
    // the first time here:

    //first up, wirelength estimation! we're not even going to calculate the wirelength yet, just
    //all the gradients. Remember that wirelength is a scalar function: you give it the placement
    //vector / list of cell centers and then it gives you a single number.
    //
    //However, the gradient of the wirelength will be a an array with 8 elements -> partial x and
    //partial y for each of our four logic elements. The really interesting thing is that we may
    //not need the wirelength estimation in order to calculate its gradient. The gradient formula is
    //very well defined, and while it's related to the wirelength function it doesn't necessarily
    //depend on it.

    let wl_gradient = wl_grad::calc_wl_grad(&cell_centers);

    //we'll also need the electric field, which requires some cosine/sine transforms of the density matrix

    let lambda_0_upper = wl_gradient.map(|partialdiv| partialdiv.abs()).sum(); //from eq 35
    let charges = array![2.25, 2.25, 2.25, 2.25];

    let coeffs = dct::calc_coeffs(&density, m);
    let elec_field_x = dct::elec_field_x(&coeffs, m);
    // let elec_field_y = dct::elec_field_y(&coeffs, m);

    dct::check_density(&coeffs, &density, m);

    let slow_elec_x = ref_dct::ref_elec_field_x(&coeffs, m);
    dct::test_elec_field_x(&coeffs, &slow_elec_x, m);

    let elec_field_y = dct::elec_field_y(&coeffs, m);
}
