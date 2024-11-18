use ndarray::{array, Array1, Array2, ArrayView1, Axis};

//use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};
use std::f64::consts::PI;

mod density;
mod wl_grad;
//TODO: FUTURE SO BRIGHT I NEED SHADES

// upload to github
// split into multiple files, i think, main's getting too big, how does that work in rust
// figure out the inverse 2D DCT, inverse DCST, and inverse DSCT

const bin_w: f64 = 1.; //bin width and height
const bin_area: f64 = 1.;

///in this oversimplified example, there will be 4 logic elements placed on a 16 x 16 grid
///
///each logic element will be 1.5 pixel wide and tall so we don't have to worry about eplace's
///smoothing.
///for simplicities sake, we'll assume all the elements are on a single global net. this is very
//silly, but it'll make the demonstration a bit easier
fn main() {
    let cell_centers = array![
        [28. / 8., 28. / 8.], //x,y, initial placement
        [56. / 8., 58. / 8.],
        [9. / 8., 19. / 8.],
        [99. / 8., 101. / 8.0],
        //  [10.8, 6.7],
        //  [ 1.27, 2.04]
    ];

    let m: usize = 16; // m == sqrt(number of bins), max 1024, must be pover of 2

    let mut density = density::calc_density(&cell_centers, m); // mxm density calculation, overlaps of
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
    dbg!(&wl_gradient);

    //we'll also need the electric field, which requires some cosine/sine transforms of the density matrix

    let lambda_0_upper = wl_gradient.map(|partialdiv| partialdiv.abs()).sum(); //from eq 35
    let charges = array![2.25, 2.25, 2.25, 2.25];

    let coefficients = dct_coeff(&density, m);
    //inverse cosine transform of coefficients will get you the density
    //the plan -> multiply coefficients by the relevant factor from equation 23, inverse 2D cosine transform, that gets you potential for each bin
    //multiply the coefficients by relevant factor from equation 24, inverse sine cosine transform, that gets you electric field in the X direction
    //multiply the coefficients by the relevant Y factors from equation 24, inverse cosine sine tranform, that gets you electric field in the Y direction.

    //I've already confirmed that the transforms as described in the paper work beautifully for density, even with only
    //16x16 bins, at least as far as the toy example here goes.
    //    let potential = eplace_potential(&coefficients, m); //not yet implemented
    //    let elec_field_x = eplace_elec_x(&coefficients, m); //not yet implemented
    //    let elec_field_y = eplace_elec_y(&coefficients, m); //not yet implemented
    //each of these functions will generate a potential, electric field x, or electric field y for *each* bin!
    //those will then be applied to given cells
}

fn calc_w(index: usize, m: usize) -> f64 {
    2.0 * PI * (index as f64) / (m as f64)
}
fn dct_coeff(density: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut coefficients = Array2::<f64>::zeros((m, m));
    for u in 0..m {
        for v in 0..m {
            coefficients[[u, v]] = eplace_dct_auv(density, m, u, v);
        }
    }
    coefficients
}
fn eplace_dct(coefficients: &Array2<f64>, m: usize, x: f64, y: f64) -> f64 {
    let mut density_dct = 0.;

    for u in 0..m {
        for v in 0..m {
            density_dct += coefficients[[u, v]]
                * (calc_w(u, m) * (x as f64)).cos()
                * (calc_w(v, m) * (y as f64)).cos();
            if u == 0 && v < 3 {
                dbg!(
                    coefficients[[u, v]],
                    calc_w(u, m),
                    calc_w(v, m),
                    u,
                    v,
                    x,
                    y,
                    &density_dct
                );
            }
        }
    }
    //subtract DC component
    dbg!(m);
    dbg!(&density_dct);
    //    dbg!(&density);
    density_dct -= coefficients[[0, 0]];
    density_dct
}

fn eplace_dct_auv(density: &Array2<f64>, m: usize, u: usize, v: usize) -> f64 {
    let scale_factor = 1_f64 / (m.pow(2) as f64); //1/m^2

    let mut coefficient: f64 = 0.0; // a_u,v
    for x in 0..m {
        for y in 0..m {
            coefficient += scale_factor
                * density[[x, y]]
                * (calc_w(u, m) * (x as f64)).cos()
                * (calc_w(v, m) * (y as f64)).cos();
        }
    }
    2. * coefficient
}

///probably not going to use this, just pick its code for parts and switch directly to fft library
fn elec_field(cell_centers: &Array2<f64>, coefficients: &Array2<f64>, m: usize) -> Array1<f64> {
    let mut elec_field = Array1::<f64>::zeros(cell_centers.len());
    dbg!(&cell_centers);
    for i in 0..elec_field.len() {
        for u in 0..m {
            for v in 0..m {
                let w_u = calc_w(u, m);
                let w_v = calc_w(v, m);
                let x = cell_centers[[i / 2, 0]];
                let y = cell_centers[[i / 2, 1]];
                if u == 0 && v == 0
                //formula as written has a divide by zero here
                {
                    elec_field[i] += 0.;
                } else if i % 2 == 0 {
                    //x coord
                    let deriv_coeff_x = coefficients[[u, v]] * w_u / (w_u.powi(2) + w_v.powi(2));

                    elec_field[i] += deriv_coeff_x * (w_u * x).sin() * (w_v * y).cos();
                } else if i % 2 == 1 {
                    //y coord
                    let deriv_coeff_y = coefficients[[u, v]] * w_v / (w_u.powi(2) + w_v.powi(2));

                    elec_field[i] += deriv_coeff_y * (w_u * x).cos() * (w_v * y).sin();
                }
            }
        }
    }
    elec_field
}

// not yet implemented fast version of DCT. the code below is sample code from the library and not related to the paper
// transformed_matrix = dct(density)
// then call the inverse

//fft is a discrete fourier transform
//you can think of the fft as having a sine component and a cosine component
//if you only need to calculate the cosine component, you use the discrete cosine transform, DCT
//
/*
    let (nx, ny) = (16, 16);
    let mut data = Array2::<f64>::zeros((nx, ny));
    let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f64;
    }
    let mut fft_handler = R2cFftHandler::<f64>::new(nx);
    ndfft_r2c(&data.view(), &mut vhat.view_mut(), &mut fft_handler, 0);
    dbg!(&vhat);
    dbg!(&data);
    println!("spacer");
    for axis in vhat.axes() {
      dbg!(axis);
    }
*/
