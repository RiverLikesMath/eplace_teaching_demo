use ndarray::{array, Array1, Array2, ArrayView1, Axis};

//use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};
use std::f64::consts::E;
use std::f64::consts::PI;

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

    let mut density = calc_density(&cell_centers, m); // mxm density calculation, overlaps of
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

    let wl_grad = calc_wl_grad(&cell_centers);
    dbg!(&wl_grad);

    //we'll also need the electric field, which requires some cosine/sine transforms of the density matrix


    let lambda_0_upper = wl_grad.map(|partialdiv| partialdiv.abs()).sum(); //from eq 35
    let charges = array![2.25, 2.25, 2.25, 2.25];

    let coefficients    = dct_coeff(&density, m);
    //inverse cosine transform of coefficients will get you the density
    //the plan -> multiply coefficients by the relevant factor from equation 23, inverse 2D cosine transform, that gets you potential for each bin 
    //multiply the coefficients by relevant factor from equation 24, inverse sine cosine transform, that gets you electric field in the X direction 
    //multiply the coefficients by the relevant Y factors from equation 24, inverse cosine sine tranform, that gets you electric field in the Y direction. 
    
    //I've already confirmed that the transforms as described in the paper work beautifully for density, even with only 
    //16x16 bins, at least as far as the toy example here goes. 
    let potential    = eplace_potential(&coefficients, m); //not yet implemented 
    let elec_field_x = eplace_elec_x(&coefficients, m); //not yet implemented 
    let elec_field_y = eplace_elec_y(&coefficients, m); //not yet implemented
    //each of these functions will generate a potential, electric field x, or electric field y for *each* bin! 
    //those will then be applied to given cells  
}

/// only pass floats for x and y that are integers
fn add_density(density: &mut Array2<f64>, x: usize, y: usize, added_density: f64) {
    density[[x, y]] += added_density;
}

///calculates the density given the current location of each centr, by calculating overlap of each cell w/ bins
///there's probably a shorter clearer way to do this
fn calc_density(cell_centers: &Array2<f64>, m: usize) -> Array2<f64> {
    let mut density = Array2::<f64>::zeros((m, m));
    //density[[0, 7]] = 29.0;
    //density[[4, 12]] = 3.0;
    //dbg!(&density);
    dbg!(cell_centers);
    for cell in 0..cell_centers.len_of(Axis(0)) {
        //calculate which bins its in
        //calculate the overlap with each bin
        //update the density array/matrix accordingly

        let left_edge = cell_centers[[cell, 0]] - 0.75;
        let right_edge = cell_centers[[cell, 0]] + 0.75;
        let upper_edge = cell_centers[[cell, 1]] + 0.75;
        let lower_edge = cell_centers[[cell, 1]] - 0.75;

        let left_edge_bin = left_edge.floor();
        let right_edge_bin = right_edge.floor();
        let upper_edge_bin = upper_edge.floor();
        let lower_edge_bin = lower_edge.floor();

        let right_width = right_edge - right_edge_bin;
        let left_width = (left_edge_bin + bin_w) - left_edge;

        let upper_height = upper_edge - upper_edge_bin;
        let lower_height = (lower_edge_bin + bin_w) - lower_edge;

        //update the upper right corner
        add_density(
            &mut density,
            right_edge_bin as usize,
            upper_edge_bin as usize,
            right_width * upper_height,
        );
        //update the lower right corner
        add_density(
            &mut density,
            right_edge_bin as usize,
            lower_edge_bin as usize,
            right_width * lower_height,
        );
        //update the lower left corner
        add_density(
            &mut density,
            left_edge_bin as usize,
            lower_edge_bin as usize,
            left_width * lower_height,
        );
        //update the upper left corner
        add_density(
            &mut density,
            left_edge_bin as usize,
            upper_edge_bin as usize,
            left_width * upper_height,
        );

        for y in ((lower_edge_bin + 1.) as usize)..(upper_edge_bin as usize) {
            //now for the left edges that aren't corners
            add_density(&mut density, left_edge_bin as usize, y, left_width); //height of a bin is 1
                                                                              //right edges
            add_density(&mut density, right_edge_bin as usize, y, right_width);
        }

        for x in ((left_edge_bin + 1.) as usize)..(right_edge_bin as usize) {
            //now for the upper edges that aren't corners
            add_density(&mut density, x, upper_edge_bin as usize, upper_height); //width of a bin is 1
                                                                                 //lower edges
            add_density(&mut density, x, lower_edge_bin as usize, lower_height);
        }

        //add density of completely filled bins
        for x in ((left_edge_bin + 1.) as usize)..(right_edge_bin as usize) {
            for y in ((lower_edge_bin + 1.) as usize)..(upper_edge_bin as usize) {
                add_density(&mut density, x, y, 1.);
            }
        }
    }
    density
}

fn calc_wl_wa_partial(x_is: &ArrayView1<f64>, x_i: f64, gamma: f64) -> f64 {
    //calculating the derivative of the weighted average wirelength function with respect to a give x_i or y_i
    // using quotient and product rules
    //they are broken up into a left and right halves, which we'll calculate separately

    //while we covered how to take the partial derivative manually, we're using a computer algebra system / symbolic
    //solver in order to make sure we haven't made any mistakes while doing so.
    //because the x and y wirelength sums are independent, a partial with respect to any x_i won't need to calculate
    //y_i and vice versa. We'll use x_i here, but the formula (and the derivative) is the same for both.

    //PARTIAL DERIVATIVE OF LEFT TERM OF THE WIRELENGTH ESTIMATION FUNCTION
    //first, let's get an array that's all of the e^(x_i/0.2), which we can the sum up using a hopefully included
    //sum function! First, we'll map over x_i to get e^(x_i) for each element of x_is
    let sum_x_e_xis = x_is.map(|x_i| *x_i * E.powf(*x_i / gamma)).sum(); //sum  x*e^(x_i/gamma)
                                                                         //ok, let's try
                                                                         //we'll also need the sum of just e^(x_i)

    let sum_e_xis = x_is.map(|x_i| E.powf(*x_i / gamma)).sum(); // //sum  e^(x_i/gamma) or sum e^(y_i/gamma)

    //we'll also need e^(x_i/gamma)  for the x_i under consedration
    let e_xi = E.powf(x_i / gamma);

    //next, we calculate the left term of the partial derivative of the left (so many lefts)
    // the partial derivative of the left is  derived from here: https://www.wolframalpha.com/input?i2d=true&i=partial+derivative+with+respect+to+x+of++%5C%2840%29%5C%2840%29+Divide%5B%5C%2840%29x*Power%5Be%2C%5C%2840%29Divide%5Bx%2Cg%5D%5C%2841%29%5D+%2B+y*Power%5Be%2C%5C%2840%29Divide%5By%2Cg%5D%5C%2841%29%5D+%2Bz*Power%5Be%2C%5C%2840%29Divide%5Bz%2Cg%5D%5C%2841%29%5D%5C%2841%29%2C%5C%2840%29Power%5Be%2C%5C%2840%29Divide%5Bx%2Cg%5D%5C%2841%29%5D+%2B+Power%5Be%2C%5C%2840%29Divide%5By%2Cg%5D%5C%2841%29%5D+%2BPower%5Be%2C%5C%2840%29Divide%5Bz%2Cg%5D%5C%2841%29%5D%5C%2841%29%5D+%5C%2841%29%5C%2841%29
    let leftleft = ((x_i * e_xi / gamma) + e_xi) / sum_e_xis;

    //and the right
    let leftright = (e_xi * sum_x_e_xis) / (gamma * sum_e_xis * sum_e_xis);
    //and finally, the partial derivative of the first term in the weighted average wirelength

    //THIS ONE'S IMPORTANT
    let partial_left = leftleft - leftright;

    //phew! So, there's probably a nice way to reuse everything we did, but we'll go ahead and do the same thing for the
    //right hand term, the generation of which in wolfram alpha I'll leave as an exercise to the reader, by which I mean
    //I'll show the reader during lesson

    let sum_x_e_mxis = x_is.map(|x| (*x) * E.powf((*x) * -1. / gamma)).sum(); //sum x_i * e^(-x_i/gamma)
                                                                              //similar code to before
    let sum_e_mxis = x_is.map(|x| E.powf((*x as f64) * (-1.0) / gamma)).sum(); //sum e^(-x_i/gamma)

    //and again. maybe should rewrite!
    let e_mxi = E.powf((-1.0) * x_i / gamma); //e^(-x_i/gamma)

    let rightleft = (e_mxi - x_i * e_mxi / gamma) / sum_e_mxis;
    let rightright = (e_mxi * sum_x_e_mxis) / (gamma * sum_e_mxis.powi(2));

    //THIS ONE ALSO IMPORTANT
    let partial_right = rightleft + rightright;

    partial_left - partial_right //and this is the partial derivative of the wirelength estimation function!
}
fn calc_wl_grad(cell_centers: &Array2<f64>) -> Array1<f64> {
    //this is one of our most important calculations, it's used over and over and over again.
    //cell centers is just a list of x and y coordinates, right?

    //So, the gradient of Wirelength is seen in equation  16, we take the partial derivative with respect to
    //the x and y of each logic element. This i'll demonstrate in class and we can type up later in
    //markdown / latex / whatever rust goodness.

    //so the gradient is basically a
    let mut partials = Array1::<f64>::zeros(cell_centers.len_of(Axis(0)) * 2);
    let all_x_i = cell_centers.column(0);

    let all_y_i = cell_centers.column(1);

    for cell in 0..cell_centers.len_of(Axis(0)) {
        let x_i = cell_centers[[cell, 0]];
        let y_i = cell_centers[[cell, 1]];

        let gamma = 0.2;
        partials[cell * 2] = calc_wl_wa_partial(&all_x_i, x_i as f64, gamma);
        partials[cell * 2 + 1] = calc_wl_wa_partial(&all_y_i, y_i as f64, gamma);
    }
    partials
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
                if (u == 0 && v == 0 )
                //formula as written has a divide by zero here 
                {
                    elec_field[i]+=0.; 
                }
                
                else if i % 2 == 0 {
                    //x coord
                    let deriv_coeff_x = coefficients[[u, v]] * w_u / (w_u.powi(2) + w_v.powi(2));
                    
                    elec_field[i] += deriv_coeff_x * (w_u * x).sin() * (w_v * y).cos();
                }
                 else if i % 2 == 1 {
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
