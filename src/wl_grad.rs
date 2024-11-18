use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::f64::consts::E;

pub fn calc_wl_grad(cell_centers: &Array2<f64>) -> Array1<f64> {
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
