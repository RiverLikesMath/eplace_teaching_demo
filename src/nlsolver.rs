use ndarray::{Array1, Array2};

use crate::density;

pub struct NLparams {
    pub placement: Array2<f64>,
    pub a: f64,
    pub alpha: f64,
    pub f_k: f64,
    pub grad_f_k: Array1<f64>,
}
pub fn nl_solver(prev: NLparams, m: usize) -> NLparams {
    let new: NLparams;
    //initially, alpha_0 is given to us as a flat 0.44 * BIN_W
    //u_k is our placement, and v_k is what's known as a "referenc placement". As we dig
    //into the algorithm and properly implement it we'll have to keep our ducks in a row
    // on it.

    //alpha * grad_f_k is scalar multiplication of an arrat
    //line 3 of our nesterov solver - the beating heart of the algorithm!
    //very important line of code
    new.placement = prev.placement - prev.alpha * prev.grad_f_k;

    //a_0 = 1

    //line 4 of nesterov solver on page 18, a is an optimization parameter, so I'm not entirely sure why this
    //is the way it is. we start with the section in the square root because rust math is a bit confusing for em
    let sqrts = (4. * prev.a * prev.a + 1.).sqrt(); //sqrt ( 4 a_k^2 +1  {}
                                                    //and here's a
    new.a = (1. + sqrts) / 2.;

    let density = density::calc_density(&new.placement, m);
    let coeffs = dct::calc_coeffs(&density, m);
    //let elec_field_x =
    //new.grad_f_k =
    let new_alpha: f64 =
        inverse_lipschitz_constant(grad_f_k, grad_f_k_1, ref_placement, previous_ref_placement);

    //line 5 of nesterov solver , calclutating v_kplus1
    let ref_placement_new: Array1<f64> =
        new_placement + (a_k - 1) * (new_placement - previous_placement) / (a_new);

    //line 6 of nesterov solver. ultimately we may want to pass in and return a struct with all the parameters and stuff
    // for each iteration of eplace
    new_placement
}

//eq 29

///the preconditioner depends on the detaills of vertex degree and net subsets incident to an object i
/// as well as vertex degrees. could a cell be connected to multiple nets?
/// a vertex is incident to a net if the vertex is one of the endpoints of that edge
/// the degree of a vertex is how many edges are incident to that vertex.
pub fn precondition(grad_f_k: Array1<f64>, lambda: f64) -> Array1<f64> {
    // interpretation one:  |E_i| is the number of nets a cell is connected to
    // interpretation two: |E_i| is the number of vertices in the single net that the cell's a part of

    //we will assume that |E_i| is 1 for every cell
    // lambda = lambda_0
    // q_i = 2.25

    let precon: f64 = 2. + lambda * 2.25;

    precon * grad_f_k // hopefully, f64 * Array1<f64> is implemented, we work around if not
}

fn inverse_lipschitz_constant(/*etc. */) -> f64 {
    // || placement - previous_placement||;

    // || (3,5) || = (3.pow(2) + 5.pow(2)).sqrt();
    // sqrt( 3^2 + 5^2 )

    let numerator: f64 = (placement - previous_placement).map(|x| x * x).sum().sqrt();
    let denominator: f64 = (grad_f_k - grad_f_kminus1).map(|x| x * x).sum().sqrt();

    numerator / denominator
}
