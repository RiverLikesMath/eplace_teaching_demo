use crate::eplace;
use crate::wirelength;
use ndarray::{Array1, Array2};

///length of a vector using the standard euclidean metric (measuring the length of a line in cartesian coordinates,
/// for example).
fn metric_length(vector: &Array1<f64>) -> f64 {
    vector.map(|&x| x.powi(2)).sum().sqrt()
}

///the big 'un!
pub fn elfplace(prev_resources: Array1<eplace::NLparams>, m:usize ) {
    //we will calculate wirelength for each resource and sum them up

    //f_k is the objective function, equation 9
    let f_k: f64 = calc_f_k(prev_resources, m);

    //gradient f k is calcluated in equation 13, and then a preconditioner will be applied to it before it's fed
    //to the solver
    let grad_f_k = todo!();
}

///the objective function we're looking to find is equation 9 on page 3 of elfplace
/// It's a modification of the objective function for eplace and works in a similar
/// way - it's just that everything is done for multiple resources and we're optimizing
/// on a vector of resources instead of just the single placement before
fn calc_f_k(prev_resources: Array1<eplace::NLparams>, m: usize) -> f64 {
    //wirelength is found similarly to eplace, but for multiple resource types

    //gamma is calculated using the density overflow of each resource - but I haven't yet checked to
    //see if the overflow calculations are the same between eplace and elfplace. It looks like they aren't so we'll
    //implement elfplace's overflow separately . Each resource will have a separate gamma for its wirelength estimator
    let resource_capacities : Array1 < Array2<f64> > = todo!(); 
    let gammas: Array1<f64> = prev_resources.iter().
                                flat_map( |res| resource_capacities. map ( 
                                   move  |capacity|
                                         eplace::calc_gamma(calc_overflow(res, capacity, m))
                                   )
                                ).collect();

    let wl: f64 = prev_resources
        .iter()
        .enumerate()
        .map(|(i, res)| wirelength::wl(&res.ref_placement, gammas[i]))
        .sum(); //probably needs a flat map!

    //the potentials will be an array of f64s
    //we'll probably need to add a new adjusted area in order to calculate potential
    //the area of each unit is adjusted for routability, pin density,
    //and clustering compatibility. We need to do that, and *that* area will be used
    //in the potential function.
    let potentials: Array1<f64> = todo!();
    //total potential of each resource in the starting placement
    let initial_potentials: Array1<f64> = todo!();

    //beta is used primarily in equation 12, but also shows up when calculating lambda
    let beta: f64 = todo!();

    //lambda section
    let prev_lambda: Array1<f64> = todo!();
    let prev_step_size: f64 = todo!();
    let lambda: Array1<f64> = calc_next_lambda(
        &prev_lambda,
        prev_step_size,
        beta,
        potentials,
        initial_potentials,
    );

    //using s for the resource index cause that's what the paper does
    let error_term: f64 = prev_resources
        .iter()
        .enumerate()
        .map(|(s, resource)| {
            let potential: f64 = todo!();
            lambda[s] * (potential + (calc_c_s() / 2_f64) * potential * potential)
        })
        .sum();
    wl + error_term
}

///the overflow for elfplace uses a related but slightly different formula, equation 7 on page 2
fn calc_overflow(prev_resource: &eplace::NLparams, resource_capacities: &ndarray::Array2<f64>,  m: usize ) -> f64 {
    let numerator = 0_f64;
    let denominator = 0_f64; 
    
    let bin_resource_areas: Array2<f64> = todo!(); 

    for u in 0..m { 
       for v in 0..m { 
          let bin_resource_capacity: f64 = resource_capacities[[u,v]];
          let bin_resource_area: f64  = bin_resource_areas[[u,v]]; 
          numerator +=  (bin_resource_area - bin_resource_capacity ).max(0.0_f64); 
          denominator += bin_resource_area;
       }
    }
    
    numerator/denominator
}
///can this be ripped directly from eplace or is it modified?
/// It's looking like it's a fairly complicated calculation,
/// with later adjustments once we do the area adjustments for
/// routing and the like. For now, we'll start with equation  21
/// on page 5. This equation relies on the equations preceding it,
/// which we'll dig into. It's used for updating lambda given
/// a previous lambda. That is, it's the general case.
/// The equation defines a vector of lambdas, one for each resource type
/// This is why equations 19 and 20 have those "..." and a transpose
/// The normalized subgradient is calculated outside the function in equation 20,
/// which we'll be implementing soon!
fn calc_next_lambda(
    prev_lambda: &Array1<f64>,
    prev_step_size: f64,
    beta: f64,
    potentials: Array1<f64>,
    initial_potentials: Array1<f64>,
) -> Array1<f64> {
    let normed_potentials: Array1<f64> = &potentials / &initial_potentials;

    // old was let normed_potential = metric_length( potentials) / metric_length (initial_potentials);
    //step size is equation 22 on page 4 - how much we should multiply the subgradient by.
    let step_size: f64 = calc_step_size(false, prev_step_size, beta, &normed_potentials);

    let normalized_subgrad = calc_normalized_subgrad(normed_potentials, beta);
    let subgrad_length = metric_length(&normalized_subgrad);
    prev_lambda + step_size * normalized_subgrad / subgrad_length
}

/// Equation 22, page 4. This is the step size to see how far in the direction of the subgradient of lamba we should
/// move when calculating our new lambda. It may be used elsewhere as well! this is t^(k) in equations 21 and 22   
fn calc_step_size(
    start: bool,
    prev_step_size: f64,
    beta: f64,
    normed_potentials: &Array1<f64>,
) -> f64 {
    let alpha_h = 1.06;
    let alpha_l = 1.05;

    let total_normed_potential = metric_length(normed_potentials);

    if start {
        alpha_h - 1_f64
    } else {
        //unsure what base of logarithm is, so assuming ln
        let log_term = (beta * total_normed_potential + 1_f64).ln();
        //Now it's small fraction mess! :D
        let big_fraction_mess = log_term / (1_f64 + log_term);
        //return the big fraction mess based term when k > 0
        prev_step_size * (big_fraction_mess * (alpha_h - alpha_l)) + alpha_l
    }
}

///equation 20, page 4. The normalized subgradient is so called because it uses the normalized potential instead of raw potential - this is so the different
/// resources more accurately reflect their level of density violation
fn calc_normalized_subgrad(normed_potentials: Array1<f64>, beta: f64) -> Array1<f64> {
    normed_potentials.map(|npot| npot + beta * npot.powi(2) / 2_f64)
}
fn calc_c_s() -> f64 {
    todo!();
}
