use crate::eplace::NLparams;
use ndarray::Array1;

pub fn elfplace( ) {
    let wl = todo!(); 
    let f_k: f64 = calc_f_k(wl);

    //gradient f k is calcluated in equation 13, and then a preconditioner will be applied to it before it's fed 
    //to the solver 
    let grad_f_k = todo!(); 
}

///the objective function we're looking to find is equation 9 on page 3 of elfplace
/// It's a modification of the objective function for eplace and works in a similar 
/// way. 
fn calc_f_k(wl: f64  ) -> f64 {
    //iterating over resource types 
    let resources : Array1<NLparams> = todo!(); 

    let prev_lambda : Array1<f64> = todo!(); 
    let normalized_lambda_subgradient = calc_normalized_subgrad(&prev_lambda); 
    let lambda: Array1<f64>  = calc_lambda(&prev_lambda);
    

        //we'll probably need to add a new adjusted area in order to calculate potential
        //the area of each unit is adjusted for routability, pin density, 
        //and clustering compatibility. We need to do that, and *that* area will be used 
        //in the potential function. 
    let potential: f64 = todo!(); 

    //using s for the resource index cause that's what the paper does
    let error_term: f64 = resources.iter().enumerate().map(|(s,resource)| {
        lambda[s] * (potential + (calc_c_s()/2_f64) * potential*potential)  }
      ).sum();
      wl + error_term

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
 fn calc_lambda (prev_lambda: &Array1<f64>) -> Array1<f64> { 

    let prev_step_size:f64 = todo!(); 
    let normalized_subgrad = calc_normalized_subgrad(&prev_lambda);

    prev_lambda + prev_step_size * prev_lambda / normalized_subgrad.map(|&x| x.powi(2)).sum().sqrt() 

 }

 fn calc_normalized_subgrad(prev_lambda: &Array1<f64>) -> Array1<f64>{
    todo!()
 }
 fn calc_c_s() -> f64 {
    todo!(); 
 }