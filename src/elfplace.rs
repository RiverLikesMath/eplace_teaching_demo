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

    //this combined potential willbe an f64, the length of the vector containing the potentials for each resource 
        //we'll probably need to add a new adjusted area in order to calculate potential
        //the area of each unit is adjusted for routability, pin density, 
        //and clustering compatibility. We need to do that, and *that* area will be used 
        //in the potential function. 
    let combined_potential = todo!(); 
    let initial_combined_potential = todo!(); 
    //beta is used primarily in equation 12, but also shows up when calculating lambda
    let beta: f64 = todo!(); 

    //lambda section
    let prev_lambda : Array1<f64> = todo!(); 
    let prev_step_size: f64 = todo!();  
    let lambda: Array1<f64>  = calc_next_lambda(&prev_lambda, prev_step_size , beta, combined_potential, initial_combined_potential);

    //using s for the resource index cause that's what the paper does
    let error_term: f64 = resources.iter().enumerate().map(|(s,resource)| {
        let potential: f64 = todo!(); 
        lambda[s] * (potential + (calc_c_s()/2_f64) * potential*potential)  }
      ).sum();
      wl + error_term

 }  


 fn metric_length (vector : &Array1<f64>) -> f64 {
    vector.map( |&x| x.powi(2)).sum().sqrt()
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
 fn calc_next_lambda (prev_lambda: &Array1<f64>, prev_step_size: f64, beta: f64,  potentials : Array1<f64> , initial_potentials: Array1<f64> ) -> Array1<f64> { 
    let normed_potential: f64 = metric_length(&potentials)/metric_length(&initial_potentials); //length combined potentials over the length of initial potentials
    
    let step_size :f64 = calc_step_size(false, prev_step_size, beta, normed_potential); 
    let normalized_subgrad = calc_normalized_subgrad(&prev_lambda, potentials, initial_potentials );
    prev_lambda + step_size * prev_lambda / metric_length(&normalized_subgrad)

 }

/// Equation 22, page 4. The 
 fn calc_step_size (start: bool, prev_step_size : f64, beta: f64, normed_potential: f64) -> f64 { 
     let alpha_h = 1.06;
     let alpha_l = 1.05; 
     if start{ 
        alpha_h - 1_f64
     }
     else  {
        let big_fraction_mess =   (beta * normed_potential + 1_f64 ).ln()  / ( 1_f64 + (beta * normed_potential+ 1_f64).ln()  );
         //return the big fraction mess based term when k > 0
         prev_step_size *  ( big_fraction_mess * (alpha_h - alpha_l) )+ alpha_l
     }
 }


 ///equation 20, page 4. The normalized subgradient is so called because it uses the normalized potential instead of raw potential - this is so the different
 /// resources more accurately reflect their level of density violation 
 fn calc_normalized_subgrad(prev_lambda: &Array1<f64>, potentials : Array1<f64>, initial_potentials: Array1<f64> ) -> Array1<f64>{
    todo!()
 }
 fn calc_c_s() -> f64 {
    todo!(); 
 }