

pub fn nl_solver(/*etc. */){
    //initially, alpha_0 is given to us as a flat 0.44 * BIN_W 
    //u_k is our placement, and v_k is what's known as a "referenc placement". As we dig 
    //into the algorithm and properly implement it we'll have to keep our ducks in a row 
    // on it. 
    let alpha :f64 = inverse_lipschitz_constant(grad_f_k, grad_f_k_1, ref_placement, previous_ref_placement);    
    
    //alpha * grad_f_k is scalar multiplication of an arrat
    //line 3 of our nesterov solver - the beating heart of the algorithm! 
    //very important line of code 
    new_placement = ref_placement - alpha * grad_f_k; 

    //a_0 = 1 

    //line 4 of nesterov solver 
    let a_new = (1.+ (4_a_k *4_a_k +1).sqrt() ) /2.;

    //line 5 of nesterov solver , calclutating v_kplus1
    ref_placement_new : Array1<f64> = new_placement + (a_k -1) * (new_placement - previous_placement)/ (a_new); 

    //line 6 of nesterov solver 
    new_placement

}

//eq 29
fn inverse_lipschitz_constant(/*etc. */) -> f64 { 
    // || placement - previous_placement||; 
    
    // || (3,5) || = (3.pow(2) + 5.pow(2)).sqrt(); 
    // sqrt( 3^2 + 5^2 )

    let numerator :f64 =  (placement - previous_placement).map( |x| x*x ).sum().sqrt();
    let denominator: f64 = (grad_f_k - grad_f_kminus1).map( |x| x*x).sum().sqrt();

    numerator/denominator 

}