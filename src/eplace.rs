use crate::dct;
use crate::density;
use crate::util::B;
use crate::util::BIN_W;
use crate::util::K;
use crate::util::TARGET_DENSITY;
use crate::wirelength;
use crate::wl_grad;
use ndarray::{Array1, Array2, Axis};

pub struct NLparams {
    pub placement: Array2<f64>,
    pub ref_placement: Array2<f64>,
    pub a: f64,
    pub alpha: f64,
    pub f_k: f64,
    pub grad_f_k: Array1<f64>,
}

//a struct to hold the electric fields  for each of the cells. Storing x and y separately,
// and they're calculated slightly differently! each of x_fields and y_fields has length
//equal to the number of cells / logic elements
pub struct CellElectricFields {
    //electric field in x direction
    x_fields: Array1<f64>,
    y_fields: Array1<f64>,
}

///The core eplace algorithm, called during each iteration .
pub fn eplace(prev: NLparams, m: usize) -> NLparams {
    //reshape the gradient to be a number of cells x 2 matrix
    let reshaped_grad = &prev
        .grad_f_k
        .to_shape((prev.grad_f_k.len_of(Axis(0)) / 2, 2))
        .expect(" wasn't able to reshape the f_k gradient");

    //each placement is based on the so-called "reference placement " - a list of cell locations
    // calculated from the actual placement from the last iteratin.
    let placement = &prev.ref_placement - prev.alpha * reshaped_grad;

    //how high above the target density are we? This variable (which the paper refers to as tau) is used in a few places
    let density_overflow = calc_density_overflow(&prev.ref_placement, m);

    //gamma is a parameter to tune our wirelength estimator. it starts high and ultimately depends on density overflow
    let gamma = calc_gamma(density_overflow);

    //wirelength gradient, the gradient of the wirelength estimator from equation 6 on page 5
    // partial derivative with respect to x and y the equation for each cell!
    let wl_gradient = wl_grad::calc_wl_grad(&prev.ref_placement, gamma);

    //electric fields of each cell. Calculating them involves a 2D DCT on the density of cells in each bin, multiplying
    //each of the coefficients (the a_u_vs referenced in equation 21 on page 12, and then calculating) by formulas based
    // on equation 24 on page 13, and then taknig the inverse DCST or DSCT
    let fields = calc_cell_fields(&prev.ref_placement, m);

    //lambda is the lagrange multiplier (I believe!), used to link the value we're optimizing for (the wirelength) and
    //the constraint / penalty function (the electric potential we're calculating )
    let lambda: f64 = calc_lambda(&prev.ref_placement, &fields, gamma);

    //gradient of the objective function - grad (wirelength estimator) + lambda * grad( penalty function )
    let grad_f_k = calc_grad_f_k(&wl_gradient, lambda, fields);

    
    //a is an optimization parameter used in the NL solver, which is algorithm 2 on page 18. It'll be used to calculate
    //the reference placement
    let a: f64 = (1. + (4. * prev.a * prev.a + 1.).sqrt()) / 2.;
    //reference placement is what we'll be actually feeding to most of our algorithm in the next iteration
    let ref_placement = &placement + (prev.a - 1.) / a * (&placement - &prev.placement);
  
    NLparams {
        placement: placement.clone(),
        a,
        alpha: inverse_lipschitz_constant(
            &ref_placement,
            &prev.placement,
            &grad_f_k,
            prev.grad_f_k,
        ),
        //f_k is our objective function. Never used directly, but we're calculating it so we can see if it's getting minimized correctly
        f_k: calc_f_k(&ref_placement, density_overflow, lambda, m),
        grad_f_k,
        ref_placement,
    }
}

///There are a variety of starter values and special cases that are fed to the first iteration of eplace and the NLSolver - they're collected here
pub fn calc_initial_params(cell_centers: &Array2<f64>, m: usize) -> NLparams {
    let initial_density_overflow = 1.0; //we can calculate this but will assume 1.0 for now. something to play with!
    let gamma_0: f64 = calc_gamma(initial_density_overflow);
    let wl_gradient_0 = wl_grad::calc_wl_grad(cell_centers, gamma_0);
    let fields = calc_cell_fields(cell_centers, m);
    let lambda_0 = calc_lambda(cell_centers, &fields, gamma_0);

    NLparams {
        placement: cell_centers.clone(), // before the first loop, placement and
        //reference placement are the same , but in general they're slightly different
        ref_placement: cell_centers.clone(),
        a: 1., //prescribed starting value, later iterations will use this to calculate the next iteration's a.
        alpha: 0.044 * BIN_W, //similar
        f_k: calc_f_k(cell_centers, initial_density_overflow, lambda_0, m),
        grad_f_k: calc_grad_f_k(&wl_gradient_0, lambda_0, fields),
    }
}
pub fn calc_cell_fields(placement: &Array2<f64>, m: usize) -> CellElectricFields {
    let density = density::calc_density(placement, m); // mxm density calculation, overlaps of

    let coeffs = dct::calc_coeffs(&density, m);

    let elec_field_x = dct::elec_field_x(&coeffs, m);
    let elec_field_y = dct::elec_field_y(&coeffs, m);

    let cell_fields_x =
        placement.map_axis(Axis(1), |x| dct::apply_bins_to_cell(&x, &elec_field_x, m));
    let cell_fields_y =
        placement.map_axis(Axis(1), |x| dct::apply_bins_to_cell(&x, &elec_field_y, m));

    CellElectricFields {
        x_fields: cell_fields_x,
        y_fields: cell_fields_y,
    }
}
//eq 29

///the preconditioner depends on the detaills of vertex degree and net subsets incident to an object i
/// as well as vertex degrees. could a cell be connected to multiple nets?
/// a vertex is incident to a net if the vertex is one of the endpoints of that edge
/// the degree of a vertex is how many edges are incident to that vertex.
fn precondition(grad_f_k: Array1<f64>, lambda: f64) -> Array1<f64> {
    // interpretation one:  |E_i| is the number of nets a cell is connected to
    // interpretation two: |E_i| is the number of vertices in the single net that the cell's a part of

    //we will assume that |E_i| is 1 for every cell
    // lambda = lambda_0
    // q_i = 2.25

    let precon: f64 = 2. + lambda * 2.25;

    precon * grad_f_k // hopefully, f64 * Array1<f64> is implemented, we work around if not
}

///This one's a doozy! Ultimately, the paper goes through a lot of trouble to prove that the inverse of something called the lipschitz constant
/// is acceptable for our steplength. The lipschitz constant is itself approximated based on equation 29 on page 19. Here we take the reciprical
/// of that equation - that's our alpha!
fn inverse_lipschitz_constant(
    ref_placement: &Array2<f64>,
    prev_ref_placement: &Array2<f64>,
    grad_f_k: &Array1<f64>,
    prev_grad_f_k: Array1<f64>,
) -> f64 {

    let numerator: f64 = (ref_placement - prev_ref_placement)
        .map(|x| x * x)
        .sum()
        .sqrt();

    let denominator: f64 = (grad_f_k - prev_grad_f_k).map(|x| x * x).sum().sqrt();

    let calcaluted_alpha = numerator / denominator;
    0.044
}

//equation 37, page 22
fn calc_density_overflow(placement: &Array2<f64>, m: usize) -> f64 {
    let grid_area = 1.;
    let bin_density = density::calc_density(placement, m);
    let numerator = bin_density
        .map(|bin_density| (bin_density - TARGET_DENSITY).max(0.) * grid_area)
        .sum();

    //denominator is the area of each cell summed up,
    let denominator = 2.25 * (placement.len_of(Axis(0)) as f64);

    let tau = numerator / denominator;
//    dbg!(tau);
    tau
}
pub fn calc_f_k(placement: &Array2<f64>, density_overflow: f64, lambda: f64, m: usize) -> f64 {
    let wl: f64 = wirelength::wl(placement, calc_gamma(density_overflow));

    let density = density::calc_density(placement, m);
    let coeffs = dct::calc_coeffs(&density, m);

    let total_pot: f64 = dct::total_potential(&coeffs, placement, m);
    wl + lambda * total_pot
}

//f should have the electric fields for each cell in the x and y direction
pub fn calc_lambda(placement: &Array2<f64>, fields: &CellElectricFields, gamma: f64) -> f64 {
    let wl_gradient = wl_grad::calc_wl_grad(placement, gamma);
    let numerator = wl_gradient.map(|x| x.abs()).sum();
    let denominator_x = fields.x_fields.map(|x| 2.25 * x.abs()).sum();
    let denominator_y = fields.y_fields.map(|y| 2.25 * y.abs()).sum();

    numerator / (denominator_x + denominator_y)
    
}

///gamma is used in the wirelength estimator (equation 6 on page 5). It's actually calculated in equation 38 on page 23
fn calc_gamma(density_overflow: f64) -> f64 {
    8. * BIN_W * (10_f64).powf(K * density_overflow + B) // 8.0 * bin_width * 10^(k *tau + b), where k = 20/9 and b = -11/9.
                                                         // as tau gets smaller, this will reduce gamma, which will approach to
                                                         // 8 * bin_width * 10^(-11/9). In our case, this'd put gamma at a minimum of approximately
                                                         // 0.7 or 0.8. We can tweak these variables if we'd like
}

///the gradient of the objective function is the gradient of equation 9 on page 6. This'll ultimately be multiplied by alpha and added to our
///reference placement to get our next placement.
pub fn calc_grad_f_k(
    wl_gradient: &Array1<f64>,
    lambda: f64,
    fields: CellElectricFields,
) -> Array1<f64> {
    //we also implement the preconditioner specified on page 19 and 20 - it helps with numerical convergence. The calculated gradient is fed to the
    //preconditioner.
    precondition(wl_gradient + lambda * calc_penalty_grad(fields), lambda)
}

///The penalty gradient is the electric field at each point multiplied by its charge. In our case, the charge of each
/// cell is a fixed 1.5 * 1.5 = 2.25
fn calc_penalty_grad(fields: CellElectricFields) -> Array1<f64> {
    let mut grad_penalty_k = Array1::<f64>::zeros(2 * fields.x_fields.len());

    for i in 0..fields.x_fields.len() {
        grad_penalty_k[i] = 2.25 * fields.x_fields[i];
        grad_penalty_k[i + 1] = 2.25 * fields.y_fields[i];
    }
    grad_penalty_k
}
