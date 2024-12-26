use ndarray::{Array2, ArrayView1};

///weighted average wirelength estimator, equation 6 on page 5 of eplace
pub fn wl(cell_centers: &Array2<f64>, gamma: f64) -> f64 {
    //gamma =0.2
    //wl_x = ul/ll - ur/lr -> ul,ll,ur,lr :f64, same for wl_y

    let all_x_i = cell_centers.column(0);

    let all_y_i = cell_centers.column(1);

    let wl = wl_component(all_x_i, gamma) + wl_component(all_y_i, gamma);
    //println!("wirelength: {wl}");
    wl
}

fn wl_component(all_comp: ArrayView1<f64>, gamma: f64) -> f64 {
    //all_comp being every instance of either x or y components of our cell center coordinates

    //numerator of the left term
    let ul = all_comp.map(|x| x * (x / gamma).exp()).sum(); //sum x * e^(x/gamma)

    //denominator of the left term
    let ll = all_comp.map(|x| (x / gamma).exp()).sum(); //sum e^(x/gamma)

    //now the right term
    let ur = all_comp.map(|x| x * (-x / gamma).exp()).sum(); //sum x*e^(-x/gamma)
    let lr = all_comp.map(|x| (-x / gamma).exp()).sum(); //sum e^(-x/gamma)

    //the final expression is just the various components divided and subtracted
    ul / ll - ur / lr
}
