use ndarray::Array2; 

pub fn wl(cell_centers: Array2<f64>, gamma: f64) -> f64 {
    //gamma =0.2 
    //wl_x = ul/ll - ur/lr -> ul,ll,ur,lr :f64
    
    let all_x_i = cell_centers.column(0);

    let ul_x = all_x_i.map(|x|  x *  (x / gamma).exp()  ).sum(); //sum x * e^(x/gamma)
    let ll_x = all_x_i.map(|x| (x/gamma).exp() ).sum();  //
    
    let ur_x = all_x_i.map( |x| x * (-x/gamma).exp()).sum(); 
    let lr_x = all_x_i.map( |x| (-x/gamma).exp() ).sum(); 

    let wl_x = ul_x / ll_x - ur_x/lr_x; 

    //rinse and repeat for y 

    wl_x + wl_y
}