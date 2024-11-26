use ndarray::{Array, Array2, Axis};

const bin_w: f64 = 1.;

fn add_density(density: &mut Array2<f64>, x: usize, y: usize, added_density: f64) {
    density[[x, y]] += added_density;
}

///calculates the density given the current location of each centr, by calculating overlap of each cell w/ bins
///there's probably a shorter clearer way to do this
pub fn calc_density(cell_centers: &Array2<f64>, m: usize) -> Array2<f64> {
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

    //subtract the DC component (the total density / m^2 )
    let dc_component = density.sum() / (m as f64).powi(2);
    let dc_array = Array::from_elem((m, m), dc_component);
    let zeroed_density = &density - &dc_array;

    zeroed_density
}
