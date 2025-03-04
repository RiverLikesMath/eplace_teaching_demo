use eplace::NLparams;
use ndarray::array;

//use nextpnr; 

mod bad_tests;
mod dct;
mod density;
mod elfplace;
mod eplace;
mod ref_dct;
mod util;
mod wirelength;
mod wl_grad;

///in this oversimplified example, there will be a small number of  logic elements placed on an mxm grid
///each logic element will be 1.5 pixel wide and tall so we don't have to worry about eplace's
///smoothing.
///for simplicities sake, we'll assume all the elements are on a single global net. this is very
///silly, but it'll make the demonstration a bit easier
#[allow(unused_variables)] //we're allowing unused variables in main here, at least for now.
fn main() {
    /* 
        calls to nextpnr
        import cells , import nets 

            how do we use these to compute things like 
                resource capacity - pin or other data from nextpnr? 
                similar things - other info about board that may be needed in order to run code, especially elfplace and multistatics 
                
        construct internal/placement grid 

        eplace/elfplace loop - now the results of this are being sent to nextpnr when converged
        what the what git/github?
        rough algorithm: 
            get context from nextpnr 
            pull all relevant data from the context (???), preferring nextPNR's data types when possible 
            run eplace, elfplace, or multistatics to convergence 
            return to nextpnr
     */

    /*
        //n and m are the dimensions of an array
        //what is this array? The tilegrid of the fpga
        //each grid element of the array will have some amount of bels in it 

        let n = getGridDimX(ctx);
        let m = getGridDimY(ctx);

        // we'll probably also have to call getBels() -- iterator over every bel
        // bels can be of the different elfplace types - they'll have to be filtered and sorted 
        // ideally called once as part of startup 
        //bel groups - yay! eplace will not really care about them, elfplace and the multistatics will 


        //n x m array
        let nets = Nets::new(ctx); //it's something! 
     */
    
    let cell_centers = array![
        [28. / 8., 28. / 8.], //x,y, initial placement
        [56. / 8., 58. / 8.],
        [9. / 8., 19. / 8.],
        [99. / 8., 101. / 8.0],
        [10.8, 6.7],
        [1.27, 2.04],
        [14.81, 14.25],
        [14.22, 14.44],
        [1.5, 1.5],
        [1.9, 12.9],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 4.0],
        [6.0, 6.0],
        [5.5, 5.5],
        [4.75, 4.75],
        [5.25, 12.25],
        [40., 40.]
    ];

    let m: usize = 64; // m == sqrt(number of bins), max 1024, must be power of 2

    //before we start the loop, we need an initial set of parameters to feed to
    //the nl solver - those are all calculated and grabbed from here
    let initial_loop_params = eplace::calc_initial_params(&cell_centers, m);

    //my heart is telling me to do this recursively, but the closest thing to the paper
    //would be a for loop
    let mut curr_eplace_iteration = initial_loop_params;
    let max_iter = 5;
    for i in 1..max_iter {
        let prev = curr_eplace_iteration;
        curr_eplace_iteration = eplace::eplace(prev, m);

        //this will mean we have a minimum of 5 iterations, probably a better way to write it
        if i % (max_iter / 5) == 0 {
            debugs(&curr_eplace_iteration, i);
        }
    }
    debugs(&curr_eplace_iteration, 1000000);
}

fn debugs(curr_eplace_iteration: &NLparams, i: usize) {
    println!();

    println!("Beginning new loop");
    println!("current iteration is: {i}");
    println!("current objective function is: ");
    dbg!(curr_eplace_iteration.f_k);
    dbg!(curr_eplace_iteration.alpha);
}
