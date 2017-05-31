package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   Weights optimize(Weights start, MStream<? extends Vector> stream, StochasticCostFunction costFunction, int numPasses, boolean verbose);


}// END OF Optimizer
