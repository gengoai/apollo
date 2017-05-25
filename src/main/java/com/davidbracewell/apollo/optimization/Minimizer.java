package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Minimizer {

   Vector minimize(Vector start, MStream<Vector> stream, StochasticCostFunction costFunction, int maxIterations, boolean verbose);

}// END OF Minimizer
