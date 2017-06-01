package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.regularization.WeightUpdater;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   Weights optimize(Weights start,
                    MStream<? extends Vector> stream,
                    StochasticCostFunction costFunction,
                    TerminationCriteria terminationCriteria,
                    LearningRate learningRate,
                    WeightUpdater weightUpdater,
                    boolean verbose
                   );


}// END OF Optimizer
