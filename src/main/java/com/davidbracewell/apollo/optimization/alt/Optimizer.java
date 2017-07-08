package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   CostWeightTuple optimize(WeightVector theta,
                            MStream<? extends Vector> stream,
                            CostFunction costFunction,
                            TerminationCriteria terminationCriteria,
                            LearningRate learningRate,
                            WeightUpdate weightUpdater,
                            boolean verbose
                           );

}// END OF Optimizer
