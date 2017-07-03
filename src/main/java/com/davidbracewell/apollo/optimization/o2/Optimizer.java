package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.optimization.CostWeightTuple;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   CostWeightTuple optimize(WeightComponent initialTheta,
                            CostFunction costFunction,
                            TerminationCriteria terminationCriteria,
                            LearningRate learningRate,
                            WeightUpdate weightUpdater,
                            boolean verbose
                           );

}// END OF Optimizer
