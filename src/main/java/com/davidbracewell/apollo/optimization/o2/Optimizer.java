package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   CostWeightTuple optimize(WeightComponent initialTheta,
                            SerializableSupplier<MStream<? extends Vector>> stream,
                            CostFunction costFunction,
                            TerminationCriteria terminationCriteria,
                            LearningRate learningRate,
                            WeightUpdate weightUpdater,
                            boolean verbose
                           );

}// END OF Optimizer
