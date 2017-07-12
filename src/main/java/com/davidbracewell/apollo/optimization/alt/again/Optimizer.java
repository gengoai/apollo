package com.davidbracewell.apollo.optimization.alt.again;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   CostWeightTuple optimize(WeightMatrix theta,
                            SerializableSupplier<MStream<? extends Vector>> stream,
                            CostFunction costFunction,
                            TerminationCriteria terminationCriteria,
                            LearningRate learningRate,
                            WeightUpdate weightUpdater,
                            int reportInterval
                           );

}// END OF Optimizer
