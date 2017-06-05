package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.regularization.Regularizer;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   LossWeightTuple optimize(Weights start,
                            SerializableSupplier<MStream<? extends Vector>> stream,
                            StochasticCostFunction costFunction,
                            TerminationCriteria terminationCriteria,
                            LearningRate learningRate,
                            Regularizer weightUpdater,
                            boolean verbose
                   );


}// END OF Optimizer
