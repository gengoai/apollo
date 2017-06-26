package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;

/**
 * The interface Online optimizer.
 *
 * @author David B. Bracewell
 */
public interface OnlineOptimizer {

   /**
    * Optimize loss weight tuple.
    *
    * @param start               the start
    * @param stream              the stream
    * @param costFunction        the cost function
    * @param terminationCriteria the termination criteria
    * @param learningRate        the learning rate
    * @param weightUpdater       the weight updater
    * @param verbose             the verbose
    * @return the loss weight tuple
    */
   CostWeightTuple optimize(Weights start,
                            SerializableSupplier<MStream<? extends Vector>> stream,
                            OnlineCostFunction costFunction,
                            TerminationCriteria terminationCriteria,
                            LearningRate learningRate,
                            WeightUpdate weightUpdater,
                            boolean verbose
                           );


}// END OF OnlineOptimizer
