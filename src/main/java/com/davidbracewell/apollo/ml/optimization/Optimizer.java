package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.logging.Loggable;
import com.davidbracewell.stream.MStream;

/**
 * @author David B. Bracewell
 */
public interface Optimizer<THETA> extends Loggable {

   double getFinalCost();

   void optimize(THETA startingTheta,
                 SerializableSupplier<MStream<NDArray>> stream,
                 CostFunction<THETA> costFunction,
                 TerminationCriteria terminationCriteria,
                 int reportInterval
                );

   default void report(int interval, int iteration, int maxIteration, boolean converged, double cost) {
      if (interval > 0 && (iteration == 0 || (iteration + 1) % interval == 0 || (iteration + 1) == maxIteration || converged)) {
         logInfo("iteration={0}, totalLoss={1}", (iteration + 1), cost);
      }
   }

   void reset();

}// END OF Optimizer