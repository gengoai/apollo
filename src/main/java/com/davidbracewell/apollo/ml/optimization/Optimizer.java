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
                 WeightUpdate weightUpdate,
                 int reportInterval
                );

   default boolean report(int interval,
                          int iteration,
                          TerminationCriteria terminationCriteria,
                          double cost,
                          String time
                         ) {
      boolean converged = terminationCriteria.check(cost);
      if (interval > 0
             && (iteration == 0
                    || (iteration + 1) % interval == 0
                    || (iteration + 1) == terminationCriteria.maxIterations()
                    || converged)) {
         logInfo("iteration={0}, loss={1}, time={2}", (iteration + 1), cost, time);
      }
      return converged;
   }

   void reset();

}// END OF Optimizer