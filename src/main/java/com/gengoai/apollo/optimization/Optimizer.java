package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.stream.MStream;

/**
 * Optimizes parameters using a set of data against a given {@link CostFunction}.
 *
 * @param <THETA> the type parameter
 * @author David B. Bracewell
 */
public interface Optimizer<THETA> {

   /**
    * Gets final cost.
    *
    * @return the final cost
    */
   double getFinalCost();

   /**
    * Optimize.
    *
    * @param startingTheta    the starting theta
    * @param stream           the stream
    * @param costFunction     the cost function
    * @param stoppingCriteria the stopping criteria
    * @param weightUpdate     the weight update
    */
   void optimize(THETA startingTheta,
                 MStream<NDArray> stream,
                 CostFunction<THETA> costFunction,
                 StoppingCriteria stoppingCriteria,
                 WeightUpdate weightUpdate
                );


   /**
    * Resets the optimizer.
    */
   void reset();

}// END OF Optimizer