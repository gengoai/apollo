package com.gengoai.apollo.optimization;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.tuple.Tuple2;

/**
 * The interface Weight update.
 *
 * @author David B. Bracewell
 */
public interface WeightUpdate extends Copyable<WeightUpdate> {

   /**
    * Reset.
    */
   void reset();

   /**
    * Update tuple 2.
    *
    * @param weights           the weights
    * @param input             the input
    * @param output            the output
    * @param delta             the delta
    * @param iteration         the iteration
    * @param calculateOutDelta the calculate out delta
    * @return the tuple 2
    */
   Tuple2<NDArray, Double> update(LinearModelParameters weights,
                                  NDArray input,
                                  NDArray output,
                                  NDArray delta,
                                  int iteration,
                                  boolean calculateOutDelta
                                 );

   /**
    * Update double.
    *
    * @param weights   the weights
    * @param gradient  the gradient
    * @param iteration the iteration
    * @return the double
    */
   double update(LinearModelParameters weights, GradientParameter gradient, int iteration);

}// END OF WeightUpdate