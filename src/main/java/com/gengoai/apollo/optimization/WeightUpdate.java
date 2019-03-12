package com.gengoai.apollo.optimization;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.p2.NDArray;
import com.gengoai.tuple.Tuple2;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdate extends Copyable<WeightUpdate> {

   void reset();

   Tuple2<NDArray, Double> update(LinearModelParameters weights,
                                  NDArray input,
                                  NDArray output,
                                  NDArray delta,
                                  int iteration,
                                  boolean calculateOutDelta
                                 );

   double update(LinearModelParameters weights, GradientParameter gradient, int iteration);

}// END OF WeightUpdate