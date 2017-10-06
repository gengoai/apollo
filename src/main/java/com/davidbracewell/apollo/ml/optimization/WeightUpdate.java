package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.Copyable;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.tuple.Tuple2;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdate extends Copyable<WeightUpdate> {

   void reset();

   Tuple2<Double, NDArray> update(LinearModelParameters weights,
                                  NDArray input,
                                  NDArray output,
                                  NDArray delta,
                                  int iteration,
                                  boolean calculateOutDelta
                                 );

   double update(LinearModelParameters weights, GradientParameter gradient, int iteration);

}// END OF WeightUpdate