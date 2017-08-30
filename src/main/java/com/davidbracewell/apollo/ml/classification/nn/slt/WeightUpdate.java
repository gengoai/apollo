package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.Copyable;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.tuple.Tuple2;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdate extends Copyable<WeightUpdate> {

   double update(Matrix weights,
                 Matrix bias,
                 Matrix wGrad,
                 Matrix bGrad,
                 int iteration
                );

   Tuple2<Matrix, Double> update(Matrix weights,
                                 Matrix bias,
                                 Matrix input,
                                 Matrix output,
                                 Matrix delta,
                                 int iteration,
                                 boolean calculateOutDelta
                                );


}// END OF WeightUpdate
