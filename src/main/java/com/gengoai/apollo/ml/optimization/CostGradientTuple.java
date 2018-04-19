package com.gengoai.apollo.ml.optimization;

import com.gengoai.apollo.linear.NDArray;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   double cost;
   GradientParameter gradient;
   NDArray[] activations;
}// END OF CostGradientTuple
