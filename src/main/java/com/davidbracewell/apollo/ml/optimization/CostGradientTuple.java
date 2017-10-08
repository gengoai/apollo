package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.apollo.linear.NDArray;
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
