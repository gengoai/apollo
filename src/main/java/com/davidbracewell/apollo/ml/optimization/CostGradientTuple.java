package com.davidbracewell.apollo.ml.optimization;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   double cost;
   GradientParameter gradient;
}// END OF CostGradientTuple
