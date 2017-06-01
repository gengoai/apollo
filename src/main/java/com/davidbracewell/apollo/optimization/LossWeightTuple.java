package com.davidbracewell.apollo.optimization;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class LossWeightTuple {
   private double loss;
   private Weights weights;
}// END OF LossWeightTuple
