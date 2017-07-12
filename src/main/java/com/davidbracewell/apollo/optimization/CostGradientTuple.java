package com.davidbracewell.apollo.optimization;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   private double cost;
   private GradientMatrix gradient;
}//END OF CostGradientTuple
