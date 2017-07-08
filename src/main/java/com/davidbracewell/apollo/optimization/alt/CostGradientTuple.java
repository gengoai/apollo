package com.davidbracewell.apollo.optimization.alt;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   private double cost;
   private Gradient gradient;
}//END OF CostGradientTuple
