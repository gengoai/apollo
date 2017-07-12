package com.davidbracewell.apollo.optimization.alt.again;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   private double cost;
   private GradientMatrix gradient;
}//END OF CostGradientTuple
