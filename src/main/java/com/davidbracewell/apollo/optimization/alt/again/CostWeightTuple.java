package com.davidbracewell.apollo.optimization.alt.again;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostWeightTuple {
   double cost;
   WeightMatrix weights;
}//END OF CostWeightTuple
