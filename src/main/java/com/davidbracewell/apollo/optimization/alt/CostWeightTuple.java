package com.davidbracewell.apollo.optimization.alt;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostWeightTuple {
   double cost;
   WeightVector weights;
}//END OF CostWeightTuple
