package com.davidbracewell.apollo.optimization;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostWeightTuple {
   double loss;
   WeightComponent components;
}// END OF CostWeightTuple
