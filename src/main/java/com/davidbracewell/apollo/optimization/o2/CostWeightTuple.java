package com.davidbracewell.apollo.optimization.o2;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostWeightTuple {
   double loss;
   WeightComponent components;
}// END OF CostWeightTuple
