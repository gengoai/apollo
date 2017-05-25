package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value
public class CostGradientTuple {
   double cost;
   Vector gradient;
}// END OF CostGradientTuple
