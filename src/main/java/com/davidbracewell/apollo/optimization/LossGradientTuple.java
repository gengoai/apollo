package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class LossGradientTuple {
   double cost;
   Vector gradient;
}// END OF LossGradientTuple
