package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class LossGradientTuple {
   double loss;
   Vector gradient;
}// END OF LossGradientTuple
