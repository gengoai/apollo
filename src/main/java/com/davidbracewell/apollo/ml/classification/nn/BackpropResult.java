package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArray;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "from")
public class BackpropResult {
   NDArray delta;
   NDArray weightGradient;
   NDArray biasGradient;
}// END OF BackpropResult
