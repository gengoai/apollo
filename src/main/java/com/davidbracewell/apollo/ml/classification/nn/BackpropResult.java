package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "from")
public class BackpropResult {
   Matrix delta;
   Matrix weightGradient;
   Matrix biasGradient;
}// END OF BackpropResult
