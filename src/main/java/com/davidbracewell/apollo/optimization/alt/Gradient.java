package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class Gradient {
   private Vector weightGradient;
   private double biasGradient;
}//END OF Gradient
