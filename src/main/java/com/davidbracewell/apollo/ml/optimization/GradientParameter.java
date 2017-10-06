package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.apollo.linear.NDArray;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class GradientParameter {
   NDArray weightGradient;
   NDArray biasGradient;

   public static GradientParameter calculate(NDArray input,
                                             NDArray error
                                            ) {
      return GradientParameter.of(error.mmul(input.T()), error);
   }

}// END OF GradientParameter
