package com.gengoai.apollo.ml.optimization;

import com.gengoai.apollo.linear.NDArray;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class GradientParameter {
   NDArray weightGradient;
   NDArray biasGradient;

   public static GradientParameter calculate(NDArray input, NDArray error) {
      //input is numFeatures x numExamples
      //error is numLabels x numExamples
      //output should be numLabels x numFeatures
      return GradientParameter.of(error.mmul(input.T()), error);
   }

   public GradientParameter add(GradientParameter other) {
      NDArray w = weightGradient.add(other.weightGradient);
      NDArray b = biasGradient.add(other.biasGradient);
      return of(w, b);
   }

}// END OF GradientParameter
