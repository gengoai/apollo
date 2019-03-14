package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.NDArray;

/**
 * @author David B. Bracewell
 */
public class GradientParameter {
   private final NDArray biasGradient;
   private final NDArray weightGradient;

   protected GradientParameter(NDArray weightGradient, NDArray biasGradient) {
      this.weightGradient = weightGradient;
      this.biasGradient = biasGradient;
   }

   public static GradientParameter calculate(NDArray input, NDArray error) {
      //input is numFeatures x numExamples
      //error is numLabels x numExamples
      //output should be numLabels x numFeatures
      return GradientParameter.of(error.mmul(input.T()), error);
   }

   public static GradientParameter of(NDArray weights, NDArray bias) {
      return new GradientParameter(weights, bias);
   }

   public GradientParameter add(GradientParameter other) {
      NDArray w = weightGradient.add(other.weightGradient);
      NDArray b = biasGradient.add(other.biasGradient);
      return of(w, b);
   }

   public NDArray getBiasGradient() {
      return biasGradient;
   }

   public NDArray getWeightGradient() {
      return weightGradient;
   }

}// END OF GradientParameter
