package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.NDArray;

/**
 * The type Gradient parameter.
 *
 * @author David B. Bracewell
 */
public class GradientParameter {
   private final NDArray biasGradient;
   private final NDArray weightGradient;

   /**
    * Instantiates a new Gradient parameter.
    *
    * @param weightGradient the weight gradient
    * @param biasGradient   the bias gradient
    */
   protected GradientParameter(NDArray weightGradient, NDArray biasGradient) {
      this.weightGradient = weightGradient;
      this.biasGradient = biasGradient;
   }

   /**
    * Calculate gradient parameter.
    *
    * @param input the input
    * @param error the error
    * @return the gradient parameter
    */
   public static GradientParameter calculate(NDArray input, NDArray error) {
      //input is numFeatures x numExamples
      //error is numLabels x numExamples
      //output should be numLabels x numFeatures
      return GradientParameter.of(error.mmul(input.T()), error);
   }

   /**
    * Of gradient parameter.
    *
    * @param weights the weights
    * @param bias    the bias
    * @return the gradient parameter
    */
   public static GradientParameter of(NDArray weights, NDArray bias) {
      return new GradientParameter(weights, bias);
   }

   /**
    * Add gradient parameter.
    *
    * @param other the other
    * @return the gradient parameter
    */
   public GradientParameter add(GradientParameter other) {
      NDArray w = weightGradient.add(other.weightGradient);
      NDArray b = biasGradient.add(other.biasGradient);
      return of(w, b);
   }

   /**
    * Gets bias gradient.
    *
    * @return the bias gradient
    */
   public NDArray getBiasGradient() {
      return biasGradient;
   }

   /**
    * Gets weight gradient.
    *
    * @return the weight gradient
    */
   public NDArray getWeightGradient() {
      return weightGradient;
   }

}// END OF GradientParameter
