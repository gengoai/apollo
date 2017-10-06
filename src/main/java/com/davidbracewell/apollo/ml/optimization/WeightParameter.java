package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.apollo.linear.Axis;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class WeightParameter {
   @Getter
   private final int numberOfLabels;
   @Getter
   private final int numberOfFeatures;
   @Getter
   private final boolean isBinary;
   @Getter
   @Setter
   private NDArray weights;
   @Getter
   @Setter
   private NDArray bias;

   public WeightParameter(int numberOfLabels, int numberOfFeatures) {
      this.numberOfLabels = numberOfLabels;
      this.numberOfFeatures = numberOfFeatures;
      this.isBinary = (numberOfLabels == 2);
      this.bias = NDArrayFactory.defaultFactory().zeros(getNumberOfWeightVectors());
      this.weights = NDArrayFactory.defaultFactory().rand(getNumberOfWeightVectors(), numberOfFeatures);
   }

   public NDArray dot(NDArray vector) {
//      System.out.println(weights.shape() + ", " + vector.shape() + ", " + bias.shape());
//      System.out.println(weights.mmul(vector).shape());
      return weights.mmul(vector).addi(bias, Axis.COlUMN);
   }

   public int getNumberOfWeightVectors() {
      return isBinary ? 1 : numberOfLabels;
   }


}// END OF WeightParameter
