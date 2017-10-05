package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linear.EmptyNDArray;
import com.davidbracewell.apollo.linear.ForwardingNDArray;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.ScalarNDArray;
import com.davidbracewell.conversion.Cast;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public class LabeledNDArray extends ForwardingNDArray {
   private static final long serialVersionUID = 1L;
   @Setter
   private Object label;
   @Getter
   @Setter
   private double weight;
   @Setter
   private Object predicted;

   public LabeledNDArray(NDArray delegate) {
      super(delegate);
   }

   public double doubleLabel() {
      if (label == null) {
         return Double.NaN;
      }
      return Cast.<Number>as(label).doubleValue();
   }

   public double doublePredictedLabel() {
      if (predicted == null) {
         return Double.NaN;
      }
      return Cast.<Number>as(predicted).doubleValue();
   }

   public boolean hasLabel() {
      return label != null;
   }

   public boolean hasPredictedLabel() {
      return predicted != null;
   }

   public <T> T label() {
      return Cast.as(label);
   }

   public <T> T predicted() {
      return Cast.as(predicted);
   }


   public NDArray vectorLabel() {
      if (label == null) {
         return new EmptyNDArray();
      } else if (label instanceof Number) {
         return new ScalarNDArray(Cast.<Number>as(label).doubleValue());
      }
      return Cast.as(label);
   }

   public NDArray vectorPredictedLabel() {
      if (predicted == null) {
         return new EmptyNDArray();
      } else if (predicted instanceof Number) {
         return new ScalarNDArray(Cast.<Number>as(predicted).doubleValue());
      }
      return Cast.as(predicted);
   }

}// END OF FeatureNDArray
