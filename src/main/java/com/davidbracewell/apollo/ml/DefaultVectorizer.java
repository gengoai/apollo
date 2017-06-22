package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.conversion.Cast;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
@NoArgsConstructor
@AllArgsConstructor
public class DefaultVectorizer implements Vectorizer {
   @Getter
   private EncoderPair encoderPair;


   @Override
   public Vector apply(Example example) {
      if (example instanceof Instance) {
         Instance ii = Cast.as(example);
         FeatureVector vector = new FeatureVector(encoderPair);
         boolean isHash = encoderPair.getFeatureEncoder() instanceof HashingEncoder;
         ii.forEach(f -> {
            int fi = (int) encoderPair.encodeFeature(f.getName());
            if (fi != -1) {
               if (isHash) {
                  vector.set(fi, 1.0);
               } else {
                  vector.set(fi, f.getValue());
               }
            }
         });
         vector.setLabel(encoderPair.encodeLabel(ii.getLabel()));
         vector.setWeight(ii.getWeight());
         return vector;
      }
      throw new UnsupportedOperationException("Only Instances are supported");
   }

   @Override
   public int getOutputDimension() {
      return encoderPair.numberOfFeatures();
   }

   @Override
   public void setEncoderPair(@NonNull EncoderPair encoderPair) {
      this.encoderPair = encoderPair;
   }
}// END OF DefaultVectorizer
