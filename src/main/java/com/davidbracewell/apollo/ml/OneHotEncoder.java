package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.conversion.Cast;
import lombok.Getter;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class OneHotEncoder implements Vectorizer {
   private final String paddingValue;
   @Getter
   private final int maxFeatures;
   @Getter
   private EncoderPair encoderPair;
   @Getter
   private int outputDimension;
   private int numFeatures;

   public OneHotEncoder(@NonNull EncoderPair encoderPair, int maxFeatures, String paddingValue) {
      this(maxFeatures, paddingValue);
      setEncoderPair(encoderPair);
   }

   public OneHotEncoder(int maxFeatures, String paddingValue) {
      this.maxFeatures = maxFeatures;
      this.paddingValue = paddingValue;
   }

   @Override
   public Vector apply(Example example) {
      Vector vPrime = Vector.sZeros(outputDimension);
      int offset = 0;

      final List<Feature> features;
      final Object label;
      final double weight;
      if (example instanceof Instance) {
         Instance instance = Cast.as(example);
         features = instance.getFeatures();
         label = instance.getLabel();
         weight = instance.getWeight();
      } else {
         features = new ArrayList<>();
         Sequence sequence = Cast.as(example);
         sequence.asInstances().forEach(i -> features.add(i.getFeatures().get(0)));
         label = sequence.get(sequence.size() - 1).getLabel();
         weight = 1.0;
      }

      for (Feature feature : features) {
         if (offset > maxFeatures) {
            break;
         }
         int index = (int) encoderPair.encodeFeature(feature.getName());
         if (index >= 0) {
            vPrime.set(offset * numFeatures + index, 1);
         }
         offset++;
      }

      while (offset < maxFeatures) {
         int index = (int) encoderPair.encodeFeature(paddingValue);
         if (index >= 0) {
            vPrime.set(offset * numFeatures + index, 1);
         }
         offset++;
      }

      vPrime.setLabel(encoderPair.encodeLabel(label));
      vPrime.setWeight(weight);
      return vPrime;
   }

   @Override
   public void setEncoderPair(@NonNull EncoderPair encoderPair) {
      this.encoderPair = encoderPair;
      this.encoderPair.getFeatureEncoder().unFreeze();
      this.encoderPair.getFeatureEncoder().encode(paddingValue);
      this.encoderPair.getFeatureEncoder().freeze();
      this.numFeatures = encoderPair.numberOfFeatures();
      this.outputDimension = maxFeatures * numFeatures;
   }


}// END OF OneHotEncoder
