package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.conversion.Cast;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

/**
 * <p>A specialized sparse vector that stores an encoded label and an optionally an encoded predicted label. The
 * dimension of the vector is dynamic and depends on the underlying feature encoder.</p>
 *
 * @author David B. Bracewell
 */
public class FeatureVector extends SparseVector {
   private static final long serialVersionUID = 1L;
   private final EncoderPair encoderPair;
   private double label = Double.NaN;
   @Getter
   private double predictedLabel = Double.NaN;
   @Getter
   @Setter
   private double weight = 1.0;

   /**
    * Instantiates a new Feature vector.
    *
    * @param encoderPair the feature encoder
    */
   public FeatureVector(@NonNull EncoderPair encoderPair) {
      super(0);
      this.encoderPair = encoderPair;
   }

   @Override
   public int dimension() {
      return encoderPair.getFeatureEncoder().size();
   }

   /**
    * Decodes the label returning its string form
    *
    * @return the decoded label
    */
   public String getDecodedLabel() {
      return Cast.as(encoderPair.decodeLabel(label));
   }

   /**
    * Gets the encoder pair used by the feature vector.
    *
    * @return the encoder pair
    */
   public EncoderPair getEncoderPair() {
      return encoderPair;
   }

   @SuppressWarnings("unchecked")
   @Override
   public Double getLabel() {
      return label;
   }

   /**
    * Sets the label of the vector.
    *
    * @param label the label
    */
   public void setLabel(Object label) {
      this.label = encoderPair.encodeLabel(label);
      if (this.label == -1) {
         this.label = Double.NaN;
      }
   }

   /**
    * Checks if the vector has a label assigned to it
    *
    * @return True if the vector has a label, False otherwise
    */
   public boolean hasLabel() {
      return Double.isFinite(label);
   }

   /**
    * Sets the given feature to the given value.
    *
    * @param featureName  the feature name
    * @param featureValue the feature value
    * @return True if the feature name was valid and the value was set
    */
   public boolean set(String featureName, double featureValue) {
      int index = (int) encoderPair.encodeFeature(featureName);
      if (index < 0) {
         return false;
      }
      set(index, featureValue);
      return true;
   }

   /**
    * Sets the value of the given feature.
    *
    * @param feature the feature
    * @return True if the feature is valid and its value set
    */
   public boolean set(Feature feature) {
      return feature != null && set(feature.getName(), feature.getValue());
   }

   /**
    * Sets the label of the vector.
    *
    * @param label the label
    */
   public void setLabel(double label) {
      this.label = label;
   }

   /**
    * Sets the predicted label of the vector.
    *
    * @param label the predicted label
    */
   public void setPredictedLabel(double label) {
      this.predictedLabel = label;
   }

   /**
    * Sets the predicted label of the vector.
    *
    * @param label the predicted label
    */
   public void setPredictedLabel(Object label) {
      this.predictedLabel = encoderPair.encodeLabel(label);
      if (this.predictedLabel == -1) {
         this.predictedLabel = Double.NaN;
      }
   }

   /**
    * Transforms the feature vector using a new encoder pair.
    *
    * @param newEncoderPair the new feature encoder
    * @return the feature vector
    */
   public FeatureVector transform(@NonNull EncoderPair newEncoderPair) {
      FeatureVector newVector = new FeatureVector(newEncoderPair);
      forEachSparse(entry -> newVector.set(encoderPair.decodeFeature(entry.getIndex()).toString(),
                                           entry.getValue()
                                          )
                   );
      newVector.setLabel(getDecodedLabel());
      return newVector;
   }

}// END OF FeatureVector
