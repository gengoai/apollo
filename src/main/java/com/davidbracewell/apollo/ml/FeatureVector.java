package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.SparseVector;
import lombok.NonNull;

/**
 * The type Feature vector.
 *
 * @author David B. Bracewell
 */
public class FeatureVector extends SparseVector {
   private static final long serialVersionUID = 1L;
   private final EncoderPair encoderPair;
   private double label = Double.NaN;
   private double predictedLabel = Double.NaN;

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
    * Gets decoded label.
    *
    * @return the decoded label
    */
   public Object getDecodedLabel() {
      return encoderPair.decodeLabel(label);
   }

   /**
    * Gets feature encoder.
    *
    * @return the feature encoder
    */
   public EncoderPair getEncoderPair() {
      return encoderPair;
   }

   /**
    * Gets label.
    *
    * @return the label
    */
   @SuppressWarnings("unchecked")
   public Double getLabel() {
      return label;
   }

   /**
    * Sets label.
    *
    * @param label the label
    */
   public void setLabel(double label) {
      this.label = label;
   }

   /**
    * Has label boolean.
    *
    * @return the boolean
    */
   public boolean hasLabel() {
      return Double.isFinite(label);
   }

   /**
    * Set boolean.
    *
    * @param featureName  the feature name
    * @param featureValue the feature value
    * @return the boolean
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
    * Set boolean.
    *
    * @param feature the feature
    * @return the boolean
    */
   public boolean set(Feature feature) {
      return feature != null && set(feature.getName(), feature.getValue());
   }

   /**
    * Sets label.
    *
    * @param o the o
    */
   public void setLabel(Object o) {
      this.label = encoderPair.encodeLabel(o);
      if (this.label == -1) {
         this.label = Double.NaN;
      }
   }

   /**
    * Transform feature vector.
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


   /**
    * Gets predicted label.
    *
    * @return the predicted label
    */
   public double getPredictedLabel() {
      return predictedLabel;
   }

   /**
    * Sets predicted label.
    *
    * @param predictedLabel the predicted label
    */
   public void setPredictedLabel(double predictedLabel) {
      this.predictedLabel = predictedLabel;
   }
}// END OF FeatureVector
