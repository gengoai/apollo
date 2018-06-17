package com.gengoai.apollo.ml.encoder;

import com.gengoai.apollo.ml.Example;
import lombok.EqualsAndHashCode;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The type Encoder pair.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
public final class EncoderPair implements Serializable {
   public static final EncoderPair NO_OPT = new EncoderPair(new NoOptLabelEncoder(), new NoOptEncoder());
   private static final long serialVersionUID = 1L;
   private final LabelEncoder labelEncoder;
   private final Encoder featureEncoder;

   /**
    * Instantiates a new Encoder pair.
    *
    * @param labelEncoder   the label encoder
    * @param featureEncoder the feature encoder
    */
   public EncoderPair(@NonNull LabelEncoder labelEncoder, @NonNull Encoder featureEncoder) {
      this.labelEncoder = labelEncoder;
      this.featureEncoder = featureEncoder;
   }

   /**
    * Creates a new Encoder pair with the same type of label and feature encoder
    *
    * @return the new and empty encoder pair
    */
   public EncoderPair createNew() {
      return new EncoderPair(labelEncoder.createNew(), featureEncoder.createNew());
   }

   /**
    * Decodes the double into a feature.
    *
    * @param value the encoded value
    * @return the feature associated with the value or null if none
    */
   public Object decodeFeature(double value) {
      return featureEncoder.decode(value);
   }

   /**
    * Decodes the double into a label.
    *
    * @param value the encoded value
    * @return the label associated with the value or null if none
    */
   public Object decodeLabel(double value) {
      return labelEncoder.decode(value);
   }

   /**
    * Encodes both the label space and the feature space of the given example
    *
    * @param <T>     the example type parameter
    * @param example the example
    * @return the example for fluent interface
    */
   public <T extends Example> T encode(T example) {
      if (example != null) {
         labelEncoder.encode(example.getLabelSpace());
         featureEncoder.encode(example.getFeatureSpace());
      }
      return example;
   }

   /**
    * Encodes the given feature into a double
    *
    * @param feature the feature
    * @return the encoded value
    */
   public double encodeFeature(Object feature) {
      return featureEncoder.encode(feature);
   }

   /**
    * Encodes the given label into a double
    *
    * @param label the label
    * @return the encoded value
    */
   public double encodeLabel(Object label) {
      return labelEncoder.encode(label);
   }

   /**
    * Gets the index of the given feature
    *
    * @param featureName the feature to lookup
    * @return the index (int value) of the encoded feature
    */
   public int featureIndex(Object featureName) {
      return featureEncoder.index(featureName);
   }

   /**
    * Freezes the label and feature encoders restricting new objects from being mapped to values.
    */
   public void freeze() {
      this.labelEncoder.freeze();
      this.featureEncoder.freeze();
   }

   /**
    * Gets the feature encoder.
    *
    * @return the feature encoder
    */
   public Encoder getFeatureEncoder() {
      return featureEncoder;
   }

   /**
    * Gets the label encoder.
    *
    * @return the label encoder
    */
   public LabelEncoder getLabelEncoder() {
      return labelEncoder;
   }

   /**
    * Gets the index of the given label
    *
    * @param label the label to lookup
    * @return the index (int value) of the encoded label
    */
   public int labelIndex(Object label) {
      return labelEncoder.index(label);
   }

   /**
    * Gets the number of features in the feature encoder
    *
    * @return the number of features
    */
   public int numberOfFeatures() {
      return featureEncoder.size();
   }

   /**
    * Gets the number of labels in the label encoder
    *
    * @return the number of labels
    */
   public int numberOfLabels() {
      return labelEncoder.size();
   }


}// END OF EncoderPair
