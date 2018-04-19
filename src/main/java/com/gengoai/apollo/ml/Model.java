package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.encoder.Encoder;
import com.gengoai.apollo.ml.encoder.EncoderPair;
import com.gengoai.apollo.ml.encoder.LabelEncoder;
import com.gengoai.mango.io.resource.Resource;
import lombok.NonNull;

import java.io.Serializable;

/**
 * <p>A generic interface to classification, regression, and sequence models</p>
 *
 * @author David B. Bracewell
 */
public interface Model extends Serializable {

   /**
    * Reads the model from the given resource
    *
    * @param <T>           the type of model to read
    * @param modelResource the resource containing the serialized model
    * @return the deserialized model
    * @throws Exception Something went wrong reading the model
    */
   static <T extends Model> T read(@NonNull Resource modelResource) throws Exception {
      return modelResource.readObject();
   }

   /**
    * Convenience method for decoding an encoded feature value into an Object
    *
    * @param value the encoded feature value
    * @return the decoded object
    */
   default Object decodeFeature(double value) {
      return getEncoderPair().decodeFeature(value);
   }

   /**
    * Convenience method for decoding an encoded label value into an Object
    *
    * @param value the encoded label value
    * @return the decoded object
    */
   default Object decodeLabel(double value) {
      return getEncoderPair().decodeLabel(value);
   }

   /**
    * Convenience method for encoding a feature into a double value
    *
    * @param feature the feature to encode
    * @return the encoded double value
    */
   default double encodeFeature(Object feature) {
      return getEncoderPair().encodeFeature(feature);
   }

   /**
    * Convenience method for encoding a label into a double value
    *
    * @param label the label to encode
    * @return the encoded double value
    */
   default double encodeLabel(Object label) {
      return getEncoderPair().encodeLabel(label);
   }

   /**
    * Signal that training of this model has finished.
    */
   default void finishTraining() {
      getEncoderPair().freeze();
   }

   /**
    * Gets the encoder pair this model uses.
    *
    * @return the encoder pair
    */
   EncoderPair getEncoderPair();

   /**
    * Convenience method for retrieving the feature encoder.
    *
    * @return the feature encoder
    */
   default Encoder getFeatureEncoder() {
      return getEncoderPair().getFeatureEncoder();
   }

   /**
    * Convenience method for retrieving the label encoder
    *
    * @return the label encoder
    */
   default LabelEncoder getLabelEncoder() {
      return getEncoderPair().getLabelEncoder();
   }

   /**
    * Convenience method for retrieving the total number of features.
    *
    * @return the number of model features
    */
   default int numberOfFeatures() {
      return getEncoderPair().numberOfFeatures();
   }

   /**
    * Convenience method for retrieving the total number of labels
    *
    * @return the number of possible labels
    */
   default int numberOfLabels() {
      return getEncoderPair().numberOfLabels();
   }

   /**
    * Serializes the model to the given resource.
    *
    * @param modelResource the resource to serialize the model to
    * @throws Exception Something went wrong serializing the model
    */
   default void write(@NonNull Resource modelResource) throws Exception {
      modelResource.setIsCompressed(true).writeObject(this);
   }

}// END OF Model
