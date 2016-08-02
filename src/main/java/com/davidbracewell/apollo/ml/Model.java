package com.davidbracewell.apollo.ml;

import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.Readable;
import com.davidbracewell.io.structured.Writable;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The type Model.
 *
 * @author David B. Bracewell
 */
public interface Model extends Serializable, Writable, Readable {

  /**
   * Read model classifier.
   *
   * @param <T>           the type parameter
   * @param modelResource the model resource
   * @return the classifier
   * @throws Exception the exception
   */
  static <T extends Model> T read(@NonNull Resource modelResource) throws Exception {
    return modelResource.readObject();
  }

  /**
   * Number of labels int.
   *
   * @return the int
   */
  default int numberOfLabels() {
    return getEncoderPair().numberOfLabels();
  }

  /**
   * Number of features int.
   *
   * @return the int
   */
  default int numberOfFeatures() {
    return getEncoderPair().numberOfFeatures();
  }

  /**
   * Gets label encoder.
   *
   * @return the label encoder
   */
  default LabelEncoder getLabelEncoder() {
    return getEncoderPair().getLabelEncoder();
  }

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  default Encoder getFeatureEncoder() {
    return getEncoderPair().getFeatureEncoder();
  }

  /**
   * Gets encoder pair.
   *
   * @return the encoder pair
   */
  EncoderPair getEncoderPair();

  /**
   * Write model.
   *
   * @param modelResource the model resource
   * @throws Exception the exception
   */
  default void write(@NonNull Resource modelResource) throws Exception {
    modelResource.setIsCompressed(true).writeObject(this);
  }

  /**
   * Encode label double.
   *
   * @param label the label
   * @return the double
   */
  default double encodeLabel(Object label) {
    return getEncoderPair().encodeLabel(label);
  }

  /**
   * Decode label object.
   *
   * @param value the value
   * @return the object
   */
  default Object decodeLabel(double value) {
    return getEncoderPair().decodeLabel(value);
  }

  /**
   * Encode feature double.
   *
   * @param feature the feature
   * @return the double
   */
  default double encodeFeature(Object feature) {
    return getEncoderPair().encodeFeature(feature);
  }

  /**
   * Decode feature object.
   *
   * @param value the value
   * @return the object
   */
  default Object decodeFeature(double value) {
    return getEncoderPair().decodeFeature(value);
  }

  /**
   * Finish training.
   */
  default void finishTraining() {
    getEncoderPair().freeze();
  }

}// END OF Model
