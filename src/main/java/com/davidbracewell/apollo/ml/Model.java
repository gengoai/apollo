package com.davidbracewell.apollo.ml;

import com.davidbracewell.io.resource.Resource;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The type Model.
 *
 * @author David B. Bracewell
 */
public abstract class Model implements Serializable {
  private static final long serialVersionUID = 1L;
  private final EncoderPair encoderPair;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   */
  public Model(@NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder) {
    this.encoderPair = new EncoderPair(labelEncoder, featureEncoder);
  }

  /**
   * Instantiates a new Model.
   *
   * @param encoderPair the encoder pair
   */
  public Model(@NonNull EncoderPair encoderPair) {
    this.encoderPair = encoderPair;
  }

  /**
   * Read model classifier.
   *
   * @param <T>           the type parameter
   * @param modelResource the model resource
   * @return the classifier
   * @throws Exception the exception
   */
  public static <T extends Model> T readModel(@NonNull Resource modelResource) throws Exception {
    return modelResource.readObject();
  }

  /**
   * Number of labels int.
   *
   * @return the int
   */
  public int numberOfLabels() {
    return encoderPair.numberOfLabels();
  }

  /**
   * Number of features int.
   *
   * @return the int
   */
  public int numberOfFeatures() {
    return encoderPair.numberOfFeatures();
  }

  /**
   * Gets label encoder.
   *
   * @return the label encoder
   */
  public Encoder getLabelEncoder() {
    return encoderPair.getLabelEncoder();
  }

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  public Encoder getFeatureEncoder() {
    return encoderPair.getFeatureEncoder();
  }

  /**
   * Gets encoder pair.
   *
   * @return the encoder pair
   */
  public EncoderPair getEncoderPair() {
    return encoderPair;
  }

  /**
   * Write model.
   *
   * @param modelResource the model resource
   * @throws Exception the exception
   */
  public void writeModel(@NonNull Resource modelResource) throws Exception {
    modelResource.setIsCompressed(true).writeObject(this);
  }

  /**
   * Encode label double.
   *
   * @param label the label
   * @return the double
   */
  public double encodeLabel(Object label) {
    return encoderPair.encodeLabel(label);
  }

  /**
   * Decode label object.
   *
   * @param value the value
   * @return the object
   */
  public Object decodeLabel(double value) {
    return encoderPair.decodeLabel(value);
  }

  /**
   * Encode feature double.
   *
   * @param feature the feature
   * @return the double
   */
  public double encodeFeature(Object feature) {
    return encoderPair.encodeFeature(feature);
  }

  /**
   * Decode feature object.
   *
   * @param value the value
   * @return the object
   */
  public Object decodeFeature(double value) {
    return encoderPair.decodeFeature(value);
  }

  /**
   * Finish training.
   */
  protected void finishTraining() {
    encoderPair.freeze();
  }

}// END OF Model
