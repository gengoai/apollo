package com.davidbracewell.apollo.ml;

import java.io.Serializable;

/**
 * The type Encoder pair.
 *
 * @author David B. Bracewell
 */
public class EncoderPair implements Serializable {
  private final Encoder labelEncoder;
  private final Encoder featureEncoder;

  /**
   * Instantiates a new Encoder pair.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   */
  public EncoderPair(Encoder labelEncoder, Encoder featureEncoder) {
    this.labelEncoder = labelEncoder;
    this.featureEncoder = featureEncoder;
  }

  /**
   * Encode label double.
   *
   * @param label the label
   * @return the double
   */
  public double encodeLabel(Object label) {
    return labelEncoder.encode(label);
  }

  /**
   * Decode label object.
   *
   * @param value the value
   * @return the object
   */
  public Object decodeLabel(double value) {
    return labelEncoder.decode(value);
  }

  /**
   * Encode feature double.
   *
   * @param feature the feature
   * @return the double
   */
  public double encodeFeature(Object feature) {
    return featureEncoder.encode(feature);
  }

  /**
   * Decode feature object.
   *
   * @param value the value
   * @return the object
   */
  public Object decodeFeature(double value) {
    return featureEncoder.decode(value);
  }

  /**
   * Number of features int.
   *
   * @return the int
   */
  public int numberOfFeatures() {
    return featureEncoder.size();
  }

  /**
   * Number of labels int.
   *
   * @return the int
   */
  public int numberOfLabels() {
    return labelEncoder.size();
  }

  /**
   * Gets label encoder.
   *
   * @return the label encoder
   */
  public Encoder getLabelEncoder() {
    return labelEncoder;
  }

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  public Encoder getFeatureEncoder() {
    return featureEncoder;
  }
}// END OF EncoderPair
