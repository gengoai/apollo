package com.davidbracewell.apollo.ml;

import lombok.NonNull;

import java.io.Serializable;

/**
 * The type Feature set.
 *
 * @author David B. Bracewell
 */
public class FeatureSet implements Serializable {
  private static final long serialVersionUID = 1L;
  private final Encoder labelEncoder;
  private final Encoder featureEncoder;

  private FeatureSet(Encoder labelEncoder, Encoder featureEncoder) {
    this.labelEncoder = labelEncoder;
    this.featureEncoder = featureEncoder;
  }

  /**
   * Classification feature set.
   *
   * @param featureEncoder the feature encoder
   * @return the feature set
   */
  public static FeatureSet classification(@NonNull Encoder featureEncoder) {
    return new FeatureSet(new IndexEncoder(), featureEncoder);
  }

  /**
   * Sequence labeling feature set.
   *
   * @param featureEncoder the feature encoder
   * @return the feature set
   */
  public static FeatureSet sequenceLabeling(@NonNull Encoder featureEncoder) {
    return new FeatureSet(new IndexEncoder(), featureEncoder);
  }

  /**
   * Regression feature set.
   *
   * @param featureEncoder the feature encoder
   * @return the feature set
   */
  public static FeatureSet regression(@NonNull Encoder featureEncoder) {
    return new FeatureSet(new RealEncoder(), featureEncoder);
  }


  /**
   * Encode feature double.
   *
   * @param name the name
   * @return the double
   */
  public double encodeFeature(Object name) {
    return featureEncoder.encode(name);
  }

  /**
   * Encode label double.
   *
   * @param name the name
   * @return the double
   */
  public double encodeLabel(Object name) {
    return featureEncoder.encode(name);
  }

  /**
   * Decode feature object.
   *
   * @param id the id
   * @return the object
   */
  public Object decodeFeature(double id) {
    return featureEncoder.decode(id);
  }

  /**
   * Decode label object.
   *
   * @param id the id
   * @return the object
   */
  public Object decodeLabel(double id) {
    return labelEncoder.decode(id);
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

}// END OF FeatureSet
