package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.SparseVector2;
import lombok.NonNull;

/**
 * The type Feature vector.
 *
 * @author David B. Bracewell
 */
public class FeatureVector extends SparseVector {
  private final Encoder featureEncoder;
  private double label = Double.NaN;

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  public Encoder getFeatureEncoder() {
    return featureEncoder;
  }

  /**
   * Instantiates a new Feature vector.
   *
   * @param featureEncoder the feature encoder
   */
  public FeatureVector(@NonNull Encoder featureEncoder) {
    super(0);
    this.featureEncoder = featureEncoder;
  }

  @Override
  public int size() {
    return featureEncoder.size();
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
   * Gets label.
   *
   * @return the label
   */
  public double getLabel() {
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
   * Set boolean.
   *
   * @param featureName  the feature name
   * @param featureValue the feature value
   * @return the boolean
   */
  public boolean set(String featureName, double featureValue) {
    int index = (int) featureEncoder.encode(featureName);
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
   * Transform feature vector.
   *
   * @param newFeatureEncoder the new feature encoder
   * @return the feature vector
   */
  public FeatureVector transform(@NonNull Encoder newFeatureEncoder) {
    FeatureVector newVector = new FeatureVector(newFeatureEncoder);
    forEachSparse(entry ->
      newVector.set(
        featureEncoder.decode(entry.getIndex()).toString(),
        entry.getValue()
      )
    );
    return newVector;
  }


}// END OF FeatureVector
