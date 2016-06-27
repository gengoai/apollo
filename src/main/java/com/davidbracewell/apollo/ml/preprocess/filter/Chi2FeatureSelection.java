package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.stats.ContingencyMeasures;

/**
 * The type Chi squared feature selection.
 *
 * @author David B. Bracewell
 */
public class Chi2FeatureSelection extends ContingencyFeatureSelection {
  private static final long serialVersionUID = 1L;

  /**
   * Instantiates a new Chi squared feature selection.
   *
   * @param numberOfFeaturesPerClass the number of features per class
   * @param threshold                the threshold
   */
  public Chi2FeatureSelection(int numberOfFeaturesPerClass, double threshold) {
    super(ContingencyMeasures.CHI_SQUARE, numberOfFeaturesPerClass, threshold);
  }

  /**
   * Instantiates a new Chi squared feature selection.
   *
   * @param numberOfFeaturesPerClass the number of features per class
   */
  public Chi2FeatureSelection(int numberOfFeaturesPerClass) {
    super(ContingencyMeasures.CHI_SQUARE, numberOfFeaturesPerClass, Double.NEGATIVE_INFINITY);
  }

}// END OF Chi2FeatureSelection
