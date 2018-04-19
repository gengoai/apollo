package com.gengoai.apollo.ml.preprocess.filter;


import com.gengoai.apollo.stat.measure.Association;

/**
 * <p>Feature selection using the x2 statistic.</p>
 *
 * @author David B. Bracewell
 */
public class Chi2FeatureSelection extends ContingencyFeatureSelection {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new x2 feature selection.
    *
    * @param numberOfFeaturesPerClass the number of features per class
    * @param threshold                the threshold
    */
   public Chi2FeatureSelection(int numberOfFeaturesPerClass, double threshold) {
      super(Association.CHI_SQUARE, numberOfFeaturesPerClass, threshold);
   }

   /**
    * Instantiates a new x2 feature selection.
    *
    * @param numberOfFeaturesPerClass the number of features per class
    */
   public Chi2FeatureSelection(int numberOfFeaturesPerClass) {
      super(Association.CHI_SQUARE, numberOfFeaturesPerClass, Double.NEGATIVE_INFINITY);
   }

   /**
    * Instantiates a new x2 feature selection.
    */
   protected Chi2FeatureSelection() {
      super(Association.CHI_SQUARE, Integer.MAX_VALUE, Double.NEGATIVE_INFINITY);
   }

   @Override
   public String describe() {
      return getClass().getSimpleName() + "{numberOfFeaturesPerClass=" + getNumFeaturesPerClass() + ", threshold=" + getThreshold() + "}";
   }

}// END OF Chi2FeatureSelection
