package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.math.EnhancedDoubleStatistics;
import com.gengoai.stream.MStream;

import java.util.Optional;

/**
 * <p>Transforms features values to Z-Scores.</p>
 *
 * @author David B. Bracewell
 */
public class ZScoreTransform extends RestrictedFeaturePreprocessor {
   private static final long serialVersionUID = 1L;

   private double mean = 0;
   private double standardDeviation = 0;

   /**
    * Instantiates a new Z score transform.
    */
   public ZScoreTransform() {
      super(null);
   }

   /**
    * Instantiates a new Z score transform.
    *
    * @param featureNamePrefix the feature name prefix
    */
   public ZScoreTransform(String featureNamePrefix) {
      super(featureNamePrefix);
   }

   @Override
   public Instance applyInstance(Instance example) {
      return example.mapFeatures(in -> {
         if (!requiresProcessing(in)) {
            return Optional.of(in);
         }
         if (standardDeviation == 0) {
            return Optional.of(Feature.realFeature(in.name, mean));
         }
         return Optional.of(Feature.realFeature(in.name, (in.value - mean) / standardDeviation));
      });
   }

   @Override
   protected void fitFeatures(MStream<Feature> stream) {
      EnhancedDoubleStatistics stats = stream.mapToDouble(f -> f.value).statistics();
      this.mean = stats.getAverage();
      this.standardDeviation = stats.getSampleStandardDeviation();
   }

   @Override
   public void reset() {
      mean = 0;
      standardDeviation = 0;
   }

   @Override
   public String toString() {
      return "ZScoreTransform[" + getRestriction() + "]{mean=" + mean + ", std=" + standardDeviation + "}";
   }

}//END OF ZScoreTransform
