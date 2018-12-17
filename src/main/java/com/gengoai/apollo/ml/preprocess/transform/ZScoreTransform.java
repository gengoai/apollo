package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.preprocess.RestrictedFeaturePreprocessor;
import com.gengoai.math.EnhancedDoubleStatistics;
import com.gengoai.stream.MStream;

/**
 * <p>Transforms features values to Z-Scores.</p>
 *
 * @author David B. Bracewell
 */
public class ZScoreTransform extends RestrictedFeaturePreprocessor implements TransformProcessor {
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
   public String describe() {
      return "ZScoreTransform[" + getRestriction() + "]{mean=" + mean + ", std=" + standardDeviation + "}";
   }

   @Override
   public void reset() {
      mean = 0;
      standardDeviation = 0;
   }

   @Override
   public boolean requiresFit() {
      return true;
   }

   @Override
   protected void fitImpl(MStream<Feature> stream) {
      EnhancedDoubleStatistics stats = stream.mapToDouble(f -> f.value).statistics();
      this.mean = stats.getAverage();
      this.standardDeviation = stats.getSampleStandardDeviation();
   }

   @Override
   public Feature preprocess(Feature in) {
      if (standardDeviation == 0) {
         return Feature.realFeature(in.name, mean);
      }
      return Feature.realFeature(in.name, (in.value - mean) / standardDeviation);
   }
}//END OF ZScoreTransform
