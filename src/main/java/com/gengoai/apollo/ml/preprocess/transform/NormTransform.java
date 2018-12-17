package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.preprocess.RestrictedFeaturePreprocessor;
import com.gengoai.math.EnhancedDoubleStatistics;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;

/**
 * <p>Transforms features values to Z-Scores.</p>
 *
 * @author David B. Bracewell
 */
public class NormTransform extends RestrictedFeaturePreprocessor implements TransformProcessor {
   private static final long serialVersionUID = 1L;

   private double min = Double.POSITIVE_INFINITY;
   private double max = Double.NEGATIVE_INFINITY;

   /**
    * Instantiates a new Z score transform.
    */
   public NormTransform() {
      super(null);
   }

   /**
    * Instantiates a new Z score transform.
    *
    * @param featureNamePrefix the feature name prefix
    */
   public NormTransform(String featureNamePrefix) {
      super(featureNamePrefix);
   }

   @Override
   public String describe() {
      return "ZScoreTransform[" + getRestriction() + "]{min=" + min + ", max=" + max + "}";
   }

   @Override
   public void reset() {
      min = 0;
      max = 0;
   }

   @Override
   public boolean requiresFit() {
      return true;
   }

   @Override
   protected void fitImpl(MStream<Feature> stream) {
      EnhancedDoubleStatistics stats = stream.mapToDouble(f -> f.value).statistics();
      this.min = stats.getMin();
      this.max = stats.getMax();
   }

   @Override
   public Feature preprocess(Feature in) {
      return Feature.realFeature(in.name, Math2.rescale(in.value, min, max, 0, 1.0));
   }
}//END OF ZScoreTransform
