package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.stream.MStream;

import java.util.Optional;

/**
 * <p>Converts a real value feature into a binary feature by converting values greater than or equal to a given
 * threshold "true" and others "false"</p>
 *
 * @author David B. Bracewell
 */
public class RealToBinaryTransform extends RestrictedFeaturePreprocessor implements InstancePreprocessor {
   private static final long serialVersionUID = 1L;
   private double threshold;

   /**
    * Instantiates a new Real to binary transform with no feature restriction.
    *
    * @param threshold the threshold with which a feature value must be <codE>>=</codE> to become a binary "true"
    */
   public RealToBinaryTransform(double threshold) {
      this(null, threshold);
   }

   /**
    * Instantiates a new Real to binary transform.
    *
    * @param featureNamePrefix the feature name prefix to restrict
    * @param threshold         the threshold with which a feature value must be <codE>>=</codE> to become a binary
    *                          "true"
    */
   public RealToBinaryTransform(String featureNamePrefix, double threshold) {
      super(featureNamePrefix);
      this.threshold = threshold;
   }


   @Override
   public Instance applyInstance(Instance example) {
      return example.mapFeatures(f -> {
         if (requiresProcessing(f)) {
            return Optional.ofNullable(f.value >= threshold
                                       ? Feature.booleanFeature(f.name)
                                       : null);
         }
         return Optional.of(f);
      });
   }

   @Override
   protected void fitFeatures(MStream<Feature> exampleStream) {

   }

   @Override
   public void reset() {

   }

   @Override
   public String toString() {
      return "RealToBinaryTransform[" + getRestriction() + "]{threshold=" + threshold + "}";
   }

}//END OF RealToBinaryTransform
