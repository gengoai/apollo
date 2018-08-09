package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.json.JsonEntry;
import com.gengoai.stream.MStream;
import com.gengoai.string.StringUtils;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

/**
 * <p>Converts a real value feature into a binary feature by converting values greater than or equal to a given
 * threshold "true" and others "false"</p>
 *
 * @author David B. Bracewell
 */
public class RealToBinaryTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private double threshold;

   /**
    * Instantiates a new Real to binary transform with no feature restriction.
    *
    * @param threshold the threshold with which a feature value must be <codE>>=</codE> to become a binary "true"
    */
   public RealToBinaryTransform(double threshold) {
      this.threshold = threshold;
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

   protected RealToBinaryTransform() {
      this(StringUtils.EMPTY, 0);
   }

   @Override
   public String describe() {
      if (applyToAll()) {
         return "RealToBinaryTransform{threshold=" + threshold + "}";
      }
      return "RealToBinaryTransform[" + getRestriction() + "]{threshold=" + threshold + "}";
   }

   public static RealToBinaryTransform fromJson(JsonEntry entry) {
      return new RealToBinaryTransform(
         entry.getStringProperty("restriction", null),
         entry.getDoubleProperty("threshold")
      );
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry object = JsonEntry.object();
      if (!applyToAll()) {
         object.addProperty("restriction", getRestriction());
      }
      object.addProperty("threshold", threshold);
      return object;
   }

   @Override
   public boolean requiresFit() {
      return false;
   }

   @Override
   public void reset() {
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {

   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.filter(f -> f.getValue() >= threshold).map(
         feature -> Feature.TRUE(feature.getFeatureName()));
   }


}// END OF RealToBinaryTransform
