package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.json.JsonReader;
import com.davidbracewell.json.JsonTokenType;
import com.davidbracewell.json.JsonWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.IOException;
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
   public RealToBinaryTransform(@NonNull String featureNamePrefix, double threshold) {
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

   @Override
   public void fromJson(@NonNull JsonReader reader) throws IOException {
      reset();
      while (reader.peek() != JsonTokenType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "threshold":
               this.threshold = reader.nextKeyValue().v2.asDoubleValue();
               break;
         }
      }
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

   @Override
   public void toJson(@NonNull JsonWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.property("restriction", getRestriction());
      }
      writer.property("threshold", threshold);
   }

}// END OF RealToBinaryTransform
