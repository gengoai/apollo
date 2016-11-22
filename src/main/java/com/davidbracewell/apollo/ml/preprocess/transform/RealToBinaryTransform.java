package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

/**
 * The type Real to binary transform.
 *
 * @author David B. Bracewell
 */
public class RealToBinaryTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private double threshold;

   /**
    * Instantiates a new Real to binary transform.
    *
    * @param threshold the threshold
    */
   public RealToBinaryTransform(double threshold) {
      this.threshold = threshold;
   }

   /**
    * Instantiates a new Real to binary transform.
    *
    * @param featureNamePrefix the feature name prefix
    * @param threshold         the threshold
    */
   public RealToBinaryTransform(@NonNull String featureNamePrefix, double threshold) {
      super(featureNamePrefix);
      this.threshold = threshold;
   }

   protected RealToBinaryTransform() {

   }


   @Override
   public void reset() {
   }

   @Override
   public String describe() {
      if (applyToAll()) {
         return "RealToBinaryTransform{threshold=" + threshold + "}";
      }
      return "RealToBinaryTransform[" + getRestriction() + "]{threshold=" + threshold + "}";
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {

   }

   @Override
   public boolean requiresFit() {
      return false;
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.filter(f -> f.getValue() >= threshold).map(feature -> Feature.TRUE(feature.getName()));
   }


   @Override
   public void write(@NonNull StructuredWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.writeKeyValue("restriction", getRestriction());
      }
      writer.writeKeyValue("threshold", threshold);
   }

   @Override
   public void read(@NonNull StructuredReader reader) throws IOException {
      reset();
      while (reader.peek() != ElementType.END_OBJECT) {
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

}// END OF RealToBinaryTransform
