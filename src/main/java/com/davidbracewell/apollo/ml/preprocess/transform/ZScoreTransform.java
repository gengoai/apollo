package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.EnhancedDoubleStatistics;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.json.JsonReader;
import com.davidbracewell.json.JsonTokenType;
import com.davidbracewell.json.JsonWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MStatisticsAccumulator;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

/**
 * <p>Transforms features values to Z-Scores.</p>
 *
 * @author David B. Bracewell
 */
public class ZScoreTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {

   private static final long serialVersionUID = 1L;
   private double mean = 0;
   private double standardDeviation = 0;

   /**
    * Instantiates a new Z-Score transform calculating statistics for all features.
    */
   public ZScoreTransform() {
      super(StringUtils.EMPTY);
   }

   /**
    * Instantiates a new Z-Score transform.
    *
    * @param featureNamePrefix the feature name prefix to restrict the transformation to
    */
   public ZScoreTransform(String featureNamePrefix) {
      super(featureNamePrefix);
   }

   @Override
   public String describe() {
      if (applyToAll()) {
         return "ZScoreTransform{mean=" + mean + ", std=" + standardDeviation + "}";
      }
      return "ZScoreTransform[" + getRestriction() + "]{mean=" + mean + ", std=" + standardDeviation + "}";
   }

   @Override
   public void fromJson(@NonNull JsonReader reader) throws IOException {
      reset();
      while (reader.peek() != JsonTokenType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "mean":
               this.mean = reader.nextKeyValue().v2.asDoubleValue();
               break;
            case "stddev":
               this.standardDeviation = reader.nextKeyValue().v2.asDoubleValue();
               break;
         }
      }
   }

   @Override
   public void reset() {
      this.mean = 0;
      this.standardDeviation = 0;
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      MStatisticsAccumulator stats = stream.getContext().statisticsAccumulator();
      stream.forEach(instance -> stats.combine(instance.stream()
                                                       .mapToDouble(Feature::getValue)
                                                       .collect(EnhancedDoubleStatistics::new,
                                                                EnhancedDoubleStatistics::accept,
                                                                EnhancedDoubleStatistics::combine)));
      this.mean = stats.value().getAverage();
      this.standardDeviation = stats.value().getSampleStandardDeviation();
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.map(feature -> Feature.real(feature.getFeatureName(),
                                                       (feature.getValue() - mean) / standardDeviation));
   }

   @Override
   public void toJson(@NonNull JsonWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.property("restriction", getRestriction());
      }
      writer.property("mean", mean);
      writer.property("stddev", standardDeviation);
   }

}// END OF ZScoreTransform
