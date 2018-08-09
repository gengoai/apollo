package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.json.JsonEntry;
import com.gengoai.math.EnhancedDoubleStatistics;
import com.gengoai.stream.MStream;
import com.gengoai.stream.accumulator.MStatisticsAccumulator;
import com.gengoai.string.StringUtils;

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

   public static ZScoreTransform fromJson(JsonEntry entry) {
      ZScoreTransform transform = new ZScoreTransform(
         entry.getStringProperty("restriction", null)
      );
      transform.mean = entry.getDoubleProperty("mean");
      transform.standardDeviation = entry.getDoubleProperty("stddev");
      return transform;
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry object = JsonEntry.object();
      if (!applyToAll()) {
         object.addProperty("restriction", getRestriction());
      }
      object.addProperty("mean", mean);
      object.addProperty("stddev", standardDeviation);
      return object;
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


}// END OF ZScoreTransform
