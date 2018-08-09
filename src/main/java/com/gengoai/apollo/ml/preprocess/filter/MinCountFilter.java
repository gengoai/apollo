package com.gengoai.apollo.ml.preprocess.filter;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.json.JsonEntry;
import com.gengoai.stream.MStream;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * <p>Removes features that occur in less than a given number of instances.</p>
 *
 * @author David B. Bracewell
 */
public class MinCountFilter extends RestrictedInstancePreprocessor implements FilterProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private long minCount;
   private volatile Set<String> selectedFeatures = Collections.emptySet();

   /**
    * Instantiates a new Min count filter.
    *
    * @param featurePrefix the feature prefix to restrict the filter to
    * @param minCount      the minimum number of instances a feature must be present in to be kept
    */
   public MinCountFilter(String featurePrefix, long minCount) {
      super(featurePrefix);
      this.minCount = minCount;
   }

   /**
    * Instantiates a new Min count filter with no restriction.
    *
    * @param minCount the minimum number of instances a feature must be present in to be kept
    */
   public MinCountFilter(long minCount) {
      this.minCount = minCount;
   }

   /**
    * Instantiates a new Min count filter.
    */
   protected MinCountFilter() {
      this.minCount = 0;
   }

   @Override
   public String describe() {
      if (applyToAll()) {
         return "MinCountFilter{minCount=" + minCount + "}";
      }
      return "MinCountFilter[" + getRestriction() + "]{minCount=" + minCount + "}";
   }

   public static MinCountFilter fromJson(JsonEntry entry) {
      MinCountFilter filter = new MinCountFilter(entry.getStringProperty("restriction", null),
                                                 entry.getLongProperty("minCount"));
      filter.selectedFeatures.addAll(entry.getProperty("selected").asArray(String.class));
      return filter;
   }

   @Override
   public void reset() {
      selectedFeatures.clear();
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      Counter<String> featureCounts = Counters.newCounter(stream.flatMap(l -> l.stream().map(Feature::getFeatureName))
                                                                .countByValue());
      selectedFeatures = new HashSet<>(featureCounts
                                          .filterByValue(v -> v >= minCount)
                                          .items());
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.filter(f -> selectedFeatures.contains(f.getFeatureName()));
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry object = JsonEntry.object();
      if (!applyToAll()) {
         object.addProperty("restriction", getRestriction());
      }
      object.addProperty("minCount", minCount);
      object.addProperty("selected", selectedFeatures);
      return object;
   }
}// END OF MinCountFilter
