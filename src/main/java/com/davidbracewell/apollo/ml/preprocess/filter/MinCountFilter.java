package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.json.JsonReader;
import com.davidbracewell.json.JsonTokenType;
import com.davidbracewell.json.JsonWriter;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.io.IOException;
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
   public MinCountFilter(@NonNull String featurePrefix, long minCount) {
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

   @Override
   public void fromJson(@NonNull JsonReader reader) throws IOException {
      reset();
      while (reader.peek() != JsonTokenType.END_OBJECT) {
         switch (reader.peekName()) {
            case "restriction":
               setRestriction(reader.nextKeyValue().v2.asString());
               break;
            case "minCount":
               this.minCount = reader.nextKeyValue().v2.asIntegerValue();
               break;
            case "selected":
               reader.nextCollection(HashSet::new).forEach(val -> selectedFeatures.add(val.asString()));
               break;
         }
      }
   }

   @Override
   public void reset() {
      selectedFeatures.clear();
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      selectedFeatures = new HashSet<>(Counters.newCounter(stream.flatMap(l -> l.stream().map(Feature::getFeatureName))
                                                                 .countByValue())
                                               .filterByValue(v -> v >= minCount)
                                               .items());
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.filter(f -> selectedFeatures.contains(f.getFeatureName()));
   }

   @Override
   public void toJson(@NonNull JsonWriter writer) throws IOException {
      if (!applyToAll()) {
         writer.property("restriction", getRestriction());
      }
      writer.property("minCount", minCount);
      writer.property("selected", selectedFeatures);
   }
}// END OF MinCountFilter
