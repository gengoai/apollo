package com.gengoai.apollo.ml.preprocess.filter;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.collection.counter.Counters;
import com.gengoai.json.JsonEntry;
import com.gengoai.stream.MStream;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * <p>Keeps only the top <code>N</code> most occurring features.</p>
 *
 * @author David B. Bracewell
 */
public class TopNFilter extends RestrictedInstancePreprocessor implements FilterProcessor<Instance>, Serializable {
   private static final long serialVersionUID = 1L;
   private int topN;
   private volatile Set<String> selectedFeatures = Collections.emptySet();

   /**
    * Instantiates a new Top N filter.
    *
    * @param featurePrefix the feature prefix to restrict the filter to
    * @param topN          the number of features to keep
    */
   public TopNFilter(@NonNull String featurePrefix, int topN) {
      super(featurePrefix);
      this.topN = topN;
   }

   /**
    * Instantiates a new Top N  filter with no restriction.
    *
    * @param topN the number of features to keep
    */
   public TopNFilter(int topN) {
      this.topN = topN;
   }

   /**
    * Instantiates a new Top N filter.
    */
   protected TopNFilter() {
      this.topN = 100_000_000;
   }

   @Override
   public String describe() {
      if (applyToAll()) {
         return "TopNFilter{minCount=" + topN + "}";
      }
      return "TopNFilter[" + getRestriction() + "]{minCount=" + topN + "}";
   }


   public static TopNFilter fromJson(JsonEntry entry) {
      TopNFilter filter = new TopNFilter(entry.getStringProperty("restriction", null),
                                         entry.getIntProperty("topN"));
      filter.selectedFeatures.addAll(entry.getProperty("selected").getAsArray(String.class));
      return filter;
   }


   @Override
   public void reset() {
      selectedFeatures.clear();
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      selectedFeatures = new HashSet<>(Counters.newCounter(stream.flatMap(l -> l.stream().map(Feature::getFeatureName))
                                                                 .countByValue())
                                               .topN(topN).items());
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
      object.addProperty("topN", topN);
      object.addProperty("selected", selectedFeatures);
      return object;
   }

}// END OF TopNFilter
