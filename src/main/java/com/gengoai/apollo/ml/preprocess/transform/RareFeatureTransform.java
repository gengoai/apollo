package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.stream.MStream;

import java.io.Serializable;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class RareFeatureTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
   private Set<String> rareWords = new HashSet<>();
   private String marker = "<UNK>";
   private int minCount;

   @Override
   public String describe() {
      return null;
   }

   @Override
   public void reset() {
      rareWords.clear();
   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
      Counter<String> featureCounts = Counters.newCounter(stream.flatMap(l -> l.stream().map(Feature::getFeatureName))
                                                                .countByValue());
      rareWords = new HashSet<>(featureCounts
                                   .filterByValue(v -> v <= minCount)
                                   .items());
   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return featureStream.map(feature -> {
         if (rareWords.contains(feature.getFeatureName())) {
            return Feature.real(marker, feature.getValue());
         }
         return feature;
      });
   }
}//END OF RareFeatureTransform
