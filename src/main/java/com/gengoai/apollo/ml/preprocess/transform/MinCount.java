package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.preprocess.RestrictedFeaturePreprocessor;
import com.gengoai.stream.MStream;
import com.gengoai.string.Strings;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class MinCount extends RestrictedFeaturePreprocessor implements TransformProcessor {
   private String unknownFeature;
   private int minCount;
   private final Set<String> toKeep = new HashSet<>();

   public MinCount(int minCount) {
      this(null, minCount, null);
   }

   public MinCount(int minCount, String unknownFeature) {
      this(null, minCount, unknownFeature);
   }


   public MinCount(String featureNamePrefix, int minCount) {
      this(featureNamePrefix, minCount, null);
   }

   public MinCount(String featureNamePrefix, int minCount, String unknownFeature) {
      super(featureNamePrefix);
      this.minCount = minCount;
      this.unknownFeature = unknownFeature;
   }

   @Override
   public String describe() {
      return "MinCount[" + getRestriction() + "]{minCount=" + minCount + ", numFeatures=" + toKeep.size() + "}";
   }

   @Override
   public void reset() {
      toKeep.clear();
   }

   @Override
   protected void fitImpl(MStream<Feature> exampleStream) {
      reset();
      Map<String, Long> m = exampleStream.map(f -> f.name).countByValue();
      m.values().removeIf(v -> v < minCount);
      this.toKeep.addAll(m.keySet());
   }

   @Override
   public Feature preprocess(Feature in) {
      if (toKeep.contains(in.name)) {
         return in.copy();
      } else if (!Strings.isNullOrBlank(unknownFeature)) {
         return Feature.realFeature(unknownFeature, in.value);
      }
      return null;
   }

}//END OF MinCount
