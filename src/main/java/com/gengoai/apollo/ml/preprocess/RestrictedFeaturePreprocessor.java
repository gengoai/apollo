package com.gengoai.apollo.ml.preprocess;

import com.gengoai.Copyable;
import com.gengoai.apollo.ml.Dataset;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.collection.Streams;
import com.gengoai.stream.MStream;
import com.gengoai.string.Strings;

/**
 * @author David B. Bracewell
 */
public abstract class RestrictedFeaturePreprocessor implements FeaturePreprocessor {
   private static final long serialVersionUID = 1L;
   private final String featureNamePrefix;
   private final boolean acceptAll;

   protected RestrictedFeaturePreprocessor(String featureNamePrefix) {
      this.featureNamePrefix = Strings.isNullOrBlank(featureNamePrefix) ? null : featureNamePrefix;
      this.acceptAll = Strings.isNullOrBlank(featureNamePrefix);
   }

   @Override
   public void fit(Dataset dataset) {
      if (requiresFit()) {
         fitImpl(dataset.stream().flatMap(Streams::asStream).flatMap(e -> e.getFeatures().stream()));
      }
   }

   @Override
   public Preprocessor copy() {
      return Copyable.copy(this);
   }

   public final String getRestriction() {
      return Strings.isNullOrBlank(featureNamePrefix) ? "*" : featureNamePrefix;
   }


   protected abstract void fitImpl(MStream<Feature> exampleStream);

   @Override
   public final Feature apply(Feature in) {
      return FeaturePreprocessor.super.apply(in);
   }

   @Override
   public boolean requirePreprocessing(Feature f) {
      return acceptAll || f.name.startsWith(featureNamePrefix);
   }

   @Override
   public String toString() {
      return describe();
   }

}//END OF RestrictedFeaturePreprocessor
